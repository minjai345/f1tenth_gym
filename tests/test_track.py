import pathlib
import shutil
import tempfile
import time
import unittest
import warnings

import numpy as np
from PIL import Image
from PIL.Image import Transpose

from f1tenth_gym.envs.track import Raceline, Track, find_track_dir
from f1tenth_gym.envs.track.track import TrackSpec, _occupancy_from_image


class TestTrackLoader(unittest.TestCase):
    """Pins ISSUES_PLAN.md #2: the loaders accept every shipped layout and fail loudly."""

    def test_from_track_path_accepts_dir_stem_and_yaml(self):
        ref = Track.from_track_name("Spielberg")
        track_dir = find_track_dir("Spielberg")
        for arg in (track_dir, track_dir / "Spielberg", track_dir / "Spielberg.yaml"):
            track = Track.from_track_path(arg)
            self.assertEqual(track.spec.resolution, ref.spec.resolution, arg)
            self.assertIsNotNone(track.centerline, arg)
            self.assertTrue(np.isclose(track.raceline.xs, ref.raceline.xs).all(), arg)

    def test_from_track_path_legacy_map_yaml(self):
        src = find_track_dir("Spielberg")
        with tempfile.TemporaryDirectory() as tmp:
            tmp = pathlib.Path(tmp)
            shutil.copy(src / "Spielberg.png", tmp / "Legacy.png")
            (tmp / "Legacy_map.yaml").write_text(
                (src / "Spielberg.yaml").read_text().replace("Spielberg.png", "Legacy.png")
            )
            shutil.copy(src / "Spielberg_raceline.csv", tmp / "Legacy_raceline.csv")
            track = Track.from_track_path(tmp / "Legacy_map.yaml")
            # the "_map" suffix must not leak into the stem used for the CSVs
            self.assertIsNotNone(track.raceline)
            self.assertEqual(track.spec.name, "Legacy")

    def test_from_track_path_missing(self):
        self.assertRaises(FileNotFoundError, Track.from_track_path, "maps/i_dont_exist")

    def test_load_spec_missing_required_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec_file = pathlib.Path(tmp) / "bad.yaml"
            spec_file.write_text("image: x.png\norigin: [0, 0, 0]\n")
            with self.assertRaisesRegex(ValueError, "resolution"):
                Track.load_spec(track="bad", filespec=str(spec_file))

    def test_load_spec_rejects_unsupported_semantics(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec_file = pathlib.Path(tmp) / "bad.yaml"
            base = "image: x.png\nresolution: 0.05\norigin: [0, 0, 0]\n"
            spec_file.write_text(base + "mode: raw\n")
            with self.assertRaisesRegex(ValueError, "mode"):
                Track.load_spec(track="bad", filespec=str(spec_file))
            spec_file.write_text(base + "negate: 5\n")
            with self.assertRaisesRegex(ValueError, "negate"):
                Track.load_spec(track="bad", filespec=str(spec_file))
            # a pixel-valued threshold is a common ROS-yaml mistake
            spec_file.write_text(base + "occupied_thresh: 128\n")
            with self.assertRaisesRegex(ValueError, "occupied_thresh"):
                Track.load_spec(track="bad", filespec=str(spec_file))

    def test_load_spec_accepts_negate(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec_file = pathlib.Path(tmp) / "neg.yaml"
            spec_file.write_text("image: x.png\nresolution: 0.05\norigin: [0, 0, 0]\nnegate: 1\n")
            spec = Track.load_spec(track="neg", filespec=str(spec_file))
            self.assertEqual(spec.negate, 1)

    def test_load_spec_ros_defaults_and_unknown_key_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec_file = pathlib.Path(tmp) / "ros.yaml"
            spec_file.write_text(
                "image: x.png\nresolution: 0.05\norigin: [0, 0, 0]\nmode: trinary\nfoo: 1\n"
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                spec = Track.load_spec(track="ros", filespec=str(spec_file))
            self.assertEqual(len(caught), 1)
            self.assertIn("foo", str(caught[0].message))
            self.assertEqual(spec.negate, 0)
            self.assertEqual(spec.occupied_thresh, 0.65)
            self.assertEqual(spec.free_thresh, 0.196)

    def test_from_track_name_raceline_only_falls_back(self):
        src = find_track_dir("Spielberg")
        tmp = src.parent / "Fallback_tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir()
        try:
            shutil.copy(src / "Spielberg.png", tmp / "Fallback_tmp.png")
            (tmp / "Fallback_tmp.yaml").write_text(
                (src / "Spielberg.yaml").read_text().replace("Spielberg.png", "Fallback_tmp.png")
            )
            shutil.copy(src / "Spielberg_raceline.csv", tmp / "Fallback_tmp_raceline.csv")
            track = Track.from_track_name("Fallback_tmp")
            self.assertIsNotNone(track.centerline)
            self.assertIs(track.centerline, track.raceline)

            (tmp / "Fallback_tmp_raceline.csv").unlink()
            with self.assertRaisesRegex(ValueError, "no reference line"):
                Track.from_track_name("Fallback_tmp")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestOccupancySemantics(unittest.TestCase):
    """Pins ISSUES_PLAN.md #1: the YAML thresholds are honoured, ROS map_server style."""

    @staticmethod
    def _spec(**kwargs):
        return TrackSpec(name=None, image=None, resolution=0.05, origin=(0.0, 0.0, 0.0), **kwargs)

    def test_threshold_cut(self):
        # occupied iff (255 - v)/255 > occupied_thresh; at the shipped 0.45 the cut is v <= 140
        gray = np.array([[0, 128, 140, 141, 200, 255]], dtype=np.uint8)
        occ = _occupancy_from_image(Image.fromarray(gray), self._spec(occupied_thresh=0.45))
        np.testing.assert_array_equal(occ, [[0.0, 0.0, 0.0, 255.0, 255.0, 255.0]])

    def test_negate_inverts_the_probability(self):
        gray = np.arange(256, dtype=np.uint8).reshape(16, 16)
        normal = _occupancy_from_image(Image.fromarray(gray), self._spec(occupied_thresh=0.45))
        inverted = _occupancy_from_image(
            Image.fromarray(255 - gray), self._spec(negate=1, occupied_thresh=0.45)
        )
        np.testing.assert_array_equal(normal, inverted)

    def test_non_grayscale_image_is_converted(self):
        rgb = np.zeros((4, 5, 3), dtype=np.uint8)
        occ = _occupancy_from_image(Image.fromarray(rgb), self._spec())
        self.assertEqual(occ.shape, (4, 5))
        np.testing.assert_array_equal(occ, np.zeros((4, 5)))

    def test_legacy_128_recipe(self):
        # occupied_thresh: 0.495 reproduces the retired hard-coded 128 cut exactly
        gray = np.arange(256, dtype=np.uint8).reshape(16, 16)
        legacy = np.where(gray <= 128, 0.0, 255.0).astype(np.float32)
        occ = _occupancy_from_image(Image.fromarray(gray), self._spec(occupied_thresh=0.495))
        np.testing.assert_array_equal(occ, legacy)

    def test_shipped_yaml_thresholds_are_honoured(self):
        # Spielberg declares occupied_thresh: 0.45, so pixels 129..140 are now walls
        track = Track.from_track_name("Spielberg")
        track_dir = find_track_dir("Spielberg")
        image = Image.open(track_dir / "Spielberg.png").transpose(Transpose.FLIP_TOP_BOTTOM)
        gray = np.array(image.convert("L"))
        legacy = np.where(gray <= 128, 0.0, 255.0).astype(np.float32)
        flipped = track.occupancy_map != legacy
        self.assertGreater(flipped.sum(), 0)
        self.assertTrue(((gray[flipped] >= 129) & (gray[flipped] <= 140)).all())
        self.assertTrue((track.occupancy_map[flipped] == 0.0).all())


class TestTrack(unittest.TestCase):
    def test_error_handling(self):
        wrong_track_name = "i_dont_exists"
        self.assertRaises(FileNotFoundError, Track.from_track_name, wrong_track_name)

    def test_raceline(self):
        track_name = "Spielberg"
        track = Track.from_track_name(track_name)

        # check raceline is not None
        self.assertNotEqual(track.raceline, None)

        # check loaded raceline match the one in the csv file
        track_dir = find_track_dir(track_name)
        assert track_dir is not None and track_dir.exists(), "track_dir does not exist"

        raceline = np.loadtxt(track_dir / f"{track_name}_raceline.csv", delimiter=";")
        s_idx, x_idx, y_idx, psi_idx, kappa_idx, vx_idx, ax_idx = range(7)

        self.assertTrue(np.isclose(track.raceline.ss, raceline[:, s_idx]).all())
        self.assertTrue(np.isclose(track.raceline.xs, raceline[:, x_idx]).all())
        self.assertTrue(np.isclose(track.raceline.ys, raceline[:, y_idx]).all())
        self.assertTrue(np.isclose(track.raceline.yaws, raceline[:, psi_idx]).all())
        self.assertTrue(np.isclose(track.raceline.ks, raceline[:, kappa_idx]).all())
        self.assertTrue(np.isclose(track.raceline.vxs, raceline[:, vx_idx]).all())
        self.assertTrue(np.isclose(track.raceline.axs, raceline[:, ax_idx]).all())

    def test_map_dir_structure(self):
        """
        Check that the map dir structure is correct:
        - maps/
            - Trackname/
                - Trackname_map.*               # map image
                - Trackname_map.yaml            # map specification
                - [Trackname_raceline.csv]      # raceline (optional)
                - [Trackname_centerline.csv]    # centerline (optional)
        """
        mapdir = pathlib.Path(__file__).parent.parent / "maps"
        for trackdir in mapdir.iterdir():
            if trackdir.is_file():
                continue

            # check subdir is capitalized (at least first letter is capitalized)
            trackdirname = trackdir.stem
            if "_tmp" in trackdirname.lower():
                continue
            self.assertTrue(
                trackdirname[0].isupper(), f"trackdir {trackdirname} is not capitalized"
            )

            # check map spec file exists (new convention: {track}.yaml)
            file_spec = trackdir / f"{trackdirname}.yaml"
            self.assertTrue(
                file_spec.exists(),
                f"map spec file {file_spec} does not exist in {trackdir}",
            )

            # read map image file from spec
            map_spec = Track.load_spec(track=str(trackdir), filespec=str(file_spec))
            file_image = trackdir / map_spec.image

            # check map image file exists
            self.assertTrue(
                file_image.exists(),
                f"map image file {file_image} does not exist in {trackdir}",
            )

            # check raceline and centerline files
            file_raceline = trackdir / f"{trackdir.stem}_raceline.csv"
            file_centerline = trackdir / f"{trackdir.stem}_centerline.csv"

            if file_raceline.exists():
                # try to load raceline files
                # it will raise an assertion error if the file format are not valid
                Raceline.from_raceline_file(file_raceline)

            if file_centerline.exists():
                # try to load raceline files
                # it will raise an assertion error if the file format are not valid
                Raceline.from_centerline_file(file_centerline)

    def test_download_racetrack(self):
        import shutil

        track_name = "Spielberg"
        track_backup = Track.from_track_name(track_name)

        # rename the track dir
        track_dir = find_track_dir(track_name)
        tmp_dir = track_dir.parent / f"{track_name}_tmp{int(time.time())}"
        track_dir.rename(tmp_dir)
        track_dir.mkdir(parents=True, exist_ok=False)
        shutil.copytree(tmp_dir, track_dir, dirs_exist_ok=True)

        # download the track
        track = Track.from_track_name(track_name)

        # check the two tracks' specs are the same
        for spec_attr in [
            "name",
            "image",
            "resolution",
            "origin",
            "negate",
            "occupied_thresh",
            "free_thresh",
        ]:
            self.assertEqual(
                getattr(track.spec, spec_attr), getattr(track_backup.spec, spec_attr)
            )

        # check the two tracks' racelines are the same
        for raceline_attr in ["ss", "xs", "ys", "yaws", "ks", "vxs", "axs"]:
            self.assertTrue(
                np.isclose(
                    getattr(track.raceline, raceline_attr),
                    getattr(track_backup.raceline, raceline_attr),
                ).all()
            )

        # check the two tracks' centerlines are the same
        for centerline_attr in ["ss", "xs", "ys", "yaws", "ks", "vxs", "axs"]:
            self.assertTrue(
                np.isclose(
                    getattr(track.centerline, centerline_attr),
                    getattr(track_backup.centerline, centerline_attr),
                ).all()
            )

        # remove the newly created track dir
        track_dir = find_track_dir(track_name)
        shutil.rmtree(track_dir, ignore_errors=True)

        # rename the backup track dir to its original name
        track_backup_dir = find_track_dir(tmp_dir.stem)
        track_backup_dir.rename(track_dir)

    def test_frenet_to_cartesian(self):
        track_name = "Spielberg"
        track = Track.from_track_name(track_name)

        # Check frenet to cartesian conversion
        # using the track's xs, ys
        for s, x, y in zip(
            track.centerline.ss, track.centerline.xs, track.centerline.ys
        ):
            x_, y_, _ = track.frenet_to_cartesian(s, 0, 0)
            self.assertAlmostEqual(x, x_, places=4)
            self.assertAlmostEqual(y, y_, places=4)

    def test_frenet_to_cartesian_to_frenet(self):
        track_name = "Spielberg"
        track = Track.from_track_name(track_name)

        # check frenet to cartesian conversion
        s_ = 0
        for s in np.linspace(0, 1, 10):
            x, y, psi = track.frenet_to_cartesian(s, 0, 0)
            s_, d, _ = track.cartesian_to_frenet(x, y, psi, s_guess=s_)
            self.assertAlmostEqual(s, s_, places=4)
            self.assertAlmostEqual(d, 0, places=4)

        # check frenet to cartesian conversion
        # with non-zero lateral offset
        s_ = 0
        for s in np.linspace(0, 1, 10):
            d = np.random.uniform(-1.0, 1.0)
            x, y, psi = track.frenet_to_cartesian(s, d, 0)
            s_, d_, _ = track.cartesian_to_frenet(x, y, psi, s_guess=s_)
            # Handle edge case where we are checking for s=0 but s_ is the last s (same point, but different s)
            self.assertTrue(np.isclose(s, s_, atol=1e-4) or np.isclose(s + track.centerline.spline.s[-1], s_, atol=1e-4))
            self.assertAlmostEqual(d, d_, places=4)


class TestSyntheticGridOrientation(unittest.TestCase):
    """Pins ISSUES_PLAN.md #3's grid bug: synthetic grids are (rows=y, cols=x)."""

    def test_from_refline_grid_is_height_by_width(self):
        track = Track.from_refline(
            x=np.linspace(0, 10, 50), y=np.zeros(50), velx=2.0 * np.ones(50)
        )
        # x-extent 20 m (10 m line + 2*5 m margin), y-extent 10 m, 0.05 m/px:
        # set_map reads height = shape[0], so rows must be the y-extent
        self.assertEqual(track.occupancy_map.shape, (200, 400))
