"""Geometry and consumption contracts for the Phase 6 benchmark fixture."""

import unittest

import numpy as np

from benchmarks.phase6_rollout import (
    _explicit_poses,
    _make_track,
    _numpy_probe,
    _parse_args,
)
from f1tenth_gym.envs.track.walls import wall_segments


class TestBenchmarkTrack(unittest.TestCase):
    def test_sensor_and_contact_fixture_has_real_equal_shape_walls(self):
        first = _make_track(2, 32, center_x=0.0, with_walls=True)
        shifted = _make_track(2, 32, center_x=50.0, with_walls=True)

        first_walls = wall_segments(first)
        shifted_walls = wall_segments(shifted)
        self.assertGreater(len(first_walls), 0)
        self.assertEqual(first.occupancy_map.shape, shifted.occupancy_map.shape)
        self.assertEqual(first_walls.a.shape, shifted_walls.a.shape)
        np.testing.assert_allclose(
            shifted_walls.a - np.array([50.0, 0.0], dtype=np.float32),
            first_walls.a,
            atol=2e-5,
        )

    def test_explicit_centerline_poses_start_in_free_road_cells(self):
        track = _make_track(4, 32, with_walls=True)
        poses = _explicit_poses(track, 4)
        resolution = float(track.spec.resolution)
        origin_x, origin_y = track.spec.origin[:2]
        columns = np.floor((poses[:, 0] - origin_x) / resolution).astype(int)
        rows = np.floor((poses[:, 1] - origin_y) / resolution).astype(int)

        self.assertTrue((track.occupancy_map[rows, columns] == 255.0).all())
        self.assertTrue((track.occupancy_map == 0.0).any())

    def test_state_fixture_remains_wall_free(self):
        track = _make_track(1, 32, with_walls=False)
        self.assertEqual(len(wall_segments(track)), 0)

    def test_mutable_probe_consumes_every_numeric_element(self):
        value = {
            "a": np.array([1.0, 2.0], dtype=np.float32),
            "b": (np.array([3], dtype=np.int32), True),
        }
        self.assertEqual(_numpy_probe(value), 7.0)

    def test_cli_rejects_degenerate_frenet_and_duplicate_pose_layouts(self):
        for arguments in (
            ["--track-points", "11"],
            ["--track-points", "12", "--agents", "13"],
        ):
            with self.subTest(arguments=arguments), self.assertRaises(SystemExit):
                _parse_args(arguments)


if __name__ == "__main__":
    unittest.main()
