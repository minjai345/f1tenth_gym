from __future__ import annotations
import time
import uuid
import pathlib
import warnings
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import yaml
from PIL import Image
from PIL.Image import Transpose

from . import Raceline
from .cubic_spline import CubicSplineND
from .utils import find_track_dir

@dataclass
class TrackSpec:
    """Track specification loaded from YAML config file.

    Attributes:
        name: Track name identifier.
        image: Filename of the occupancy map image.
        resolution: Map resolution in meters/pixel.
        origin: Map origin as (x, y, theta) in world coordinates.
        negate: Whether to negate the image values.
        occupied_thresh: Threshold for occupied cells.
        free_thresh: Threshold for free cells.
    """

    name: Optional[str]
    image: Optional[str]
    resolution: float
    origin: Tuple[float, float, float]
    negate: int = 0
    occupied_thresh: float = 0.65
    free_thresh: float = 0.196


# Keys accepted from a map YAML. `image`/`resolution`/`origin` have no sane
# fallback, so they stay required; the rest take the ROS map_server defaults.
_SPEC_REQUIRED_KEYS = ("image", "resolution", "origin")
_SPEC_OPTIONAL_KEYS = ("negate", "occupied_thresh", "free_thresh", "mode")


def _occupancy_from_image(image: Image.Image, spec: TrackSpec) -> np.ndarray:
    """Binarise a map image following ROS map_server semantics.

    Occupancy probability is ``(255 - pixel) / 255``, or ``pixel / 255`` when
    ``spec.negate``; a cell is an obstacle (0.0) iff that probability exceeds
    ``spec.occupied_thresh``, else free (255.0). The ROS "unknown" band
    (between ``free_thresh`` and ``occupied_thresh``) maps to free: the EDT
    and the ray tracer need a binary world.
    """
    gray = np.array(image.convert("L"), dtype=np.float32)
    occ_prob = gray / 255.0 if spec.negate else (255.0 - gray) / 255.0
    occupancy_map = np.full(gray.shape, 255.0, dtype=np.float32)
    occupancy_map[occ_prob > spec.occupied_thresh] = 0.0
    return occupancy_map


def _grayscale_from_image(image: Image.Image) -> np.ndarray:
    """The un-thresholded greys, kept so wall extraction can read sub-pixel edges."""
    return np.array(image.convert("L"), dtype=np.uint8)


@dataclass
class Track:
    """Racing track with occupancy map and reference lines.

    Provides track geometry, collision detection via occupancy grid,
    and Frenet frame coordinate transformations.

    Attributes:
        spec: Track specification with map metadata.
        filepath: Path to the track files.
        ext: File extension of the map image.
        occupancy_map: 2D occupancy grid for collision detection.
        occupancy_grey: Un-thresholded map greys, or None for synthetic tracks.
        centerline: Track centerline as a Raceline.
        raceline: Optimal racing line as a Raceline.
    """

    spec: TrackSpec
    filepath: Optional[str]
    ext: Optional[str]
    occupancy_map: np.ndarray
    occupancy_grey: Optional[np.ndarray]
    centerline: Raceline
    raceline: Raceline

    def __init__(
        self,
        spec: TrackSpec,
        occupancy_map: np.ndarray,
        filepath: Optional[str] = None,
        ext: Optional[str] = None,
        centerline: Optional[Raceline] = None,
        raceline: Optional[Raceline] = None,
        occupancy_grey: Optional[np.ndarray] = None,
    ):
        """
        Initialize track object.

        Args:
            spec: track specification (TrackSpec).
            filepath: path to the track image (str).
            ext: file extension of the track image file (str).
            occupancy_map: occupancy grid map (np.ndarray).
            centerline: centerline of the track (Raceline), by default None.
            raceline: raceline of the track (Raceline), by default None.
        """
        self.spec = spec
        self.filepath = filepath
        self.ext = ext
        self.occupancy_map = occupancy_map
        self.occupancy_grey = occupancy_grey
        self.centerline = centerline
        self.raceline = raceline
        self.s_guess = None
        self.frenet_search_range = 10 # meters

    @staticmethod
    def load_spec(track: str, filespec: str) -> TrackSpec:
        """
        Load track specification from yaml file.

        Args:
            track: name of the track (str).
            filespec: path to the yaml file (str).

        Returns:
            TrackSpec: track specification.

        Raises:
            ValueError: if a required key is missing, a threshold is out of range,
                or a key has unsupported semantics (``mode: raw``/``scale``).
        """
        with open(filespec, "r") as yaml_stream:
            map_metadata = yaml.safe_load(yaml_stream)
        if not isinstance(map_metadata, dict):
            raise ValueError(f"{filespec}: expected a YAML mapping of map metadata")
        missing = [key for key in _SPEC_REQUIRED_KEYS if key not in map_metadata]
        if missing:
            raise ValueError(f"{filespec}: missing required map key(s): {', '.join(missing)}")
        known = (*_SPEC_REQUIRED_KEYS, *_SPEC_OPTIONAL_KEYS)
        for key in sorted(set(map_metadata) - set(known)):
            warnings.warn(f"{filespec}: ignoring unsupported map key '{key}'")
        metadata = {key: map_metadata[key] for key in known if key in map_metadata}
        mode = metadata.pop("mode", "trinary")
        if mode != "trinary":
            raise ValueError(f"{filespec}: mode '{mode}' is not supported (only 'trinary')")
        if metadata.get("negate", 0) not in (0, 1):
            raise ValueError(f"{filespec}: negate must be 0 or 1")
        if not 0 < metadata.get("occupied_thresh", 0.65) <= 1:
            raise ValueError(f"{filespec}: occupied_thresh must be in (0, 1]")
        if not 0 <= metadata.get("free_thresh", 0.196) < 1:
            raise ValueError(f"{filespec}: free_thresh must be in [0, 1)")
        if not isinstance(metadata["resolution"], (int, float)) or metadata["resolution"] <= 0:
            raise ValueError(f"{filespec}: resolution must be a positive number")
        origin = metadata["origin"]
        if not isinstance(origin, (list, tuple)) or len(origin) != 3:
            raise ValueError(f"{filespec}: origin must be [x, y, theta]")
        return TrackSpec(name=track, **metadata)

    @staticmethod
    def from_track_name(track: str, track_scale: float = 1.0) -> Track:
        """
        Load track from track name.

        Args:
            track: name of the track (str).
            track_scale: scale of the track (float), by default 1.0.

        Returns:
            Track: track object.

        Raises:
            FileNotFoundError: if the track cannot be loaded.
        """
        try:
            track_dir = find_track_dir(track)
            # Try new naming convention first ({track}.yaml), then fall back to old ({track}_map.yaml)
            yaml_path = track_dir / f"{track_dir.stem}.yaml"
            if not yaml_path.exists():
                yaml_path = track_dir / f"{track_dir.stem}_map.yaml"
            if not yaml_path.exists():
                raise FileNotFoundError(
                    f"no map YAML in {track_dir} (tried '{track_dir.stem}.yaml' and '{track_dir.stem}_map.yaml')"
                )
            track_spec = Track.load_spec(
                track=track, filespec=str(yaml_path)
            )
            track_spec.resolution = track_spec.resolution * track_scale
            track_spec.origin = (
                track_spec.origin[0] * track_scale,
                track_spec.origin[1] * track_scale,
                track_spec.origin[2],
            )

            # load occupancy grid
            map_filename = pathlib.Path(track_spec.image)
            image = Image.open(track_dir / str(map_filename)).transpose(
                Transpose.FLIP_TOP_BOTTOM
            )
            occupancy_map = _occupancy_from_image(image, track_spec)
            occupancy_grey = _grayscale_from_image(image)

            # if exists, load centerline
            if (track_dir / f"{track}_centerline.csv").exists():
                centerline = Raceline.from_centerline_file(
                    track_dir / f"{track}_centerline.csv",
                    track_scale=track_scale,
                )
            else:
                centerline = None

            # if exists, load raceline
            if (track_dir / f"{track}_raceline.csv").exists():
                raceline = Raceline.from_raceline_file(
                    track_dir / f"{track}_raceline.csv",
                    track_scale=track_scale,
                )
            else:
                raceline = centerline

            if centerline is None and raceline is None:
                raise ValueError(
                    f"no reference line in {track_dir}: neither '{track}_centerline.csv' "
                    f"nor '{track}_raceline.csv' exists"
                )
            if centerline is None:
                centerline = raceline

            return Track(
                spec=track_spec,
                filepath=str((track_dir / track).absolute()),
                ext=map_filename.suffix,
                occupancy_map=occupancy_map,
                occupancy_grey=occupancy_grey,
                centerline=centerline,
                raceline=raceline,
            )
        except (FileNotFoundError, ValueError, KeyError) as ex:
            raise type(ex)(f"Could not load track '{track}': {ex}") from ex

    @staticmethod
    def from_track_path(path: pathlib.Path, track_scale: float = 1.0) -> Track:
        """
        Load track from a track directory, a path stem, or a map YAML file.

        Args:
            path: a pathlib.Path that is any of: the track directory
                (``maps/Spielberg``), a stem inside it
                (``maps/Spielberg/Spielberg``), or the map YAML itself
                (``maps/Spielberg/Spielberg.yaml``, legacy ``..._map.yaml``).
            track_scale: scale of the track (float), by default 1.0.

        Returns:
            Track: track object.

        Raises:
            FileNotFoundError: if the track cannot be loaded.
        """
        try:
            path = pathlib.Path(path)
            if path.suffix in (".yaml", ".yml"):
                track_dir = path.parent
                stem = path.stem[: -len("_map")] if path.stem.endswith("_map") else path.stem
                candidates = [path]
            else:
                track_dir = path if path.is_dir() else path.parent
                stem = path.name if path.is_dir() else path.stem
                candidates = [
                    track_dir / f"{stem}.yaml",
                    track_dir / f"{stem}_map.yaml",
                ]
            yaml_path = next((c for c in candidates if c.exists()), None)
            if yaml_path is None:
                tried = ", ".join(str(c) for c in candidates)
                raise FileNotFoundError(f"no map YAML for '{path}' (tried: {tried})")

            track_spec = Track.load_spec(track=stem, filespec=str(yaml_path))
            track_spec.resolution = track_spec.resolution * track_scale
            track_spec.origin = (
                track_spec.origin[0] * track_scale,
                track_spec.origin[1] * track_scale,
                track_spec.origin[2],
            )

            # load occupancy grid
            image_path = track_dir / track_spec.image
            image = Image.open(image_path).transpose(Transpose.FLIP_TOP_BOTTOM)
            occupancy_map = _occupancy_from_image(image, track_spec)
            occupancy_grey = _grayscale_from_image(image)

            # if exists, load centerline
            if (track_dir / f"{stem}_centerline.csv").exists():
                centerline = Raceline.from_centerline_file(track_dir / f"{stem}_centerline.csv",
                                                           track_scale=track_scale,)
            else:
                centerline = None

            # if exists, load raceline
            if (track_dir / f"{stem}_raceline.csv").exists():
                raceline = Raceline.from_raceline_file(track_dir / f"{stem}_raceline.csv",
                                                       track_scale=track_scale,)
            else:
                raceline = centerline

            if centerline is None and raceline is None:
                raise ValueError(
                    f"no reference line in {track_dir}: neither '{stem}_centerline.csv' "
                    f"nor '{stem}_raceline.csv' exists"
                )
            if centerline is None:
                centerline = raceline

            return Track(
                spec=track_spec,
                filepath=str((track_dir / stem).absolute()),
                ext=image_path.suffix,
                occupancy_map=occupancy_map,
                occupancy_grey=occupancy_grey,
                centerline=centerline,
                raceline=raceline,
            )
        except (FileNotFoundError, ValueError, KeyError) as ex:
            raise type(ex)(f"Could not load track '{path}': {ex}") from ex

    @staticmethod
    def from_refline(x: np.ndarray, y: np.ndarray, velx: np.ndarray):
        """
        Create an empty track reference line.

        Args:
            x: x-coordinates of the waypoints (np.ndarray).
            y: y-coordinates of the waypoints (np.ndarray).
            velx: velocities at the waypoints (np.ndarray).

        Returns:
            Track: track object.
        """
        ds = 0.1
        resolution = 0.05
        margin = 5.0  # Fixed margin in meters around the track

        spline = CubicSplineND(x=x, y=y, vxs=velx)
        ss, xs, ys, yaws, ks, vxs = spline.ss, spline.xs, spline.ys, spline.psis, spline.ks, spline.vxs

        refline = Raceline(
            ss=np.array(ss).astype(np.float32),
            xs=np.array(xs).astype(np.float32),
            ys=np.array(ys).astype(np.float32),
            psis=np.array(yaws).astype(np.float32),
            kappas=np.array(ks).astype(np.float32),
            velxs=np.array(vxs).astype(np.float32),
            accxs=np.zeros_like(ss).astype(np.float32),
            spline=spline,
        )

        min_x, max_x = np.min(xs), np.max(xs)
        min_y, max_y = np.min(ys), np.max(ys)

        # Calculate map size with fixed margin (handles straight lines gracefully)
        map_width = (max_x - min_x) + 2 * margin
        map_height = (max_y - min_y) + 2 * margin

        # rows are world y (height), columns world x (width) -- set_map reads
        # height = shape[0], width = shape[1]
        occupancy_map = 255.0 * np.ones(
            (
                int(map_height / resolution),
                int(map_width / resolution),
            ),
            dtype=np.float32,
        )
        # origin is the bottom left corner
        origin = (min_x - margin, min_y - margin, 0.0)

        track_spec = TrackSpec(
            name=None,
            image=None,
            resolution=resolution,
            origin=origin,
            negate=False,
            occupied_thresh=0.65,
            free_thresh=0.196,
        )

        return Track(
            spec=track_spec,
            filepath=None,
            ext=None,
            occupancy_map=occupancy_map,
            raceline=refline,
            centerline=refline,
        )
    
    def from_raceline_file(filepath: pathlib.Path, delimiter: str = ";", skip_rows: int = 3, track_scale: float = 1.0) -> Track:
        """
        Creates a Track object from a raceline file of the format [s, x, y, psi, k, vx, ax].
        
        Args:
            filepath (pathlib.Path): path to the raceline file
            delimiter (str, optional): delimiter used in the file. Defaults to ";".
            skip_rows (int, optional): number of rows to skip. Defaults to 3.
            track_scale (float, optional): scale of the track. Defaults to 1.0.
        
        Returns:
            Track: track object
        """
        raceline = Raceline.from_raceline_file(filepath, delimiter, skip_rows, track_scale)
        xs = raceline.xs
        ys = raceline.ys
        resolution = 0.05
        margin_perc = 0.1

        min_x, max_x = np.min(xs), np.max(xs)
        min_y, max_y = np.min(ys), np.max(ys)
        x_range = max_x - min_x
        y_range = max_y - min_y
        # rows are world y (height), columns world x (width), as above
        occupancy_map = 255.0 * np.ones(
            (
                int((1 + 2 * margin_perc) * y_range / resolution),
                int((1 + 2 * margin_perc) * x_range / resolution),
            ),
            dtype=np.float32,
        )
        # origin is the bottom left corner
        origin = (min_x - margin_perc * x_range, min_y - margin_perc * y_range, 0.0)

        track_spec = TrackSpec(
            name=None,
            image=None,
            resolution=resolution,
            origin=origin,
            negate=False,
            occupied_thresh=0.65,
            free_thresh=0.196,
        )

        track_spec = TrackSpec(
            name=None,
            image=None,
            resolution=0.05,
            origin=(0.0, 0.0, 0.0),
            negate=False,
            occupied_thresh=0.65,
            free_thresh=0.196,
        )
        return Track(
            spec=track_spec,
            filepath=None,
            ext=None,
            occupancy_map=occupancy_map,
            raceline=raceline,
            centerline=raceline,
        )

    def save_raceline(self, outdir: pathlib.Path):
        """
        Save track raceline.

        Args:
            outdir: output directory (pathlib.Path).
        """
        raceline_filepath = outdir / f"{self.spec.name}_raceline.csv"
        with open(raceline_filepath, "w") as raceline_csv:
            raceline_csv.write("# " + str(uuid.uuid4()) + "\n") # same as TUM opt
            raceline_csv.write('# {}\n'.format(time.strftime('%Y-%m-%d %H:%M:%S'))) # TUM opt uses ggv hash, but no ggv here
            raceline_csv.write("# s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2\n")
            for i in range(len(self.raceline.ss)):
                raceline_csv.write(
                    f"{self.raceline.ss[i]}; {self.raceline.xs[i]}; {self.raceline.ys[i]}; {self.raceline.yaws[i]}; {self.raceline.ks[i]}; {self.raceline.vxs[i]}; {self.raceline.axs[i]}\n"
                )

    def save_centerline(self, outdir: pathlib.Path, half_width: float):
        """Save the track CENTERLINE as ``{name}_centerline.csv``.

        Note: the writer emits a uuid comment line before the header, which
        ``Raceline.from_centerline_file`` (``header_row=0``) cannot read back.

        Args:
            outdir: output directory (pathlib.Path).
            half_width: half width of the track in metres (float).
        """
        raceline_filepath = outdir / f"{self.spec.name}_centerline.csv"
        with open(raceline_filepath, "w") as raceline_csv:
            raceline_csv.write("# " + str(uuid.uuid4()) + "\n") # same as TUM opt
            raceline_csv.write("# x_m, y_m, w_tr_right_m, w_tr_left_m\n")
            for i in range(len(self.centerline.ss)):
                raceline_csv.write(
                    f"{self.centerline.xs[i]}, {self.centerline.ys[i]}, {half_width}, {half_width}\n"
                )

    def frenet_to_cartesian(self, s, ey, ephi, use_raceline=False):
        """
        Convert Frenet coordinates to Cartesian coordinates.

        s: distance along the raceline
        ey: lateral deviation
        ephi: heading deviation

        returns:
            x: x-coordinate
            y: y-coordinate
            psi: yaw angle
        """
        line = self.raceline if use_raceline else self.centerline
        s = s % line.s_frame_max
        x, y = line.spline.calc_position(s)
        psi = line.spline.calc_yaw(s)

        # Adjust x,y by shifting along the normal vector
        x -= ey * np.sin(psi)
        y += ey * np.cos(psi)

        # Adjust psi by adding the heading deviation
        psi += ephi
        # return x, y, np.arctan2(np.sin(psi), np.cos(psi))
        return x, y, (psi + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]

    def cartesian_to_frenet(self, x, y, psi, use_raceline=False, s_guess=None, use_s_guess=True):
        """
        Convert Cartesian coordinates to Frenet coordinates.

        x: x-coordinate
        y: y-coordinate
        psi: yaw angle

        returns:
            s: distance along the centerline
            ey: lateral deviation
            ephi: heading deviation
        """
        line = self.raceline if use_raceline else self.centerline
        if s_guess is None:
            s_guess = self.s_guess

        if use_s_guess and s_guess is not None:
            s_inds = line.spline.find_segment_for_s(s_guess)
            extend_length = int(self.frenet_search_range / 2 / line.spline.s_interval)
            s_inds = np.arange(s_inds - extend_length, s_inds + extend_length) % (len(line.spline.s)-1)
        else:
            s_inds = None
        s, ey = line.spline.calc_arclength(x, y, s_inds)
        # Wrap around
        s = s % line.spline.s[-1]
        self.s_guess = s
        segment = line.spline.find_segment_for_s(s)

        # Use the normal to calculate the signed lateral deviation. Evaluate
        # position and yaw in one shared polynomial evaluation (hot path).
        x_eval, y_eval, yaw = line.spline.calc_position_and_yaw(s, segment)
        normal = np.asarray([-np.sin(yaw), np.cos(yaw)])
        dx = x - x_eval
        dy = y - y_eval
        distance_sign = np.sign(np.dot([dx, dy], normal))
        ey = ey * distance_sign

        psi = psi - yaw
        # return s, ey, np.arctan2(np.sin(psi), np.cos(psi))
        return s, ey, (psi + np.pi) % (2 * np.pi) - np.pi
