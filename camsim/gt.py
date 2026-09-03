"""정답 waypoint(전방 호길이 기준)와 학습용 pose 샘플링."""
import numpy as np
from .config import Config
from .track import Track
from .render import to_vehicle


def nearest_index(track: Track, xy) -> int:
    d = track.center - np.asarray(xy)[:2]
    return int(np.argmin(d[:, 0] ** 2 + d[:, 1] ** 2))


def lateral_error(track: Track, xy) -> float:
    i = nearest_index(track, xy)
    return float(np.hypot(*(track.center[i] - np.asarray(xy)[:2])))


def waypoints_ahead(pose, track: Track, cfg: Config) -> np.ndarray:
    i = nearest_index(track, pose[:2])
    s_t = (track.s[i] + np.asarray(cfg.waypoints.ahead_m)) % track.length
    j = np.searchsorted(track.s, s_t) % len(track.s)
    return to_vehicle(pose, track.center[j])


def sample_pose(track: Track, cfg: Config, rng: np.random.Generator) -> np.ndarray:
    i = int(rng.integers(len(track.center)))
    lat = rng.uniform(-1, 1) * cfg.sampling.lateral_frac * cfg.lane.track_width_m
    dth = rng.uniform(-1, 1) * np.deg2rad(cfg.sampling.heading_deg)
    h = track.heading[i]
    n = np.array([-np.sin(h), np.cos(h)])
    xy = track.center[i] + lat * n
    return np.array([xy[0], xy[1], h + dth])
