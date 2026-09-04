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


def body_corners(pose, cfg: Config) -> np.ndarray:
    """차체 사각형 네 모서리(world). gym의 collision_models.get_vertices 와 같은 규약: pose를 중심으로 length x width."""
    x, y, th = pose
    L, W = cfg.closed_loop.car_length_m / 2, cfg.closed_loop.car_width_m / 2
    local = np.array([[L, W], [L, -W], [-L, -W], [-L, W]])
    c, s = np.cos(th), np.sin(th)
    return local @ np.array([[c, s], [-s, c]]) + [x, y]


def crosses_tape(pose, track: Track, cfg: Config) -> bool:
    """실격 규칙: 차체 모서리 중 하나라도 테이프 안쪽 선(중심선에서 track_width/2 - tape_width/2)을 넘으면 True."""
    inner = cfg.lane.track_width_m / 2 - cfg.lane.tape_width_m / 2
    return any(lateral_error(track, c) > inner for c in body_corners(pose, cfg))
