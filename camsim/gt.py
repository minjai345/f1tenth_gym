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
    corridor = float(track.left_m[i] + track.right_m[i])      # 그 지점의 실제 트랙 폭
    lat = rng.uniform(-1, 1) * cfg.sampling.lateral_frac * corridor
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


def signed_lateral(track: Track, xy):
    """기준 경로에서의 부호 있는 횡 오프셋 (+ = 왼쪽). (offset, 최근접 인덱스) 반환."""
    i = nearest_index(track, xy)
    n = np.array([-np.sin(track.heading[i]), np.cos(track.heading[i])])
    return float((np.asarray(xy)[:2] - track.center[i]) @ n), i


def crosses_tape(pose, track: Track, cfg: Config) -> bool:
    """실격 규칙: 차체 모서리 중 하나라도 테이프 안쪽 선을 넘으면 True.

    테이프 위치는 지점마다 다를 수 있으므로(벽 추종 트랙) track.left_m / right_m 을 쓴다.
    """
    half_tape = cfg.lane.tape_width_m / 2
    for c in body_corners(pose, cfg):
        lat, i = signed_lateral(track, c)
        if lat > track.left_m[i] - half_tape or -lat > track.right_m[i] - half_tape:
            return True
    return False
