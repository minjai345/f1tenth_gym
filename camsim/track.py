"""중심선 CSV -> 등간격 중심선, 헤딩, 호길이, 테이프 사각형(quad) 목록."""
from dataclasses import dataclass
import numpy as np
import os

import yaml

from .config import Config


@dataclass
class Track:
    center: np.ndarray    # (N,2) world m
    heading: np.ndarray   # (N,) rad
    s: np.ndarray         # (N,) cumulative arc length, s[0]=0
    length: float         # closed-loop length
    quads: np.ndarray     # (M,4,2) tape rectangles, world m


def resample(xy: np.ndarray, step: float) -> np.ndarray:
    """Closed-loop resample at uniform arc-length step. Output has no duplicated closing point."""
    xy = np.asarray(xy, dtype=np.float64)
    if np.hypot(*(xy[0] - xy[-1])) > 1e-6:
        xy = np.vstack([xy, xy[:1]])
    seg = np.hypot(*np.diff(xy, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n = int(np.floor(s[-1] / step))
    s_new = np.arange(n) * step
    return np.column_stack([np.interp(s_new, s, xy[:, 0]), np.interp(s_new, s, xy[:, 1])])


def _heading(center: np.ndarray) -> np.ndarray:
    d = np.roll(center, -1, axis=0) - np.roll(center, 1, axis=0)
    return np.arctan2(d[:, 1], d[:, 0])


def _tape_quads(center, heading, offset, tape_w):
    """offset 은 스칼라 또는 지점별 (N,) 배열 (벽을 따라갈 때 폭이 변한다)."""
    nrm = np.column_stack([-np.sin(heading), np.cos(heading)])
    line = center + np.asarray(offset).reshape(-1, 1) * nrm if np.ndim(offset) else center + offset * nrm
    nxt = np.roll(line, -1, axis=0)
    nrm_n = np.roll(nrm, -1, axis=0)
    h = tape_w / 2.0
    return np.stack([line - h * nrm, nxt - h * nrm_n, nxt + h * nrm_n, line + h * nrm], axis=1)


def wall_offsets(center: np.ndarray, heading: np.ndarray, map_yaml: str, margin_m: float,
                 max_search_m: float = 6.0, step_m: float = 0.02):
    """중심선 각 지점에서 좌/우 법선 방향으로 벽까지 거리를 재고, margin_m 만큼 안쪽 오프셋을 돌려준다.

    맵 PNG 의 자유 공간(밝은 픽셀)을 따라 법선 방향으로 조금씩 나아가다 벽을 만나면 멈춘다.
    반환: (left_offset (N,), right_offset (N,)) — 둘 다 양수이며 각각 +normal / -normal 방향 거리(m).
    """
    from PIL import Image                            # gym 이 이미 의존하는 Pillow (cv2 없이도 track 을 만들 수 있게)
    meta = yaml.safe_load(open(map_yaml))
    img = np.asarray(Image.open(os.path.join(os.path.dirname(map_yaml), meta["image"])).convert("L"))
    res, (ox, oy) = float(meta["resolution"]), (float(meta["origin"][0]), float(meta["origin"][1]))
    h, w = img.shape
    free = img > 128                                   # 흰색 = 자유 공간

    def is_free(xy):
        col = ((xy[:, 0] - ox) / res).astype(int)
        row = (h - 1) - ((xy[:, 1] - oy) / res).astype(int)
        ok = (col >= 0) & (col < w) & (row >= 0) & (row < h)
        out = np.zeros(len(xy), bool)
        out[ok] = free[row[ok], col[ok]]
        return out

    nrm = np.column_stack([-np.sin(heading), np.cos(heading)])
    dists = []
    for sign in (+1.0, -1.0):
        d = np.zeros(len(center))
        alive = np.ones(len(center), bool)
        for t in np.arange(step_m, max_search_m + step_m, step_m):
            probe = center + sign * t * nrm
            hit = alive & ~is_free(probe)
            d[hit] = t
            alive &= ~hit
            if not alive.any():
                break
        d[alive] = max_search_m
        dists.append(np.maximum(d - margin_m, 0.0))
    return dists[0], dists[1]


def _smooth_closed(x: np.ndarray, k: int) -> np.ndarray:
    """닫힌 배열의 이동평균 (테이프가 픽셀 단위로 들쭉날쭉하지 않도록)."""
    if k < 2:
        return x
    ker = np.ones(k) / k
    return np.convolve(np.concatenate([x[-k:], x, x[:k]]), ker, mode="same")[k:-k]


def from_csv(path: str, cfg: Config, x_col: int = 1, y_col: int = 2, delimiter: str = ";") -> Track:
    raw = np.loadtxt(path, delimiter=delimiter, comments="#")
    step = cfg.lane.segment_len_m
    center = resample(raw[:, [x_col, y_col]], step)
    heading = _heading(center)
    s = np.arange(len(center)) * step
    length = len(center) * step
    if cfg.lane.follow_walls:                          # 맵의 벽 안쪽을 따라간다 (폭이 구간마다 다름)
        left, right = wall_offsets(center, heading, cfg.closed_loop.map_yaml, cfg.lane.wall_margin_m)
        k = max(1, int(round(0.5 / step)))             # 0.5 m 창으로 부드럽게
        left, right = _smooth_closed(left, k), _smooth_closed(right, k)
    else:                                              # 중심선에서 일정 폭 (실차 테이프 트랙 방식)
        half = cfg.lane.track_width_m / 2.0
        left = right = np.full(len(center), half)
    quads = np.concatenate([_tape_quads(center, heading, +left, cfg.lane.tape_width_m),
                            _tape_quads(center, heading, -right, cfg.lane.tape_width_m)])
    return Track(center, heading, s, float(length), quads)


def save(track: Track, path) -> None:
    np.savez_compressed(path, center=track.center, heading=track.heading, s=track.s,
                        length=track.length, quads=track.quads)


def load(path) -> Track:
    d = np.load(path)
    return Track(d["center"], d["heading"], d["s"], float(d["length"]), d["quads"])
