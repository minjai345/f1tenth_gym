"""중심선 CSV -> 등간격 중심선, 헤딩, 호길이, 테이프 사각형(quad) 목록."""
from dataclasses import dataclass
import numpy as np
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
    nrm = np.column_stack([-np.sin(heading), np.cos(heading)])
    line = center + offset * nrm
    nxt = np.roll(line, -1, axis=0)
    nrm_n = np.roll(nrm, -1, axis=0)
    h = tape_w / 2.0
    return np.stack([line - h * nrm, nxt - h * nrm_n, nxt + h * nrm_n, line + h * nrm], axis=1)


def from_csv(path: str, cfg: Config, x_col: int = 1, y_col: int = 2, delimiter: str = ";") -> Track:
    raw = np.loadtxt(path, delimiter=delimiter, comments="#")
    step = cfg.lane.segment_len_m
    center = resample(raw[:, [x_col, y_col]], step)
    heading = _heading(center)
    s = np.arange(len(center)) * step
    length = len(center) * step
    half = cfg.lane.track_width_m / 2.0
    quads = np.concatenate([_tape_quads(center, heading, +half, cfg.lane.tape_width_m),
                            _tape_quads(center, heading, -half, cfg.lane.tape_width_m)])
    return Track(center, heading, s, float(length), quads)


def save(track: Track, path) -> None:
    np.savez_compressed(path, center=track.center, heading=track.heading, s=track.s,
                        length=track.length, quads=track.quads)


def load(path) -> Track:
    d = np.load(path)
    return Track(d["center"], d["heading"], d["s"], float(d["length"]), d["quads"])
