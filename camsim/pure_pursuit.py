"""차량 좌표계 waypoint (K,2) -> 조향각. 속도는 호출자가 정한다."""
import numpy as np


def pure_pursuit(wp: np.ndarray, lookahead: float, wheelbase: float, steer_max: float) -> float:
    wp = np.asarray(wp, float)
    d = np.hypot(wp[:, 0], wp[:, 1])
    idx = np.where(d >= lookahead)[0]
    i = int(idx[0]) if len(idx) else len(wp) - 1
    x, y = wp[i]
    L2 = max(x * x + y * y, 1e-6)
    curvature = 2.0 * y / L2
    return float(np.clip(np.arctan(wheelbase * curvature), -steer_max, steer_max))
