import numpy as np, pytest
from camsim import config
from camsim.pure_pursuit import pure_pursuit

_cfg = config.load()
WB, SMAX = _cfg.closed_loop.wheelbase_m, _cfg.closed_loop.steer_max_rad

def test_straight_gives_zero():
    wp = np.column_stack([np.arange(1, 7) * 0.5, np.zeros(6)])
    assert pure_pursuit(wp, 1.2, WB, SMAX) == pytest.approx(0.0)

def test_left_curve_positive_and_clipped():
    wp = np.column_stack([np.arange(1, 7) * 0.5, np.arange(1, 7) * 0.4])
    s = pure_pursuit(wp, 1.2, WB, SMAX)
    assert 0 < s <= SMAX
    wp[:, 1] *= 3
    assert pure_pursuit(wp, 1.2, WB, SMAX) == pytest.approx(SMAX)

def test_right_curve_negative():
    wp = np.column_stack([np.arange(1, 7) * 0.5, -np.arange(1, 7) * 0.2])
    assert pure_pursuit(wp, 1.2, WB, SMAX) < 0

def test_short_waypoints_use_last():
    wp = np.array([[0.3, 0.1], [0.6, 0.2]])
    assert pure_pursuit(wp, 5.0, WB, SMAX) > 0
