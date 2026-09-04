import numpy as np, pytest
from camsim import config, track, gt

@pytest.fixture(scope="module")
def ctx():
    cfg = config.load()
    return cfg, track.from_csv("examples/example_waypoints.csv", cfg)

def test_waypoints_on_straight_are_ahead(ctx):
    cfg, trk = ctx
    i = 10
    pose = np.array([*trk.center[i], trk.heading[i]])
    wp = gt.waypoints_ahead(pose, trk, cfg)
    assert wp.shape == (len(cfg.waypoints.ahead_m), 2)
    assert np.all(np.diff(wp[:, 0]) > 0)
    assert np.allclose(wp[:, 0], cfg.waypoints.ahead_m, atol=0.3)

def test_waypoints_arc_length_monotonic_everywhere(ctx):
    cfg, trk = ctx
    for i in range(0, len(trk.center), 37):
        pose = np.array([*trk.center[i], trk.heading[i]])
        wp = gt.waypoints_ahead(pose, trk, cfg)
        d = np.hypot(*np.diff(np.vstack([[0, 0], wp]), axis=0).T)
        assert np.all(d > 0.2)

def test_left_turn_has_positive_y(ctx):
    cfg, trk = ctx
    dh = np.angle(np.exp(1j * (np.roll(trk.heading, -40) - trk.heading)))
    i = int(np.argmax(dh))              # strongest left turn over next 2 m
    pose = np.array([*trk.center[i], trk.heading[i]])
    wp = gt.waypoints_ahead(pose, trk, cfg)
    assert wp[-1, 1] > 0.05

def test_wraps_at_track_end(ctx):
    cfg, trk = ctx
    i = len(trk.center) - 3
    pose = np.array([*trk.center[i], trk.heading[i]])
    wp = gt.waypoints_ahead(pose, trk, cfg)
    # The farthest waypoint should stay well forward even across the wrap; allow slack for
    # resample-grid snapping rather than requiring it match cfg.waypoints.ahead_m[-1] exactly.
    assert np.all(np.isfinite(wp)) and wp[-1, 0] > cfg.waypoints.ahead_m[-1] * 0.6

def test_sample_pose_within_bounds(ctx):
    cfg, trk = ctx
    rng = np.random.default_rng(1)
    for _ in range(200):
        p = gt.sample_pose(trk, cfg, rng)
        assert gt.lateral_error(trk, p[:2]) <= cfg.sampling.lateral_frac * cfg.lane.track_width_m + 0.03
        i = gt.nearest_index(trk, p[:2])
        dth = np.angle(np.exp(1j * (p[2] - trk.heading[i])))
        assert abs(dth) <= np.deg2rad(cfg.sampling.heading_deg) + 0.05


def test_body_corners_are_rectangle_around_pose(ctx):
    cfg, trk = ctx
    c = gt.body_corners(np.array([1.0, 2.0, np.pi / 2]), cfg)
    assert c.shape == (4, 2)
    assert np.allclose(c.mean(0), [1.0, 2.0])
    L, W = cfg.closed_loop.car_length_m, cfg.closed_loop.car_width_m
    d = np.sort(np.hypot(*(c - [1.0, 2.0]).T))
    assert np.allclose(d, np.hypot(L / 2, W / 2))


def test_crosses_tape_uses_car_body(ctx):
    """실격 = 차체 모서리가 테이프 안쪽 선을 넘음. 중심선 이탈 0.4 m 보다 훨씬 먼저 걸린다."""
    cfg, trk = ctx
    i = 10
    h = trk.heading[i]
    n = np.array([-np.sin(h), np.cos(h)])
    margin = cfg.lane.track_width_m / 2 - cfg.lane.tape_width_m / 2 - cfg.closed_loop.car_width_m / 2
    on = np.array([*trk.center[i], h])
    inside = np.array([*(trk.center[i] + (margin - 0.05) * n), h])
    outside = np.array([*(trk.center[i] + (margin + 0.05) * n), h])
    assert not gt.crosses_tape(on, trk, cfg)
    assert not gt.crosses_tape(inside, trk, cfg)
    assert gt.crosses_tape(outside, trk, cfg)
    # 차를 옆으로 돌리면 모서리가 더 튀어나온다 -> 같은 중심 위치에서도 실격
    turned = np.array([*trk.center[i], h + np.pi / 4])
    assert gt.crosses_tape(turned, trk, cfg) == (
        np.hypot(cfg.closed_loop.car_length_m, cfg.closed_loop.car_width_m) / 2 > cfg.lane.track_width_m / 2 - cfg.lane.tape_width_m / 2)
