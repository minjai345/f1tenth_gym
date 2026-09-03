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
    assert wp.shape == (6, 2)
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
    assert np.all(np.isfinite(wp)) and wp[-1, 0] > 2.0

def test_sample_pose_within_bounds(ctx):
    cfg, trk = ctx
    rng = np.random.default_rng(1)
    for _ in range(200):
        p = gt.sample_pose(trk, cfg, rng)
        assert gt.lateral_error(trk, p[:2]) <= 0.35 * 0.8 + 0.03
        i = gt.nearest_index(trk, p[:2])
        dth = np.angle(np.exp(1j * (p[2] - trk.heading[i])))
        assert abs(dth) <= np.deg2rad(15) + 0.05
