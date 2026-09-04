import numpy as np, pytest
from camsim import config, track, gt, viz

MAP_YAML = "examples/example_map.yaml"

@pytest.fixture(scope="module")
def ctx():
    cfg = config.load()
    trk = track.from_csv(cfg.closed_loop.centerline_csv, cfg)
    return cfg, trk, viz.MapImage(MAP_YAML)

def test_to_world_inverts_to_vehicle(ctx):
    from camsim.render import to_vehicle
    pose = np.array([3.0, -2.0, 0.7])
    pts = np.random.default_rng(0).uniform(-5, 5, (20, 2))
    assert np.allclose(viz.to_world(pose, to_vehicle(pose, pts)), pts)

def test_map_pixels_match_gym_convention(ctx):
    cfg, trk, m = ctx
    # 맵 origin은 좌하단, y가 커지면 row가 줄어든다
    a = m.world_to_px([m.ox, m.oy]); b = m.world_to_px([m.ox, m.oy + m.res])
    assert np.allclose(a, [0, m.h - 1]) and np.allclose(b, [0, m.h - 2])

def test_track_lands_on_free_space(ctx):
    cfg, trk, m = ctx
    px = np.round(m.world_to_px(trk.center)).astype(int)
    vals = m.gray[px[:, 1], px[:, 0]]
    assert (vals > 128).mean() > 0.99          # centerline is on free (white) cells

def test_draw_track_on_map_shape_and_tape(ctx):
    cfg, trk, m = ctx
    img, off = viz.draw_track_on_map(m, trk, cfg)
    assert img.ndim == 3 and img.shape[0] <= m.h and img.shape[1] <= m.w
    assert np.all(img == cfg.lane.color_tape, axis=-1).sum() > 1000

def test_local_view_waypoints_at_expected_pixels(ctx):
    cfg, trk, m = ctx
    i = 10
    pose = np.array([*trk.center[i], trk.heading[i]])
    wp = gt.waypoints_ahead(pose, trk, cfg)
    img = viz.local_view(pose, trk, wp, cfg, m, ahead_m=5.0, behind_m=1.5, half_width_m=3.0, res_m=0.01)
    assert img.shape == (650, 600, 3)
    for p in wp:                                # waypoint marker (green) drawn where the geometry says
        u, v = int(round((3.0 - p[1]) / 0.01)), int(round((5.0 - p[0]) / 0.01))
        assert (img[v, u] == viz.COL_WP).all()

def test_side_by_side_heights_match():
    a = np.zeros((400, 640, 3), np.uint8); b = np.zeros((650, 600, 3), np.uint8)
    out = viz.side_by_side(a, b)
    assert out.shape[0] == 650 and out.shape[1] > 640 + 600
