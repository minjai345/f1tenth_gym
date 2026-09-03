import time, numpy as np, pytest
from camsim import config, camera, track, render

CSV = "examples/example_waypoints.csv"

@pytest.fixture(scope="module")
def ctx():
    cfg = config.load()
    trk = track.from_csv(CSV, cfg)
    H_g2i, H_i2g = camera.build(cfg)
    return cfg, trk, H_g2i, H_i2g

def pose_on_track(trk, i=50):
    return np.array([trk.center[i, 0], trk.center[i, 1], trk.heading[i]])

def test_shape_and_background(ctx):
    cfg, trk, H, _ = ctx
    img = render.render(pose_on_track(trk), trk.quads, None, H, cfg)
    assert img.shape == (400, 640, 3) and img.dtype == np.uint8
    assert (img[0, 0] == cfg.lane.color_floor).all()      # top-left is sky/floor color

def test_tape_is_drawn_below_horizon_only(ctx):
    cfg, trk, H, _ = ctx
    img = render.render(pose_on_track(trk), trk.quads, None, H, cfg)
    tape = np.all(img == cfg.lane.color_tape, axis=-1)
    assert tape[200:].sum() > 500
    assert tape[:199].sum() == 0

def test_left_tape_on_left(ctx):
    cfg, trk, H, _ = ctx
    img = render.render(pose_on_track(trk), trk.quads, None, H, cfg)
    # Row 380 (forward distance ~0.356 m) is geometrically unreachable by tape at the
    # +/-0.4 m track half-width given hfov=90 deg (visible half-width == forward
    # distance, which never exceeds 0.4 m until row <= 367 for this config). Measured:
    # img[380] has zero tape pixels regardless of pose. Row 350 (forward distance
    # ~0.416 m, comfortably above the 0.4 m half-width) is used instead.
    row = np.all(img[350] == cfg.lane.color_tape, axis=-1)
    cols = np.where(row)[0]
    assert cols.min() < 320 < cols.max()

def test_culling_behind_and_far(ctx):
    cfg, trk, H, _ = ctx
    qv = np.array([[[-1, -0.1], [-0.9, -0.1], [-0.9, 0.1], [-1, 0.1]],     # behind
                   [[20, -0.1], [20.1, -0.1], [20.1, 0.1], [20, 0.1]],     # beyond far
                   [[2, -0.1], [2.1, -0.1], [2.1, 0.1], [2, 0.1]]], float)  # visible
    keep = render.visible_quads(qv, None, cfg)
    assert keep.tolist() == [False, False, True]

def test_lidar_occludes(ctx):
    cfg, trk, H, _ = ctx
    qv = np.array([[[2, -0.1], [2.1, -0.1], [2.1, 0.1], [2, 0.1]]], float)
    scan = np.full(1080, 30.0)
    assert render.visible_quads(qv, scan, cfg)[0]
    scan[:] = 1.0                     # wall at 1 m in every direction
    assert not render.visible_quads(qv, scan, cfg)[0]

def test_ipm_round_trip(ctx):
    """Bottom-of-image tape pixels mapped back to ground land within tape_width of a tape centerline."""
    cfg, trk, H, H_i2g = ctx
    pose = pose_on_track(trk)
    img = render.render(pose, trk.quads, None, H, cfg)
    vs, us = np.where(np.all(img == cfg.lane.color_tape, axis=-1))
    sel = vs > 300
    g = camera.project(H_i2g, np.column_stack([us[sel], vs[sel]]).astype(float))
    qv = render.to_vehicle(pose, trk.quads).reshape(-1, 2)
    d = np.sqrt(((g[:, None, :] - qv[None, :, :]) ** 2).sum(-1)).min(1)
    assert np.percentile(d, 95) < 0.05

def test_render_speed(ctx):
    cfg, trk, H, _ = ctx
    pose = pose_on_track(trk)
    render.render(pose, trk.quads, None, H, cfg)
    t = time.perf_counter()
    for _ in range(100):
        render.render(pose, trk.quads, None, H, cfg)
    assert (time.perf_counter() - t) / 100 < 0.005
