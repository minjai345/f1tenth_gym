import copy, time, numpy as np, pytest
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
    assert img.shape == (cfg.camera.image_height, cfg.camera.image_width, 3) and img.dtype == np.uint8
    assert (img[0, 0] == cfg.lane.color_floor).all()      # top-left is sky/floor color

def test_tape_is_drawn_below_horizon_only(ctx):
    cfg, trk, H, _ = ctx
    img = render.render(pose_on_track(trk), trk.quads, None, H, cfg)
    horizon = cfg.camera.image_height // 2
    tape = np.all(img == cfg.lane.color_tape, axis=-1)
    assert tape[horizon:].sum() > 500
    assert tape[:horizon - 1].sum() == 0

def test_left_tape_on_left(ctx):
    cfg, trk, H, _ = ctx
    img = render.render(pose_on_track(trk), trk.quads, None, H, cfg)
    # A row 3/4 of the way from the horizon to the bottom of the image is geometrically
    # guaranteed to be past the point where the +/-track_width/2 tape is still within the
    # hfov-limited visible half-width (visible half-width grows with forward distance, and
    # forward distance shrinks closer to the horizon) -- measured true for this config.
    horizon = cfg.camera.image_height // 2
    row_idx = horizon + round(0.75 * (cfg.camera.image_height - horizon))
    row = np.all(img[row_idx] == cfg.lane.color_tape, axis=-1)
    cols = np.where(row)[0]
    cx = cfg.camera.image_width // 2
    assert cols.min() < cx < cols.max()

def test_culling_behind_and_far(ctx):
    cfg, trk, H, _ = ctx
    qv = np.array([[[-1, -0.1], [-0.9, -0.1], [-0.9, 0.1], [-1, 0.1]],     # behind
                   [[20, -0.1], [20.1, -0.1], [20.1, 0.1], [20, 0.1]],     # beyond far
                   [[2, -0.1], [2.1, -0.1], [2.1, 0.1], [2, 0.1]]], float)  # visible
    keep = render.visible_quads(qv, None, cfg)
    assert keep.tolist() == [False, False, True]

def test_culling_is_camera_relative_not_vehicle_relative(ctx):
    """With camera.offset_x_m > 0 the camera sits ahead of the rear axle, so near/far
    culling must be measured from the camera, not the vehicle origin. Quads at vehicle-frame
    x in (0, offset_x_m) are behind the camera; if culled with vehicle-relative x they pass,
    get a negative-depth (flipped) projection, and land above the horizon."""
    cfg, trk, _, _ = ctx
    cfg2 = copy.deepcopy(cfg)
    cfg2.camera.offset_x_m = 0.6
    H2, _ = camera.build(cfg2)
    pose = pose_on_track(trk)
    img = render.render(pose, trk.quads, None, H2, cfg2)
    horizon = cfg2.camera.image_height // 2
    tape = np.all(img == cfg2.lane.color_tape, axis=-1)
    assert tape[:horizon - 2].sum() == 0

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
    horizon = cfg.camera.image_height // 2
    row_thresh = horizon + (cfg.camera.image_height - horizon) // 2
    vs, us = np.where(np.all(img == cfg.lane.color_tape, axis=-1))
    sel = vs > row_thresh
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
