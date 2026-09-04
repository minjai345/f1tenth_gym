import copy, time, numpy as np, pytest, cv2
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


# ---- BEV (top-down) rendering and IPM warp ----------------------------------

def _bev_size(cfg):
    b = cfg.bev
    return (int(round((b.x_range_m[1] - b.x_range_m[0]) / b.resolution_m)),
            int(round((b.y_range_m[1] - b.y_range_m[0]) / b.resolution_m)))


def test_render_bev_shape_from_config(ctx):
    cfg, trk, H, _ = ctx
    bev = render.render_bev(pose_on_track(trk), trk.quads, cfg)
    h, w = _bev_size(cfg)
    assert bev.shape == (h, w, 3) and bev.dtype == np.uint8
    assert (bev[0, 0] == cfg.lane.color_floor).all()


def test_render_bev_tape_columns_near_car(ctx):
    """On a straight-ish segment the two tapes sit at y = ±track_width/2 -> known BEV columns."""
    cfg, trk, H, _ = ctx
    pose = np.array([*trk.center[10], trk.heading[10]])
    bev = render.render_bev(pose, trk.quads, cfg)
    b = cfg.bev
    x_probe = b.x_range_m[0] + 0.3
    v = int(round((b.x_range_m[1] - x_probe) / b.resolution_m))
    half = cfg.lane.track_width_m / 2
    u_left = int(round((b.y_range_m[1] - half) / b.resolution_m))
    u_right = int(round((b.y_range_m[1] + half) / b.resolution_m))
    u_mid = (u_left + u_right) // 2
    tape = np.all(bev[v] == cfg.lane.color_tape, axis=-1)
    tol = int(0.1 / b.resolution_m)
    assert tape[u_left - tol:u_left + tol].any()
    assert tape[u_right - tol:u_right + tol].any()
    assert not tape[u_mid - tol:u_mid + tol].any()
    assert u_left < u_mid < u_right                 # +y (left) is on the image's left


def test_bev_pixels_roundtrip(ctx):
    cfg = ctx[0]
    pts = np.array([[cfg.bev.x_range_m[1], cfg.bev.y_range_m[1]], [cfg.bev.x_range_m[0], 0.0]])
    uv = render.bev_pixels(pts, cfg)
    assert np.allclose(uv[0], [0.0, 0.0])
    h, w = _bev_size(cfg)
    assert np.allclose(uv[1], [w / 2, h])


def test_ipm_bev_agrees_with_render_bev_near(ctx):
    """Warping the perspective render with H_i2g must reproduce the true BEV where IPM is well resolved."""
    cfg, trk, H, H_i2g = ctx
    pose = pose_on_track(trk)
    truth = render.render_bev(pose, trk.quads, cfg)
    ipm = render.ipm_bev(render.render(pose, trk.quads, None, H, cfg), H_i2g, cfg)
    assert ipm.shape == truth.shape
    b = cfg.bev
    v_near = int(round((b.x_range_m[1] - 1.5) / b.resolution_m))     # rows closer than 1.5 m
    t = np.all(truth[v_near:] == cfg.lane.color_tape, axis=-1)
    i = np.all(ipm[v_near:] == cfg.lane.color_tape, axis=-1)
    k = np.ones((5, 5), np.uint8)
    t_d = cv2.dilate(t.astype(np.uint8), k).astype(bool)
    assert i.sum() > 500
    assert (i & t_d).sum() / i.sum() > 0.9         # ipm tape lies on (dilated) true tape


def test_bev_visibility_mask_geometry(ctx):
    """마스크: 정면 3 m는 보이고, 0.2 m는 근거리 사각, 전방 1 m·좌측 1.4 m는 화각(90도) 밖."""
    cfg, trk, H, _ = ctx
    m = render.bev_visibility_mask(H, cfg)
    assert m.shape == render.bev_size(cfg) and m.dtype == bool
    def at(x, y):
        u, v = render.bev_pixels(np.array([[x, y]]), cfg)[0]
        return m[int(v), int(u)]
    assert at(3.0, 0.0) and at(1.0, 0.3)
    assert not at(0.25, 0.0)
    assert not at(1.0, 1.4)          # 45도 밖


def test_render_bev_applies_mask(ctx):
    cfg, trk, H, _ = ctx
    pose = pose_on_track(trk)
    m = render.bev_visibility_mask(H, cfg)
    bev = render.render_bev(pose, trk.quads, cfg, m)
    assert np.all(bev[~m] == cfg.lane.color_floor)
    assert np.all(bev == cfg.lane.color_tape, axis=-1).sum() > 1000


def test_draw_points_bev(ctx):
    cfg = ctx[0]
    img = np.zeros((*render.bev_size(cfg), 3), np.uint8)
    render.draw_points_bev(img, np.array([[2.0, 0.0]]), cfg, color=(1, 2, 3))
    u, v = render.bev_pixels(np.array([[2.0, 0.0]]), cfg)[0]
    assert (img[int(v), int(u)] == (1, 2, 3)).all()
