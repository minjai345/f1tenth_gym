import os, numpy as np, pytest
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


def test_draw_paths_on_map_marks_paths(ctx):
    cfg, trk, m = ctx
    path = trk.center[:200] + 0.1
    img, off = viz.draw_paths_on_map(m, trk, cfg, {"a": path, "b": trk.center[300:400]})
    px = np.round(m.world_to_px(path[100]) - off).astype(int)
    near = img[px[1] - 2:px[1] + 3, px[0] - 2:px[0] + 3].reshape(-1, 3).astype(int)
    d = np.abs(near - np.array(viz.PATH_COLORS[0])).sum(1)   # 얇은 AA 선이라 정확 일치 대신 근접도로 본다
    assert d.min() < 90 and d.min() < np.abs(255 - np.array(viz.PATH_COLORS[0])).sum()
    zoom = viz.crop_around(img, off, m, path[-1], half_m=3.0, scale=2)
    assert zoom.shape[0] > 0 and zoom.ndim == 3


def test_to_h264_converts(tmp_path):
    import cv2
    src = str(tmp_path / "v.mp4")
    w = cv2.VideoWriter(src, cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 200))
    for i in range(10):
        f = np.zeros((200, 320, 3), np.uint8); f[:, : 32 * i] = 255; w.write(f)
    w.release()
    dst = viz.to_h264(src, max_width=160)
    assert dst.endswith("_h264.mp4") and os.path.getsize(dst) > 500
    cap = cv2.VideoCapture(dst)
    assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == 160


def test_dashed_paths_leave_gaps(ctx):
    """두 번째 이후 경로는 파선이라 같은 경로를 실선으로 그린 것보다 픽셀이 적어야 한다."""
    cfg, trk, m = ctx
    p = trk.center[:400]
    solid, _ = viz.draw_paths_on_map(m, trk, cfg, {"a": p}, dashed=False)
    dash, _ = viz.draw_paths_on_map(m, trk, cfg, {"first": p + 5.0, "b": p}, dashed=True)
    def near_count(img, col):
        return (np.abs(img.astype(int) - np.array(col)).sum(-1) < 90).sum()
    n_solid = near_count(solid, viz.PATH_COLORS[0])
    n_dash = near_count(dash, viz.PATH_COLORS[1])
    assert 0 < n_dash < n_solid * 0.8


def test_magnify_offsets_scales_deviation_only(ctx):
    cfg, trk, m = ctx
    i = np.arange(0, 300)
    h = trk.heading[i]
    nrm = np.column_stack([-np.sin(h), np.cos(h)])
    path = trk.center[i] + 0.05 * nrm
    out = viz.magnify_offsets(trk, path, 10.0)
    dev = ((out - trk.center[i]) * nrm).sum(1)
    assert np.allclose(dev, 0.5, atol=0.02)                       # 0.05 m -> 0.5 m
    assert np.allclose(viz.magnify_offsets(trk, path, 1.0), path)  # factor 1 = 원본


def test_legend_band_shifts_offset(ctx):
    cfg, trk, m = ctx
    plain, off0 = viz.draw_paths_on_map(m, trk, cfg, {"a": trk.center[:100]}, legend=False)
    withl, off1 = viz.draw_paths_on_map(m, trk, cfg, {"a": trk.center[:100]}, legend=True)
    assert withl.shape[0] > plain.shape[0] and off1[1] < off0[1]
    px = np.round(m.world_to_px(trk.center[50]) - off1).astype(int)   # 좌표 보정이 맞는지
    near = withl[px[1] - 2:px[1] + 3, px[0] - 2:px[0] + 3].reshape(-1, 3).astype(int)
    assert np.abs(near - np.array(viz.PATH_COLORS[0])).sum(1).min() < 90
