import numpy as np, torch, pytest
from camsim import config, track, dataset, model, gt

@pytest.fixture(scope="module")
def ctx():
    cfg = config.load()
    return cfg, track.from_csv("examples/example_waypoints.csv", cfg)

def bev_hw(cfg):
    from camsim.render import bev_size
    return bev_size(cfg)

def test_make_sample_is_bev_with_camera_mask(ctx):
    """모델 입력은 BEV. 카메라가 못 보는 근거리(0.32 m 안쪽)는 바닥색이어야 한다(실차 IPM 출력과 동일)."""
    from camsim import camera, render
    cfg, trk = ctx
    bev, wp, pose, cam = dataset.make_sample(trk, cfg, np.random.default_rng(0), do_augment=False, with_camera=True)
    h, w = bev_hw(cfg)
    assert bev.shape == (h, w, 3) and cam.shape == (cfg.camera.image_height, cfg.camera.image_width, 3)
    near_rows = int((cfg.bev.x_range_m[1] - 0.30) / cfg.bev.resolution_m)      # x < 0.30 m -> 아래쪽 행들
    assert np.all(bev[near_rows:] == cfg.lane.color_floor)
    assert np.all(bev == cfg.lane.color_tape, axis=-1).sum() > 1000

def test_dataset_yields_tensor_pairs(ctx):
    cfg, trk = ctx
    n_out = 2 * len(cfg.waypoints.ahead_m)
    it = iter(dataset.SynthDataset(trk, cfg, seed=0))
    x, y = next(it)
    assert x.shape == (3, *bev_hw(cfg))
    assert x.dtype == torch.float32 and 0 <= x.min() <= x.max() <= 1
    assert y.shape == (n_out,)

def test_dataset_is_deterministic_per_seed(ctx):
    cfg, trk = ctx
    a = next(iter(dataset.SynthDataset(trk, cfg, seed=5)))
    b = next(iter(dataset.SynthDataset(trk, cfg, seed=5)))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])

def test_dataloader_batches(ctx):
    cfg, trk = ctx
    n_out = 2 * len(cfg.waypoints.ahead_m)
    dl = torch.utils.data.DataLoader(dataset.SynthDataset(trk, cfg), batch_size=4, num_workers=0)
    x, y = next(iter(dl))
    assert x.shape == (4, 3, *bev_hw(cfg))
    assert y.shape == (4, n_out)

def test_model_forward_and_size(ctx):
    cfg, _ = ctx
    net = model.WaypointNet()
    out = net(torch.zeros(2, 3, *bev_hw(cfg)))
    assert out.shape == (2, 12)   # WaypointNet()'s own default n_out, not config-driven
    assert sum(p.numel() for p in net.parameters()) < 1_000_000

def test_predictor_shape(ctx):
    cfg, _ = ctx
    net = model.WaypointNet(n_out=2 * len(cfg.waypoints.ahead_m))
    p = model.Predictor(net, cfg)
    wp = p.predict(np.zeros((*bev_hw(cfg), 3), np.uint8))
    assert wp.shape == (len(cfg.waypoints.ahead_m), 2)

def test_oracle_matches_gt(ctx):
    cfg, trk = ctx
    pose = np.array([*trk.center[20], trk.heading[20]])
    o = model.OraclePredictor(trk, cfg)
    o.set_pose(pose)
    assert np.allclose(o.predict(None), gt.waypoints_ahead(pose, trk, cfg))

def test_save_load(ctx, tmp_path):
    cfg, _ = ctx
    net = model.WaypointNet(n_out=2 * len(cfg.waypoints.ahead_m))
    model.save(net, tmp_path / "m.pt")
    net2 = model.load(tmp_path / "m.pt", cfg)
    x = torch.zeros(1, 3, *bev_hw(cfg))
    net.eval(); net2.eval()
    assert torch.allclose(net(x), net2(x))

def test_load_rejects_n_out_mismatch(ctx, tmp_path):
    cfg, _ = ctx
    bad_n_out = 2 * len(cfg.waypoints.ahead_m) + 2   # deliberately mismatched
    net = model.WaypointNet(n_out=bad_n_out)
    model.save(net, tmp_path / "bad.pt")
    with pytest.raises(ValueError, match="n_out"):
        model.load(tmp_path / "bad.pt", cfg)


def test_predict_camera_matches_predict_on_ipm(ctx):
    """실차 경로(카메라 -> IPM -> predict)는 같은 BEV를 직접 넣은 것과 같아야 한다."""
    from camsim import camera, render
    cfg, trk = ctx
    H_g2i, H_i2g = camera.build(cfg)
    p = model.Predictor(model.WaypointNet(), cfg)
    cam = render.render(np.array([*trk.center[50], trk.heading[50]]), trk.quads, None, H_g2i, cfg)
    assert np.allclose(p.predict_camera(cam, H_i2g), p.predict(render.ipm_bev(cam, H_i2g, cfg)))
