import numpy as np, torch, pytest
from camsim import config, track, dataset, model, gt

@pytest.fixture(scope="module")
def ctx():
    cfg = config.load()
    return cfg, track.from_csv("examples/example_waypoints.csv", cfg)

def test_crop_is_bottom_half(ctx):
    cfg, _ = ctx
    img = np.zeros((400, 640, 3), np.uint8); img[200:] = 7
    c = dataset.crop(img, cfg)
    assert c.shape == (200, 640, 3) and (c == 7).all()

def test_dataset_yields_tensor_pairs(ctx):
    cfg, trk = ctx
    it = iter(dataset.SynthDataset(trk, cfg, seed=0))
    x, y = next(it)
    assert x.shape == (3, 200, 640) and x.dtype == torch.float32 and 0 <= x.min() <= x.max() <= 1
    assert y.shape == (12,)

def test_dataset_is_deterministic_per_seed(ctx):
    cfg, trk = ctx
    a = next(iter(dataset.SynthDataset(trk, cfg, seed=5)))
    b = next(iter(dataset.SynthDataset(trk, cfg, seed=5)))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])

def test_dataloader_batches(ctx):
    cfg, trk = ctx
    dl = torch.utils.data.DataLoader(dataset.SynthDataset(trk, cfg), batch_size=4, num_workers=0)
    x, y = next(iter(dl))
    assert x.shape == (4, 3, 200, 640) and y.shape == (4, 12)

def test_model_forward_and_size():
    net = model.WaypointNet()
    out = net(torch.zeros(2, 3, 200, 640))
    assert out.shape == (2, 12)
    assert sum(p.numel() for p in net.parameters()) < 1_000_000

def test_predictor_shape(ctx):
    cfg, _ = ctx
    p = model.Predictor(model.WaypointNet(), cfg)
    wp = p.predict(np.zeros((400, 640, 3), np.uint8))
    assert wp.shape == (6, 2)

def test_oracle_matches_gt(ctx):
    cfg, trk = ctx
    pose = np.array([*trk.center[20], trk.heading[20]])
    o = model.OraclePredictor(trk, cfg)
    o.set_pose(pose)
    assert np.allclose(o.predict(None), gt.waypoints_ahead(pose, trk, cfg))

def test_save_load(ctx, tmp_path):
    cfg, _ = ctx
    net = model.WaypointNet()
    model.save(net, tmp_path / "m.pt")
    net2 = model.load(tmp_path / "m.pt", cfg)
    x = torch.zeros(1, 3, 200, 640)
    net.eval(); net2.eval()
    assert torch.allclose(net(x), net2(x))
