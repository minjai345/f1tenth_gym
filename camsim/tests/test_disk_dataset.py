import os, numpy as np, torch, pytest, cv2
from camsim import config, track, dataset, train, model

@pytest.fixture(scope="module")
def ctx(tmp_path_factory):
    cfg = config.load()
    trk = track.from_csv(cfg.closed_loop.centerline_csv, cfg)
    root = str(tmp_path_factory.mktemp("ds"))
    dataset.generate_dataset(trk, cfg, 30, root, seed=1, log_every=0)
    return cfg, trk, root

def test_files_and_labels_written(ctx):
    cfg, trk, root = ctx
    files, poses, wps = dataset.read_labels(root)
    assert len(files) == 30 and poses.shape == (30, 3) and wps.shape == (30, len(cfg.waypoints.ahead_m), 2)
    assert sorted(os.listdir(os.path.join(root, "images"))) == files
    img = cv2.imread(os.path.join(root, "images", files[0]))
    from camsim.render import bev_size
    assert img.shape == (*bev_size(cfg), 3)      # BEV(모델 입력) 저장

def test_labels_match_geometry(ctx):
    """저장된 waypoint는 저장된 pose에서 다시 계산한 GT와 같아야 한다 (라벨 파일이 자기 설명적)."""
    from camsim import gt
    cfg, trk, root = ctx
    _, poses, wps = dataset.read_labels(root)
    for p, w in zip(poses[:5], wps[:5]):
        assert np.allclose(gt.waypoints_ahead(p, trk, cfg), w, atol=1e-3)

def test_split_is_disjoint_and_deterministic(ctx):
    cfg, trk, root = ctx
    tr = dataset.DiskDataset(root, cfg, "train", val_frac=0.2)
    va = dataset.DiskDataset(root, cfg, "val", val_frac=0.2)
    assert len(tr) == 24 and len(va) == 6
    assert not set(tr.idx) & set(va.idx)
    assert (dataset.split_indices(30, "val", 0.2) == va.idx).all()

def test_item_shapes_and_augment_toggle(ctx):
    cfg, trk, root = ctx
    ds = dataset.DiskDataset(root, cfg, "all", image_augment=False)
    x, y = ds[0]
    from camsim.render import bev_size
    assert x.shape == (3, *bev_size(cfg)) and y.shape == (2 * len(cfg.waypoints.ahead_m),)
    x2, _ = ds[0]
    assert torch.equal(x, x2)                       # 증강 없으면 결정적
    assert torch.allclose(y * cfg.waypoints.norm_m, torch.from_numpy(ds.wps[0].reshape(-1)).float())

def test_train_on_disk_and_evaluate(ctx, tmp_path):
    cfg, trk, root = ctx
    ds = dataset.DiskDataset(root, cfg, "train", val_frac=0.2)
    net, hist = train.train(trk, cfg, steps=12, batch_size=4, dataset=ds, log_every=6, out_path=tmp_path / "m.pt")
    assert len(hist) == 2 and (tmp_path / "m.pt").exists()
    r = train.evaluate_dataset(model.Predictor(net, cfg), dataset.DiskDataset(root, cfg, "val", val_frac=0.2))
    assert r["n"] == 6 and np.isfinite(r["mean_m"])

def test_waypoint_count_mismatch_raises(ctx):
    import copy
    cfg, trk, root = ctx
    cfg2 = copy.deepcopy(cfg); cfg2.waypoints.ahead_m = [1.0, 2.0]
    with pytest.raises(ValueError):
        dataset.DiskDataset(root, cfg2)
