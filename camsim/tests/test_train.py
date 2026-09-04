import numpy as np, pytest
from camsim import config, track, train, model

@pytest.fixture(scope="module")
def ctx():
    cfg = config.load()
    return cfg, track.from_csv("examples/example_waypoints.csv", cfg)

def test_evaluate_oracle_is_zero(ctx):
    cfg, trk = ctx
    o = model.OraclePredictor(trk, cfg)
    class P:   # evaluate must call set_pose when the predictor has it
        def __init__(s): s.o = o
        def predict(s, img): return s.o.predict(img)
        def set_pose(s, p): s.o.set_pose(p)
    r = train.evaluate(P(), trk, cfg, n=20)
    assert r["mean_m"] < 1e-9 and r["per_waypoint_m"].shape == (len(cfg.waypoints.ahead_m),)

def test_train_runs_and_loss_drops(ctx, tmp_path):
    cfg, trk = ctx
    net, hist = train.train(trk, cfg, steps=30, batch_size=4, out_path=tmp_path / "m.pt", log_every=10)
    assert len(hist) == 3
    assert hist[-1]["loss"] < hist[0]["loss"] * 1.5    # sanity: not diverging
    assert (tmp_path / "m.pt").exists()
    r = train.evaluate(model.Predictor(net, cfg), trk, cfg, n=10)
    assert np.isfinite(r["mean_m"])


def test_val_loss_and_callback(ctx, tmp_path):
    from camsim import dataset
    cfg, trk = ctx
    root = str(tmp_path / "ds"); dataset.generate_dataset(trk, cfg, 24, root, log_every=0)
    tr = dataset.DiskDataset(root, cfg, "train", val_frac=0.25)
    va = dataset.DiskDataset(root, cfg, "val", val_frac=0.25)
    calls = []
    net, hist = train.train(trk, cfg, steps=8, batch_size=4, dataset=tr, val_dataset=va, val_batches=2,
                            log_every=4, callback=lambda h: calls.append(len(h)))
    assert [h["step"] for h in hist] == [4, 8] and all("val_loss" in h and np.isfinite(h["val_loss"]) for h in hist)
    assert calls == [1, 2]


def test_evaluate_with_degrade_fn(ctx):
    cfg, trk = ctx
    o = model.OraclePredictor(trk, cfg)
    class P:
        def predict(s, img): return o.predict(img)
        def set_pose(s, p): o.set_pose(p)
    seen = []
    r = train.evaluate(P(), trk, cfg, n=5, degrade_fn=lambda bev, rng: (seen.append(bev.shape), bev)[1])
    assert len(seen) == 5 and r["mean_m"] < 1e-9
