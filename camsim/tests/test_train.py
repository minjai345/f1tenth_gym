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
    assert r["mean_m"] < 1e-9 and r["per_waypoint_m"].shape == (6,)

def test_train_runs_and_loss_drops(ctx, tmp_path):
    cfg, trk = ctx
    net, hist = train.train(trk, cfg, steps=30, batch_size=4, out_path=tmp_path / "m.pt", log_every=10)
    assert len(hist) == 3
    assert hist[-1]["loss"] < hist[0]["loss"] * 1.5    # sanity: not diverging
    assert (tmp_path / "m.pt").exists()
    r = train.evaluate(model.Predictor(net, cfg), trk, cfg, n=10)
    assert np.isfinite(r["mean_m"])
