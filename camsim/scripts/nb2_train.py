"""NB2: 온더플라이 데이터로 학습 + 오프라인 지표. 인자: steps [batch] [device]"""
import os, sys, numpy as np, torch
sys.path.insert(0, os.getcwd())
from camsim import config, track, train, model

steps = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
batch = int(sys.argv[2]) if len(sys.argv) > 2 else 32
device = sys.argv[3] if len(sys.argv) > 3 else "cpu"
if device == "cuda" and not torch.cuda.is_available():
    print("cuda 사용 불가 — cpu로 대체합니다")
    device = "cpu"

cfg = config.load()
trk = track.from_csv(cfg.closed_loop.centerline_csv, cfg)
net, hist = train.train(trk, cfg, steps=steps, batch_size=batch, device=device,
                        out_path="model.pt", num_workers=2 if device != "cpu" else 0)
r = train.evaluate(model.Predictor(net, cfg, device), trk, cfg, n=300)
print("per-waypoint error (m):", np.round(r["per_waypoint_m"], 3))
print(f"mean {r['mean_m']:.3f} m   max {r['max_m']:.3f} m")
