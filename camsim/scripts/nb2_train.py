"""NB2: 학습 + 오프라인 지표. 인자: steps [batch] [device] [data_dir=out/dataset]

data_dir 에 labels.csv 가 있으면 저장된 데이터셋(DiskDataset, train/val 9:1)으로 학습하고 val 오차를 낸다.
없으면 온더플라이(SynthDataset)로 학습한다. 어느 쪽인지 첫 줄에 출력된다.
"""
import os, sys, numpy as np, torch
sys.path.insert(0, os.getcwd())
from camsim import config, track, train, model, dataset

steps = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
batch = int(sys.argv[2]) if len(sys.argv) > 2 else 32
device = sys.argv[3] if len(sys.argv) > 3 else "cpu"
data_dir = sys.argv[4] if len(sys.argv) > 4 else "out/dataset"
if device == "cuda" and not torch.cuda.is_available():
    print("cuda 사용 불가 — cpu로 대체합니다")
    device = "cpu"

cfg = config.load()
trk = track.from_csv(cfg.closed_loop.centerline_csv, cfg)
workers = 2 if device != "cpu" else 0
use_disk = os.path.isfile(os.path.join(data_dir, dataset.LABELS_CSV))
if use_disk:
    ds_train = dataset.DiskDataset(data_dir, cfg, "train", val_frac=0.1)
    ds_val = dataset.DiskDataset(data_dir, cfg, "val", val_frac=0.1)
    print(f"데이터: {data_dir} (train {len(ds_train)}장, val {len(ds_val)}장)")
else:
    ds_train = None
    print(f"데이터: 온더플라이 생성 ({data_dir} 없음. 저장 데이터로 학습하려면 make_dataset.py 를 먼저 실행)")

net, hist = train.train(trk, cfg, steps=steps, batch_size=batch, device=device,
                        out_path="model.pt", num_workers=workers, dataset=ds_train)
pred = model.Predictor(net, cfg, device)
if use_disk:
    r = train.evaluate_dataset(pred, ds_val, n=300)
    print(f"[val 저장 이미지 {r['n']}장] per-waypoint error (m):", np.round(r["per_waypoint_m"], 3))
    print(f"[val] mean {r['mean_m']:.3f} m   max {r['max_m']:.3f} m")
r = train.evaluate(pred, trk, cfg, n=300)
print("[새 pose 300개] per-waypoint error (m):", np.round(r["per_waypoint_m"], 3))
print(f"[새 pose] mean {r['mean_m']:.3f} m   max {r['max_m']:.3f} m")
