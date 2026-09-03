"""NB3: 폐루프 평가 + latency/noise 스윕 + mp4. 인자: [model.pt]"""
import os, sys, json
sys.path.insert(0, os.getcwd())
from camsim import config, camera, track, model, closed_loop as cl

cfg = config.load()
trk = track.from_csv(cfg.closed_loop.centerline_csv, cfg)
H_g2i, _ = camera.build(cfg)
env = cl.make_env(cfg)
os.makedirs("out", exist_ok=True)

print("== oracle sweep (Track B: 인지 오차 주입) ==")
rows = cl.sweep(env, trk, cfg, H_g2i, latency_list=[0, 2, 4, 6], sigma_list=[0.0, 0.05, 0.1, 0.2])
json.dump(rows, open("out/sweep_oracle.json", "w"), indent=1)

if len(sys.argv) > 1:
    model_path = sys.argv[1]
    if not os.path.exists(model_path):
        print(f"model.pt 없음 — NB2를 먼저 실행하세요 ({model_path})")
    else:
        net = model.load(model_path, cfg)
        pred = model.Predictor(net, cfg)
        print("== trained model ==")
        for lat in [0, 2, 4, 6]:
            r = cl.run(env, pred, trk, cfg, H_g2i, latency_steps=lat,
                       video_path=f"out/run_latency{lat}.mp4" if lat == 0 else None)
            print(f"latency={lat}: {r}")
