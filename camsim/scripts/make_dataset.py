"""데이터셋 생성: N장을 out/dataset/images/*.png + labels.csv 로 저장. 인자: [N=20000] [out_dir=out/dataset] [seed=0]

pitch 지터·테이프 결손은 여기서 샘플마다 다르게 들어가고, 블러·글레어는 학습 로딩 때 적용된다.
이미지는 640x400 전체 렌더(지평선 포함)이고 모델 입력 크롭은 로딩 때 한다. 학생이 파일을 열어 보면 카메라가 본 그대로다.
"""
import os, sys, time
sys.path.insert(0, os.getcwd())
from camsim import config, track, dataset

n = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
out_dir = sys.argv[2] if len(sys.argv) > 2 else "out/dataset"
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0

cfg = config.load()
trk = track.from_csv(cfg.closed_loop.centerline_csv, cfg)
t0 = time.time()
path = dataset.generate_dataset(trk, cfg, n, out_dir, seed=seed)
size_mb = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(out_dir) for f in fs) / 1e6
print(f"{n}장 저장: {out_dir}  ({size_mb:.0f} MB, {time.time() - t0:.0f}s)  라벨: {path}")
