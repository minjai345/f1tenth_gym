"""notebooks/camsim_lab.ipynb 를 생성한다. 노트북 내용은 이 파일이 원본이다 (수정 후 재실행).

    .venv/bin/python camsim/scripts/build_notebook.py
    # 로컬 검증 (축소 파라미터):
    CAMSIM_SMOKE=1 .venv/bin/jupyter nbconvert --to notebook --execute notebooks/camsim_lab.ipynb \
        --output /tmp/lab_out.ipynb --ExecutePreprocessor.timeout=1800
"""
import json
import os

cells = []


def _id():
    return f"cell-{len(cells):02d}"


def md(*lines):
    cells.append({"cell_type": "markdown", "id": _id(), "metadata": {}, "source": [l + "\n" for l in lines]})


def code(*lines):
    cells.append({"cell_type": "code", "id": _id(), "metadata": {}, "execution_count": None, "outputs": [],
                  "source": [l + "\n" for l in lines]})


# --------------------------------------------------------------------------- 0. 설정
md("# camsim 실습 노트북 — 카메라 기반 waypoint 모델: 시뮬 검증",
   "",
   "`f1tenth_gym`은 카메라를 그리지 못한다. 바닥 테이프 트랙은 전부 평면이므로 캘리브레이션 행렬 하나로 가짜 전방 영상을 그릴 수 있다.",
   "이 노트북은 그 지오메트리로 **BEV(top-down) 학습 데이터**를 만들고, waypoint CNN을 학습하고, gym 안에서 폐루프로 검증한다.",
   "모델 입력은 BEV다. 시뮬은 BEV를 직접 그리고, 실차는 카메라 영상을 IPM으로 펴서 같은 규격의 BEV를 만든다.",
   "",
   "**진행 방법**: 각 장의 첫 셀에 있는 파라미터를 바꾸고 그 장을 다시 실행하면서 무엇이 달라지는지 본다.",
   "노트북 하나가 커널 하나이므로 앞 장에서 만든 것(데이터, 모델)은 뒤 장에서 그대로 쓴다.",
   "",
   "| 장 | 내용 | 바꿔 볼 값 |",
   "|---|---|---|",
   "| 1 | 카메라와 트랙 | 카메라 높이·pitch·화각, 트랙 폭, 테이프 색 |",
   "| 2 | 정답(GT)과 데이터셋 (BEV) | pose 샘플링 범위, waypoint 거리, 증강 세기, 장 수 |",
   "| 3 | 학습 | 스텝, 배치, 학습률 |",
   "| 4 | 폐루프 | 속도, lookahead, 지연, 인지 노이즈 |",
   "| 5 | 결과 보관 | |")

md("## 0. 설치와 설정",
   "코랩에서는 첫 셀이 레포를 받고 의존성을 설치한다 (1~2분). 이미 레포 안에서 실행 중이면(로컬) 건너뛴다.")
code('REPO_URL = "https://github.com/minjai345/f1tenth_gym.git"   # 조교 fork. 본인 fork를 쓰면 여기만 바꾸세요',
     'BRANCH = "main"',
     'import os',
     'if not os.path.isfile("camsim/config.yaml"):                 # 코랩: 레포 밖에서 시작',
     '    if os.path.isdir("f1tenth_gym"):',
     '        !git -C f1tenth_gym pull -q',
     '    else:',
     '        !git clone --branch $BRANCH $REPO_URL',
     '    assert os.path.isfile("f1tenth_gym/camsim/config.yaml"), "clone 실패: REPO_URL/BRANCH 를 확인하세요"',
     '    %cd f1tenth_gym',
     '    !pip install -q -r camsim/requirements.txt',
     '    !bash camsim/scripts/install_gym019.sh',
     '    !pip install -q --no-deps -e .',
     'else:                                                       # 이미 레포 안 (코랩 재실행 또는 로컬)',
     '    try:',
     '        import google.colab                                 # 코랩이면 최신 코드·의존성으로 갱신',
     '        !git pull -q',
     '        !pip install -q -r camsim/requirements.txt',
     '    except ImportError:',
     '        pass                                                # 로컬은 건드리지 않음',
     'print("repo:", os.getcwd())')

code('import sys, copy, json, time, numpy as np, cv2, torch, pandas as pd',
     'import matplotlib.pyplot as plt',
     'plt.rcParams["axes.unicode_minus"] = False   # 그래프 글자는 영어(코랩 기본 폰트에 한글 없음)',
     'from IPython.display import Image, Video, display, clear_output',
     'sys.path.insert(0, os.getcwd())',
     'from camsim import config, camera, track, render, gt, augment, dataset, model, train, closed_loop, viz',
     '',
     'def show(img_bgr, width=640, title=None):',
     '    """BGR numpy 이미지를 셀 출력에 띄운다 (임시 파일 없음)."""',
     '    if title: print(title)',
     '    display(Image(data=cv2.imencode(".png", img_bgr)[1].tobytes(), width=width))',
     '',
     'os.makedirs("out", exist_ok=True)',
     'SMOKE = os.environ.get("CAMSIM_SMOKE") == "1"          # (테스트용) 자동 검증 시 데이터·학습을 축소',
     'DEVICE = "cuda" if torch.cuda.is_available() else "cpu"',
     'print("device:", DEVICE, "| torch", torch.__version__, "| numpy", np.__version__)')

md("### 파라미터",
   "`camsim/config.yaml` 이 기본값이다. 아래에서 덮어쓴 값이 이 노트북 전체에 적용된다. 값을 바꾼 뒤에는 이 셀부터 다시 실행한다.")
code('cfg = config.load()',
     '',
     '# ---- 1장: 카메라와 트랙 -------------------------------------------------',
     'cfg.camera.height_m   = 0.20      # 바닥에서 렌즈 중심까지 (m). 마운트 확정 전 가정값',
     'cfg.camera.pitch_deg  = 0.0       # 아래로 숙인 각도 (+ = 아래). 정면 = 0',
     'cfg.camera.hfov_deg   = 90.0      # 렌즈 수평 화각',
     'cfg.lane.track_width_m = 0.8      # 좌우 테이프 중심 사이 거리 (m)',
     'cfg.lane.color_floor  = [128, 128, 128]   # BGR. 회색 바닥',
     'cfg.lane.color_tape   = [0, 220, 255]     # BGR. 노란 테이프',
     '',
     '# ---- 2장: GT와 데이터 ---------------------------------------------------',
     'cfg.waypoints.ahead_m = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]   # 전방 호길이 (m). 개수가 모델 출력 크기',
     'cfg.sampling.lateral_frac = 0.35  # 중심선에서 ±track_width*frac 까지 pose를 뿌린다',
     'cfg.sampling.heading_deg  = 15.0  # 헤딩 ±',
     'cfg.augment.pitch_jitter_deg = 2.0',
     'cfg.augment.tape_dropout_prob = 0.3',
     'cfg.augment.glare_prob = 0.3',
     'N_DATASET = 300 if SMOKE else 20000     # 저장할 장 수',
     '',
     '# ---- 3장: 학습 --------------------------------------------------------',
     'STEPS = 30 if SMOKE else 3000',
     'BATCH = 8 if SMOKE else 32',
     'LR = 1e-3',
     '',
     '# ---- 4장: 폐루프 ------------------------------------------------------',
     'cfg.closed_loop.speed_mps = 2.0',
     'cfg.closed_loop.lookahead_m = 1.2',
     'cfg.closed_loop.max_steps = 200 if SMOKE else 4000',
     'LATENCIES = [0, 3] if SMOKE else [0, 2, 4, 6]        # 제어 틱 단위 지연 (1틱 = 1/control_hz 초)',
     'SIGMAS = [0.0, 0.1] if SMOKE else [0.0, 0.05, 0.1, 0.2]  # 인지 노이즈 σ (m)',
     '',
     'trk = track.from_csv(cfg.closed_loop.centerline_csv, cfg)',
     'H_g2i, H_i2g = camera.build(cfg)',
     'print(f"트랙 길이 {trk.length:.1f} m, 테이프 사각형 {len(trk.quads)}개, 초점거리 {camera.focal_px(cfg):.0f} px")')

# --------------------------------------------------------------------------- 1. 카메라와 트랙
md("## 1. 카메라와 트랙",
   "지면(z=0) ↔ 이미지는 homography `H` 하나로 닫힌다. 2주차 IPM에서 구하는 `H_i2g`의 역행렬이 곧 합성 카메라다.",
   "아래 그림에서 **지평선 높이**와 **가까운 바닥이 어디부터 보이는지**를 확인하고, `pitch_deg`·`height_m`을 바꿔 다시 실행해 보라.")
code('horizon_v = camera.project(H_g2i, np.array([[1000.0, 0.0]]))[0, 1]',
     'near_x = camera.project(H_i2g, np.array([[cfg.camera.image_width / 2, cfg.camera.image_height]]))[0, 0]',
     'print(f"지평선 행 v = {horizon_v:.0f} px (이미지 높이 {cfg.camera.image_height}), 가장 가까운 보이는 바닥 = {near_x:.2f} m")',
     '',
     'i = 50',
     'pose = np.array([*trk.center[i], trk.heading[i]])',
     'img_raw = render.render(pose, trk.quads, None, H_g2i, cfg)      # 순수 카메라 뷰 (IPM 비교용)',
     'img = render.draw_points(img_raw.copy(), gt.waypoints_ahead(pose, trk, cfg), H_g2i)',
     'show(img, title="합성 카메라 뷰 (초록 = GT waypoint). 위 절반은 지평선 위라 모델 입력에서 잘라낸다")')

md("### 트랙 어디인가",
   "왼쪽: 카메라 뷰. 오른쪽: 차 주변 top-down (검은 띠 = gym 맵의 벽, 하늘색 부채꼴 = 카메라 화각, 보라 사각형 = BEV 범위, 초록 = GT waypoint).",
   "실차 트랙은 벽이 없고 테이프가 경계다. 벽은 gym 충돌·LiDAR에만 쓰이고 카메라 모델은 테이프만 본다.")
code('mapimg = viz.MapImage(cfg.closed_loop.map_yaml)',
     'show(viz.side_by_side(img, viz.local_view(pose, trk, gt.waypoints_ahead(pose, trk, cfg), cfg, mapimg)), width=900)',
     'overview, off = viz.draw_track_on_map(mapimg, trk, cfg)',
     'viz.mark_poses_on_map(overview, off, mapimg, [pose])',
     'show(overview, width=600, title="전체 맵에서의 위치")')

md("### BEV 세 장: 정답 / 시뮬 모델 입력 / 실차 IPM",
   "왼쪽: 지오메트리에서 직접 그린 top-down(정답). 가운데: 같은 것에 **카메라 가시 마스크**를 씌운 것 — 카메라가 못 보는 근거리(0.3 m 안쪽)와 화각 밖을 바닥색으로 가린다. 이것이 **시뮬 학습 데이터**다.",
   "오른쪽: 위 카메라 뷰를 `H_i2g`로 편 것 — 실차 2주차 IPM이 만들어 낼 BEV. 가운데와 오른쪽이 같은 모양이어야 시뮬 학습 가중치가 실차로 넘어간다.",
   "멀어질수록 오른쪽이 늘어지고 끊기는 것이 IPM의 해상도 한계다. **waypoint를 3 m 안쪽으로 잡는 이유**가 이 그림이다. `cfg.bev` 범위·해상도는 세 장이 공유한다.")
code('vis_mask = render.bev_visibility_mask(H_g2i, cfg)',
     'bev_true = render.render_bev(pose, trk.quads, cfg)',
     'bev_sim = render.render_bev(pose, trk.quads, cfg, vis_mask)      # 모델 입력',
     'bev_ipm = render.ipm_bev(img_raw, H_i2g, cfg)',
     'show(viz.side_by_side(bev_true, bev_sim, bev_ipm), width=900, title=f"정답 | 시뮬 모델 입력 | 실차 IPM   (전방 {cfg.bev.x_range_m} m, 좌우 {cfg.bev.y_range_m} m, {cfg.bev.resolution_m*1000:.0f} mm/px)")',
     'print(f"BEV 크기 {bev_sim.shape[1]} x {bev_sim.shape[0]} px, 카메라가 보는 비율 {vis_mask.mean()*100:.0f} %")',
     '',
     '# 왕복 오차: 카메라 뷰의 테이프 픽셀을 지면으로 되돌리면 원래 테이프에서 얼마나 벗어나나',
     'vs, us = np.where(np.all(img_raw == cfg.lane.color_tape, axis=-1))',
     'g = camera.project(H_i2g, np.column_stack([us, vs]).astype(float))',
     'qv = render.to_vehicle(pose, trk.quads).reshape(-1, 2)',
     'd = np.sqrt(((g[:, None, :] - qv[None, ::4, :]) ** 2).sum(-1)).min(1)',
     'print(f"IPM 왕복 오차: 중앙값 {np.median(d)*100:.1f} cm, 95% {np.percentile(d, 95)*100:.1f} cm")')

# --------------------------------------------------------------------------- 2. GT와 데이터
md("## 2. 정답(GT)과 데이터셋",
   "**주행하면서 데이터를 모으지 않는다.** 주행 가능 영역에 pose를 무작위로 뿌리고, 각 pose에서 영상을 그리고, 중심선을 따라 전방 호길이 기준으로 waypoint를 뽑는다.",
   "그래서 \"차선 이탈 직전\" 상황도 데이터에 자연히 들어가고, 모델은 복귀 동작을 따로 라벨링하지 않아도 배운다.",
   "각 샘플은 **BEV(모델 입력)** 로 직접 그려진다. 증강은 테이프 결손, pitch 지터(BEV가 휘는 워프), 블러, 글레어.",
   "왼쪽: 참고용 카메라 뷰 / 가운데: 모델이 보는 BEV / 오른쪽: 차 주변 top-down. `cfg.sampling.lateral_frac`, `heading_deg`, `augment.*` 를 바꿔 보라.")
code('rng = np.random.default_rng(0)',
     'poses = []',
     'for n in range(6):',
     '    bev, wp, p, cam = dataset.make_sample(trk, cfg, rng, with_camera=True, mask=vis_mask)',
     '    poses.append(p)',
     '    render.draw_points(cam, wp, H_g2i); render.draw_points_bev(bev, wp, cfg)',
     '    show(viz.side_by_side(cam, bev, viz.local_view(p, trk, wp, cfg, mapimg)), width=1000, title=f"샘플 {n}: 카메라 뷰(참고) | BEV 모델 입력(증강 포함) | 위치")',
     'overview, off = viz.draw_track_on_map(mapimg, trk, cfg)',
     'show(viz.mark_poses_on_map(overview, off, mapimg, poses), width=600, title="샘플 6개의 위치")')

md("### 데이터셋 저장",
   "`out/dataset/images/NNNNNN.png`(**BEV**, 모델 입력) + `labels.csv`(pose, waypoint)로 저장한다. 테이프 결손·pitch 지터는 여기서 샘플마다 들어가고, 블러·글레어는 학습 로딩 때 넣는다.",
   "20,000장에 몇 분, 수백 MB. 실차에서는 IPM으로 만든 BEV를 같은 포맷으로 저장하면 그대로 학습된다.")
code('DATA_DIR = "out/dataset"',
     't0 = time.time()',
     'dataset.generate_dataset(trk, cfg, N_DATASET, DATA_DIR, seed=0, log_every=5000)',
     'files, ds_poses, ds_wps = dataset.read_labels(DATA_DIR)',
     'print(f"{len(files)}장 저장, {time.time()-t0:.0f}s")',
     'display(pd.read_csv(f"{DATA_DIR}/labels.csv").head())')

md("### 데이터 훑어보기", "저장된 BEV에 라벨(초록)을 찍어 본다. 라벨이 테이프 사이 중심을 따라가는지, 횡 오프셋·헤딩 분포가 의도한 범위인지 확인한다.")
code('for i in np.random.default_rng(1).choice(len(files), 3, replace=False):',
     '    im = cv2.imread(f"{DATA_DIR}/images/{files[i]}")',
     '    show(render.draw_points_bev(im, ds_wps[i], cfg), width=360, title=f"{files[i]}  pose={np.round(ds_poses[i], 2)}")',
     'lat = np.array([gt.lateral_error(trk, p[:2]) for p in ds_poses])',
     'dth = np.rad2deg(np.angle(np.exp(1j * (ds_poses[:, 2] - trk.heading[[gt.nearest_index(trk, p[:2]) for p in ds_poses]]))))',
     'fig, ax = plt.subplots(1, 2, figsize=(9, 3))',
     'ax[0].hist(lat, 30); ax[0].set_xlabel("lateral offset from centerline (m)")',
     'ax[1].hist(dth, 30); ax[1].set_xlabel("heading error (deg)")',
     'plt.tight_layout(); plt.show()')

# --------------------------------------------------------------------------- 3. 학습
md("## 3. 학습",
   "작은 CNN(약 38만 파라미터)이 BEV 이미지를 받아 waypoint 좌표 12개(m)를 낸다. 손실은 Huber. BEV 해상도(`cfg.bev.resolution_m`)를 낮추면 학습이 빨라진다.",
   "train/val은 9:1로 나눈다. **train loss만 내려가고 val loss가 멈추면 오버피팅**이다. 스텝·장 수를 바꿔 비교해 보라.",
   "연한 선이 원 값, 진한 선이 5점 이동평균이다. val은 매번 약 1,000장으로 재므로 남는 흔들림은 데이터 자체의 난이도 차이(예: pitch 지터로 먼 테이프가 잘린 샘플)다.")
code('ds_train = dataset.DiskDataset(DATA_DIR, cfg, "train", val_frac=0.1)',
     'ds_val = dataset.DiskDataset(DATA_DIR, cfg, "val", val_frac=0.1, image_augment=False)',
     'print(f"train {len(ds_train)}장, val {len(ds_val)}장, device {DEVICE}")',
     '',
     'def smooth(y, k=5):',
     '    """이동평균 (양끝은 가능한 범위만). 원 데이터는 연한 선으로 같이 그린다."""',
     '    return np.array([y[max(0, i - k + 1):i + 1].mean() for i in range(len(y))])',
     '',
     'def live_plot(history):',
     '    clear_output(wait=True)',
     '    st = [h["step"] for h in history]',
     '    plt.figure(figsize=(7, 3.5))',
     '    tr = np.array([h["loss"] for h in history])',
     '    plt.plot(st, tr, color="C0", alpha=.35); plt.plot(st, smooth(tr), color="C0", label="train (smoothed)")',
     '    if "val_loss" in history[-1]:',
     '        va = np.array([h["val_loss"] for h in history])',
     '        plt.plot(st, va, color="C1", alpha=.35); plt.plot(st, smooth(va), color="C1", label="val (smoothed)")',
     '    plt.yscale("log"); plt.xlabel("step"); plt.ylabel("Huber loss"); plt.grid(alpha=.3); plt.legend()',
     '    plt.title(f"step {st[-1]}  train {history[-1][\'loss\']:.4f}" + (f"  val {history[-1][\'val_loss\']:.4f}" if "val_loss" in history[-1] else ""))',
     '    plt.show()',
     '',
     'net, hist = train.train(trk, cfg, steps=STEPS, batch_size=BATCH, lr=LR, device=DEVICE, out_path="model.pt",',
     '                        num_workers=2 if DEVICE == "cuda" else 0, log_every=max(STEPS // 30, 1),',
     '                        dataset=ds_train, val_dataset=ds_val, val_batches=32, callback=live_plot)   # val 32배치(약 1000장)로 평가해 곡선 안정화',
     'print("model.pt 저장")')

md("### waypoint별 오차", "가까운 waypoint는 잘 맞고 먼 것이 나쁘면 정상이다(카메라 해상도 한계). 전부 나쁘면 학습이 덜 된 것이다.")
code('pred = model.Predictor(net, cfg, DEVICE)',
     'r_val = train.evaluate_dataset(pred, ds_val, n=300)',
     'r_new = train.evaluate(pred, trk, cfg, n=300)',
     'x = np.arange(len(cfg.waypoints.ahead_m))',
     'plt.figure(figsize=(7, 3.5))',
     'plt.bar(x - 0.2, r_val["per_waypoint_m"] * 100, 0.4, label=f"val, saved images (n={r_val[\'n\']})")',
     'plt.bar(x + 0.2, r_new["per_waypoint_m"] * 100, 0.4, label="fresh poses x300, no aug")',
     'plt.xticks(x, [f"{a:g} m" for a in cfg.waypoints.ahead_m]); plt.ylabel("mean error (cm)"); plt.legend(); plt.grid(axis="y", alpha=.3)',
     'plt.title(f"mean {r_new[\'mean_m\']*100:.1f} cm"); plt.show()',
     '',
     'bev, wp, p, cam = dataset.make_sample(trk, cfg, np.random.default_rng(7), do_augment=False, with_camera=True, mask=vis_mask)',
     'wp_pred = pred.predict(bev)',
     'render.draw_points_bev(bev, wp, cfg, (0, 255, 0)); render.draw_points_bev(bev, wp_pred, cfg, (255, 0, 255))',
     'render.draw_points(cam, wp, H_g2i, (0, 255, 0)); render.draw_points(cam, wp_pred, H_g2i, (255, 0, 255))',
     'show(viz.side_by_side(cam, bev), width=900, title="초록 = 정답, 자홍 = 모델 예측  (왼쪽 카메라 뷰는 참고용, 모델은 오른쪽 BEV만 본다)")')

# --------------------------------------------------------------------------- 4. 폐루프
md("## 4. 폐루프 검증 (gym, ROS 없음)",
   "매 제어 틱마다 BEV 렌더(가시 마스크 포함) → 모델 추론 → pure pursuit → `env.step`. 종료 조건은 실차 규칙과 같다: **차체 모서리가 테이프를 넘으면 실격**(`tape_crossed`), 한 바퀴 돌면 `lap`.",
   "",
   "코랩은 실시간이 아니라 지연이 0이 된다. 젯슨의 지연을 흉내내는 버퍼(`latency_steps`, 1틱 = 1/control_hz 초)를 꼭 넣는다.",
   "먼저 **오라클**(정답 waypoint + 가우시안 노이즈)로 \"인지 오차 σ와 지연이 얼마까지면 완주하는가\"를 표로 만든다. 이것이 배포 게이트의 기준이 된다.")
code('env = closed_loop.make_env(cfg)',
     'rows = closed_loop.sweep(env, trk, cfg, H_g2i, latency_list=LATENCIES, sigma_list=SIGMAS)',
     'df = pd.DataFrame(rows); json.dump(rows, open("out/sweep_oracle.json", "w"), indent=1)',
     'print(f"제어 주기 {cfg.closed_loop.control_hz} Hz, 속도 {cfg.closed_loop.speed_mps} m/s, lookahead {cfg.closed_loop.lookahead_m} m")',
     'print("종료 이유 (lap = 완주, tape_crossed = 실격):")',
     'display(df.pivot(index="latency_steps", columns="sigma", values="reason"))',
     'print("평균 횡오차 (m):")',
     'display(df.pivot(index="latency_steps", columns="sigma", values="mean_lateral_m").round(3))')

md("### 학습 모델로 주행", "지연을 바꿔 가며 달리고 횡오차 곡선을 겹쳐 그린다. 영상은 지연 0 주행: 왼쪽 카메라 뷰(참고), 오른쪽 모델 입력 BEV, 초록 = 모델 예측 waypoint.")
code('results = {}',
     'plt.figure(figsize=(8, 3.5))',
     'for lat in LATENCIES:',
     '    r = closed_loop.run(env, pred, trk, cfg, H_g2i, latency_steps=lat, video_path="out/run_latency0.mp4" if lat == LATENCIES[0] else None)',
     '    results[lat] = r',
     '    t = np.arange(r.steps) / r.control_hz_eff',
     '    plt.plot(t, r.lateral_trace * 100, label=f"latency {lat} ticks -> {r.reason} ({r.progress_m:.0f} m)")',
     '    print(f"latency={lat:2d}: {r.reason:12s} 진행 {r.progress_m:6.1f} m  횡오차 평균 {r.mean_lateral_m*100:5.1f} cm  최대 {r.max_lateral_m*100:5.1f} cm")',
     'inner = (cfg.lane.track_width_m - cfg.lane.tape_width_m) / 2 - cfg.closed_loop.car_width_m / 2',
     'plt.axhline(inner * 100, color="r", ls="--", label="body touches tape (straight)")',
     'plt.xlabel("time (s)"); plt.ylabel("lateral error (cm)"); plt.legend(fontsize=8); plt.grid(alpha=.3); plt.show()',
     'display(Video(viz.to_h264("out/run_latency0.mp4"), embed=True, width=900))   # mp4v -> H.264 변환 후 표시 (브라우저 재생용)')

md("### 맵 위의 주행 경로",
   "검은 점선 = 중심선(GT 경로), 노란 선 = 테이프, 회색 = 오라클(정답 waypoint, 지연 0) 주행, 색 선 = 학습 모델 주행(지연별).",
   "빈 원이 출발, 채운 원이 종료 지점이다. 실격이면 그 자리에서 끊긴다. 아래는 지연 0 주행의 종료 지점 확대.")
code('paths = {}',
     'r_or = closed_loop.run(env, model.OraclePredictor(trk, cfg), trk, cfg, H_g2i, latency_steps=0)',
     'paths[f"oracle -> {r_or.reason} ({r_or.progress_m:.0f} m)"] = r_or.pose_trace',
     'for lat, r in results.items():',
     '    paths[f"model latency {lat} -> {r.reason} ({r.progress_m:.0f} m)"] = r.pose_trace',
     'pm, off = viz.draw_paths_on_map(mapimg, trk, cfg, paths)',
     'pm_colors = list(paths.keys())',
     'show(pm, width=900, title="전체 맵: 경로 비교")',
     'r0 = results[LATENCIES[0]]',
     'show(viz.crop_around(pm, off, mapimg, r0.pose_trace[-1], half_m=3.0, scale=3), width=500, title=f"지연 {LATENCIES[0]} 주행의 종료 지점 ({r0.reason})")')

# --------------------------------------------------------------------------- 5. 보관
md("## 5. 결과 보관",
   "이 파일들은 코랩 VM에 있고 런타임이 끊기면 사라진다. 아래 셀은 구글 드라이브가 마운트 가능할 때(브라우저 코랩) `model.pt`와 결과를 복사한다.",
   "VSCode 원격 커널에서는 마운트가 안 되므로 필요한 값은 셀 출력으로 확인하거나 브라우저 코랩에서 다운로드한다.")
code('try:',
     '    from google.colab import drive',
     '    drive.mount("/content/drive")',
     '    dst = "/content/drive/MyDrive/camsim_results"; os.makedirs(dst, exist_ok=True)',
     '    !cp model.pt out/sweep_oracle.json out/run_latency0.mp4 $dst/',
     '    !cd out && zip -qr $dst/dataset.zip dataset',
     '    print("복사 완료:", dst)',
     'except Exception as e:',
     '    print("드라이브 마운트 불가 (VSCode 원격 커널 등):", type(e).__name__)',
     '    print("보관할 파일: model.pt, out/sweep_oracle.json, out/run_latency0.mp4, out/dataset/")')

nb = {"cells": cells,
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(os.path.dirname(__file__), "..", "..", "notebooks", "camsim_lab.ipynb")
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(nb, open(out, "w"), indent=1, ensure_ascii=False)
print("wrote", os.path.normpath(out), len(cells), "cells")
