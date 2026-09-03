# camsim: 합성 카메라 waypoint 파이프라인 설계

작성일: 2026-09-03
상위 문서: `camera_waypoint_pipeline.md` (§3, §4, §5, §7)
범위: 코랩/로컬에서 도는 부분만. 실차 녹화·젯슨 배포는 제외.

## 1. 목표

`f1tenth_gym`은 카메라를 렌더링하지 못한다. 바닥 테이프 트랙은 전부 z=0 평면이므로 homography 하나로 가짜 전방 영상을 그릴 수 있다. 이 패키지는 다음 세 가지를 제공한다.

1. pose 하나를 주면 그 위치에서 보이는 테이프 영상을 그리는 렌더러
2. 주행 없이 pose를 뿌려 만드는 온더플라이 학습 데이터와 학습 스크립트
3. gym 안에서 렌더·추론·pure pursuit를 한 루프로 돌리는 폐루프 검증

학생이 손대는 것은 `camsim/config.yaml` 하나다.

## 2. 확정된 결정

| 항목 | 결정 | 이유 |
|---|---|---|
| gym | 이 레포의 `f110_gym` (gym 0.19) 그대로 | 수업 자료와 일치 |
| 로컬 환경 | uv, Python 3.9 가상환경 | numpy<=1.22 제약 |
| 트랙 | `examples/example_waypoints.csv` 중심선 + 양옆 0.4m 테이프 | 추가 에셋 없이 오늘 시작 |
| 코스 아웃 | 중심선 이탈 0.4m 초과 시 종료 | 이 맵은 벽이 테이프보다 훨씬 멀다 |
| 카메라 | FLIR Blackfly S BFS-U3-23S3C, 센서 6.62×4.14mm, 1920×1200 | 데이터시트 |
| 렌즈 | 수평 화각 90° (초점거리 약 3.3mm) | 사용자 확인 |
| 마운트 | 정면 (pitch 0°), 높이 0.2m (가정) | 마운트 미정 |
| 렌더 해상도 | 640×400 (센서 비율 1.6 유지) | 실차도 1920×1200을 여기로 축소 |
| 모델 입력 | 지평선 아래 절반 크롭 → 640×200 | pitch 0이면 위 절반은 바닥이 아님 |
| waypoint | 전방 호길이 0.5, 1.0, 1.5, 2.0, 2.5, 3.0m | 0.2m 높이·정면에서 3m 이상은 해상도 부족 |

실제 캘리브레이션 `H_i2g`가 나오면 `camera.h_i2g_file`에 경로를 적는다. 그러면 위 카메라 가정값은 무시된다.

## 3. 패키지 구조

```
camsim/
  config.yaml      학생이 편집하는 유일한 파일
  config.py        yaml 로드 → dataclass. 키 누락 시 명확한 에러
  camera.py        가정값 → K, R|t → H_g2i / H_i2g. 또는 파일에서 H 로드
  track.py         CSV → 5cm 리샘플 중심선, 헤딩, 누적 호길이, 테이프 quad
  render.py        (pose, quads, scan, H_g2i) → 640×400 BGR 이미지
  gt.py            sample_pose, waypoints_ahead
  augment.py       pitch 지터, 모션 블러, 테이프 결손, 글레어
  dataset.py       IterableDataset: 매 샘플 pose 샘플링 → 렌더 → 증강 → 크롭
  model.py         작은 CNN + predict(img) -> (6,2) 래퍼
  train.py         학습 루프, 오프라인 지표
  pure_pursuit.py  (6,2) waypoint → (steer, speed)
  closed_loop.py   gym 루프, 지연 버퍼, 지표, mp4
  tests/
notebooks/
  NB1_render.ipynb, NB2_train.ipynb, NB3_closed_loop.ipynb  (git clone 후 import만)
```

### 3.1 config.yaml

```yaml
camera:
  image_width: 640        # 렌더/입력 해상도
  image_height: 400
  sensor_width_mm: 6.62   # BFS-U3-23S3 데이터시트
  sensor_height_mm: 4.14
  hfov_deg: 90            # 렌즈 수평 화각
  height_m: 0.20          # 바닥에서 렌즈 중심까지
  pitch_deg: 0.0          # 아래로 숙인 각도 (+가 아래)
  offset_x_m: 0.0         # 후륜축 기준 카메라 전방 오프셋 (미정이면 0)
  h_i2g_file: null        # 실측 캘리브레이션이 있으면 .npy 경로

lane:
  track_width_m: 0.8      # 좌우 테이프 중심 사이 거리
  tape_width_m: 0.05
  segment_len_m: 0.05     # 테이프를 쪼개는 사각형 길이
  color_floor: [200, 200, 200]   # BGR
  color_tape: [30, 30, 220]

render:
  near_m: 0.15
  far_m: 10.0
  lidar_fov_deg: 270      # gym 기본값

waypoints:
  ahead_m: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

sampling:
  lateral_frac: 0.35      # ±track_width*frac
  heading_deg: 15.0

augment:
  pitch_jitter_deg: 2.0
  blur_max_px: 3
  tape_dropout_prob: 0.3
  glare_prob: 0.3

closed_loop:
  lookahead_m: 1.2
  speed_mps: 2.0
  latency_steps: 0
  max_steps: 4000
  offtrack_m: 0.4
```

### 3.2 좌표계

- world: gym 맵 좌표 (m). 중심선 CSV와 동일.
- vehicle: 차량 후륜축 중심, x 전방, y 좌측.
- camera ground: vehicle과 동일하다고 둔다 (카메라가 후륜축 위에 있다고 가정, `camera.offset_x_m`으로 나중에 보정 가능하도록 키만 예약).
- image: OpenCV 픽셀 (u 우측, v 아래).

`H_g2i`는 ground (x, y, 1) → image (u, v, 1). pitch 0이면 지평선은 v = image_height/2.

### 3.3 camera.py

가정값에서 H를 만드는 절차:

1. 초점거리 px: `fx = (image_width/2) / tan(hfov/2)`. 픽셀이 정사각(3.45µm)이고 렌더 해상도가 센서 비율을 유지하므로 `fy = fx`. 주점은 이미지 중심.
2. 카메라 좌표: 광축 z, 우측 x, 아래 y. vehicle (X 전방, Y 좌측, Z 위) → camera는 `Rc = [[0,-1,0],[0,0,-1],[1,0,0]]`에 pitch 회전을 곱한다.
3. `H_g2i = K @ [r1 r2 t]` (Z=0 평면). `H_i2g = inv(H_g2i)`.

`h_i2g_file`이 있으면 파일을 읽고 그 역행렬로 `H_g2i`를 만든다. 둘 다 같은 함수 시그니처로 나온다.

### 3.4 track.py

- CSV 파싱: `;` 구분, 3줄 헤더, x=열1, y=열2. 헤딩은 CSV의 psi를 쓰지 않고 리샘플 후 차분으로 다시 계산 (리샘플과 일관).
- 5cm 등간격 리샘플 (호길이 기준 `np.interp`).
- 좌/우 테이프 중심선 = 중심선 ± normal × track_width/2.
- quads: 각 테이프를 `segment_len_m` 길이, `tape_width_m` 폭의 사각형으로. shape (N, 4, 2).
- 출력: `Track(center, heading, s_cum, quads)`; `save/load npz`.

### 3.5 render.py

상위 문서 §3.2 그대로. 이미지 밖 폴리곤은 `fillPoly`가 클리핑한다. 카메라 뒤(x < near) 사각형은 투영 전에 제거해야 한다 (w ≤ 0이면 부호가 뒤집힘). LiDAR scan이 None이면 가림 처리를 건너뛴다 (NB1은 gym 없이 돈다).

### 3.6 gt.py

- `sample_pose(track, rng)`: 상위 문서 §4.1.
- `waypoints_ahead(pose, track)`: §4.2. `searchsorted` 결과를 `% len`으로 감싸 트랙 끝을 넘긴다. 반환 (6,2) vehicle 좌표.

### 3.7 augment.py

전부 확률적으로 적용. pitch 지터는 H를 바꾸는 것이므로 렌더 **전에** 적용된다 (`camera.build_h(pitch + jitter)`). 나머지 세 개는 이미지에 적용. 글로벌 셔터라 블러는 약하게.

### 3.8 dataset.py / model.py / train.py

- `IterableDataset`. `__iter__`가 무한히 (img_crop, target) 를 낸다. worker마다 seed 분리.
- 모델: 입력 (3, 200, 640), conv 5단 + GAP + FC 12. 파라미터 1M 이하. 타깃은 m/1.0.
- 손실 Huber. 지표: waypoint별 유클리드 오차 m, 전체 평균, 최대.
- `Predictor(model, cfg).predict(img_bgr) -> (6,2)`: 크롭, 정규화, 추론, 역정규화. closed_loop과 ROS2 노드가 공유할 인터페이스.

### 3.9 closed_loop.py

```
env.reset(pose0) → obs
buf = deque([zeros]*latency_steps)
loop:
  img  = render(pose(obs), track.quads, obs['scans'][0], H_g2i)
  buf.append(predictor.predict(img)); wp = buf.popleft()
  steer, speed = pure_pursuit(wp, lookahead, speed)
  obs = env.step([[steer, speed]])
  lateral = dist(pose, track.center)
  stop if collision or lateral > offtrack_m or step > max_steps
```

지표: 완주 여부(호길이 한 바퀴), 평균/최대 횡오차, 걸린 스텝. `sweep(latency_steps=[0,2,4,6], noise_sigma=[...])`가 표를 낸다. mp4는 우리 렌더러 출력에 waypoint를 찍어 저장.

## 4. 검증

| 테스트 | 기준 |
|---|---|
| IPM 왕복 | quad 정점을 `H_g2i`로 보내고 `H_i2g`로 되돌리면 1cm 이내 |
| 지평선 | pitch 0에서 먼 지점(100m)의 v가 image_height/2 ± 1px |
| 렌더 컬링 | 카메라 뒤 quad, far 밖 quad, scan보다 먼 quad는 안 그려짐 |
| GT 단조 | 임의 pose에서 6개 waypoint의 호길이가 증가 |
| GT 부호 | 좌회전 구간에서 waypoint y > 0 |
| 렌더 속도 | 1000회 평균 2ms 이내 (로컬 CPU) |
| config | 키 누락 시 어떤 키인지 알려주는 에러 |

## 5. 공유 인터페이스

- `camsim.config.load(path) -> Config`
- `camsim.camera.build(cfg) -> (H_g2i, H_i2g)`
- `camsim.track.from_csv(path, cfg) -> Track`
- `camsim.render.render(pose, track, scan, H_g2i, cfg) -> np.ndarray[H,W,3]`
- `camsim.gt.sample_pose(track, cfg, rng)`, `camsim.gt.waypoints_ahead(pose, track, cfg)`
- `camsim.model.Predictor.predict(img) -> np.ndarray[6,2]`
- `camsim.closed_loop.run(env, predictor, track, cfg) -> Result`

## 6. 제외

- 커스텀 실습실 트랙과 맵 PNG 생성 (다음 단계)
- ROS2 노드, HIL, TensorRT
- 실차 자동 라벨링
