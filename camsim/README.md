# camsim — 합성 카메라 waypoint 파이프라인

`f1tenth_gym`은 카메라를 못 그린다. 바닥 테이프 트랙은 전부 평면이므로 캘리브레이션 행렬 하나로
가짜 전방 영상을 그릴 수 있다. 이 패키지는 그 렌더러, 온더플라이 학습, gym 폐루프 검증을 담는다.
설계: `docs/superpowers/specs/2026-09-03-camsim-design.md`. 배경 설명은 강의 자료 참고.

## 학생이 손대는 파일
`camsim/config.yaml` 하나. 카메라 높이/각도/화각, 테이프 폭, waypoint 거리, 지연, 속도가 전부 여기 있다.
2주차 캘리브레이션이 끝나면 `camera.h_i2g_file`에 `.npy` 경로를 적는다. 그러면 가정 카메라 값은 무시된다.

## 로컬 설치 (Python 3.9)
    uv venv --python 3.9 .venv
    uv pip install --python .venv/bin/python setuptools==65.5.0 wheel numpy==1.22.0
    bash camsim/scripts/install_gym019.sh .venv/bin/python
    uv pip install --python .venv/bin/python --no-build-isolation -e .
    uv pip install --python .venv/bin/python "opencv-python-headless<4.10" pytest pyyaml
    uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
    .venv/bin/python -m pytest camsim/tests -q

gym 0.19의 `setup.py`는 pip이 그대로 못 읽는 요구사항 문자열을 갖고 있어서, 일반 `pip install gym==0.19.0`이
실패한다. 그래서 gym은 `camsim/scripts/install_gym019.sh`로 따로 설치한다.

## 실행: 노트북 하나
학생용 경로는 `notebooks/camsim_lab.ipynb` 하나다. 0장(설치·파라미터) → 1장 카메라와 트랙 → 2장 GT와 데이터셋 → 3장 학습(train/val loss 실시간) → 4장 폐루프(스윕 표, 횡오차 곡선, 영상) → 5장 보관.
각 장 첫 셀의 파라미터를 바꾸고 그 장을 다시 실행하면서 진행한다. `config.yaml`은 기본값이고, 노트북의 파라미터 셀이 그 위에 덮어쓴다.
노트북 원본은 `camsim/scripts/build_notebook.py` 다 (수정 후 재생성). 로컬 검증:

    .venv/bin/python camsim/scripts/build_notebook.py
    cp notebooks/camsim_lab.ipynb _lab.ipynb && CAMSIM_SMOKE=1 <py3.12+ python> -m jupyter nbconvert --to notebook --execute _lab.ipynb --output /tmp/lab_out.ipynb

(`CAMSIM_SMOKE=1` 이면 데이터 300장·30스텝으로 축소. 레포 루트에서 실행해야 첫 셀이 clone을 건너뛴다.)

`camsim/scripts/` 의 `nb1_render.py`, `make_dataset.py`, `nb2_train.py`, `nb3_closed_loop.py` 는 같은 단계의 헤드리스 버전으로, 로컬 테스트용이다.

## 모델 입력은 BEV
시뮬: `render.render_bev(pose, quads, cfg, mask)` 가 테이프를 top-down 으로 직접 그린다. `mask = render.bev_visibility_mask(H_g2i, cfg)` 로
카메라가 못 보는 영역(근거리 사각, 화각 밖)을 바닥색으로 가려 실차 IPM 출력과 같은 모양을 만든다.
실차: 카메라 → undistort → `render.ipm_bev(cam, H_i2g, cfg)` → 같은 규격의 BEV. `Predictor.predict(bev)` 하나를 양쪽이 공유하고,
실차용 편의 함수로 `Predictor.predict_camera(cam, H_i2g)` 가 있다. 원근 렌더(`render.render`)는 시각화·IPM 비교용이다.
BEV 범위·해상도는 `config.yaml` 의 `bev:` 섹션 하나로 정한다 (시뮬·실차 공용).

## 데이터셋
`dataset.generate_dataset` (스크립트 `make_dataset.py N`) 이 `out/dataset/images/NNNNNN.png`(BEV, 모델 입력) 와
`labels.csv`(file, x, y, theta, wp0_x..wp5_y) 를 만든다. 저장 데이터는 증강 없는 plain 이다. train/val 은 9:1 결정적 분리.
증강은 `DiskDataset(..., augment_fn=fn)` 의 `fn(bev, rng) -> bev` 하나로 로딩 때 넣는다 (기본 None).
`camsim/augment.py` 의 `jitter_bev`(pitch 워프), `erase_patches`, `glare`, `motion_blur`, `example_augment` 는 예시이며
세기는 config `augment:` 로 조절한다 (기본 전부 off). `train.evaluate(..., degrade_fn=fn)` 으로 열화 입력에 대한 강건성을 잰다.
sim-to-real 증강 설계는 노트북 3장 과제다. 실차에서 IPM으로 만든 BEV를 같은 `labels.csv` 포맷으로 저장하면 그대로 학습된다.
코랩 로컬 디스크에 생성하고 드라이브에는 zip 하나로 옮길 것 (드라이브는 작은 파일 다량에 매우 느리다).

## 실격 규칙 (폐루프 종료 조건)
실차 트랙은 벽 없이 테이프가 경계다. 시뮬도 같은 규칙을 쓴다: 차체(`closed_loop.car_length_m` x `car_width_m`) 네 모서리 중
하나라도 테이프 안쪽 선(중심선에서 `track_width_m/2 - tape_width_m/2`)을 넘으면 `reason="tape_crossed"`로 종료. 완주(`lap`)는
테이프를 한 번도 안 넘고 한 바퀴 돈 것이다. gym 벽 충돌(`collision`)은 이 맵에서는 벽이 멀어 거의 안 난다.

## 좌표계
world = gym 맵 (m). vehicle = 후륜축, x 전방, y 좌측. image = OpenCV (u 우, v 아래).
`H_g2i`: ground (x,y,1) → image. pitch 0이면 지평선은 이미지 세로 중앙.

## 코랩 주의
- 코랩 Python(3.13)에는 numpy 1.22 wheel이 없다. 그래서 코랩에서는 numpy를 코랩 기본(2.x) 그대로 두고 f110_gym을
  `pip install --no-deps -e .` 로 설치한다 (numpy 2.5에서 테스트 60개 통과 확인). 노트북 첫 셀이 이 순서대로 되어 있다.
- numpy를 설치 중에 바꾸면 런타임을 재시작해야 한다. 재시작 없이 진행하면 `numpy.dtype size changed` 오류가 난다.
- 주행 영상(mp4)이 셀에 안 뜨면 코덱 문제다. OpenCV의 mp4v는 브라우저가 못 읽으므로 `viz.to_h264()` 로 변환해 띄운다 (노트북은 이미 그렇게 함).
- `pyglet` import 에러가 나면 `!apt-get install -y libgl1` 후 재시도.
- 노트북 첫 코드 셀의 `REPO_URL` 기본값은 조교 fork(`https://github.com/minjai345/f1tenth_gym.git`, `main`)다.
  다른 fork를 쓰려면 그 줄만 바꾸면 된다. clone이 실패하면 `%cd f1tenth_gym`부터 전부 깨지므로 주소를 먼저 확인할 것.
