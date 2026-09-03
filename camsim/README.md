# camsim — 합성 카메라 waypoint 파이프라인

`f1tenth_gym`은 카메라를 못 그린다. 바닥 테이프 트랙은 전부 평면이므로 캘리브레이션 행렬 하나로
가짜 전방 영상을 그릴 수 있다. 이 패키지는 그 렌더러, 온더플라이 학습, gym 폐루프 검증을 담는다.
설계: `docs/superpowers/specs/2026-09-03-camsim-design.md`, 배경: `camera_waypoint_pipeline.md`.

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

## 실행 순서
| | 스크립트 | 노트북 | gym 필요 |
|---|---|---|---|
| 렌더러 + IPM 왕복 | `camsim/scripts/nb1_render.py` | `notebooks/NB1_render.ipynb` | ✗ |
| 학습 | `camsim/scripts/nb2_train.py STEPS BATCH DEVICE` | `NB2_train.ipynb` | ✗ |
| 폐루프 + 스윕 | `camsim/scripts/nb3_closed_loop.py model.pt` | `NB3_closed_loop.ipynb` | ✓ |

세 스크립트 모두 저장소 루트에서 실행해야 한다(`sys.path.insert(0, os.getcwd())`). 노트북은
`%run` 전에 `%cd f1tenth_gym`을 하므로 동일하게 루트에서 실행되는 셈이다.

## 좌표계
world = gym 맵 (m). vehicle = 후륜축, x 전방, y 좌측. image = OpenCV (u 우, v 아래).
`H_g2i`: ground (x,y,1) → image. pitch 0이면 지평선은 이미지 세로 중앙.

## 코랩 주의
- `pip install -r camsim/requirements.txt` 후 numpy가 바뀌면 런타임 재시작 한 번.
- `pyglet` import 에러가 나면 `!apt-get install -y libgl1` 후 재시도.
- 노트북의 clone URL(`https://github.com/f1tenth/f1tenth_gym.git`)은 아직 이 브랜치가 없는 자리표시자다.
  강사 fork로 바꿀 것.
