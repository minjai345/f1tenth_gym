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

## 실행 순서
| | 스크립트 | 노트북 | gym 필요 |
|---|---|---|---|
| 렌더러 + IPM 왕복 | `camsim/scripts/nb1_render.py` | `notebooks/NB1_render.ipynb` | ✗ |
| 학습 | `camsim/scripts/nb2_train.py STEPS BATCH DEVICE` | `NB2_train.ipynb` | ✗ |
| 폐루프 + 스윕 | `camsim/scripts/nb3_closed_loop.py model.pt` | `NB3_closed_loop.ipynb` | ✓ |

세 스크립트 모두 저장소 루트에서 실행해야 한다(`sys.path.insert(0, os.getcwd())`). 노트북은
`%run` 전에 `%cd f1tenth_gym`을 하므로 동일하게 루트에서 실행되는 셈이다.
스크립트를 다시 실행하면 `out/`, `track.npz`, `model.pt`가 덮어써진다.

## 좌표계
world = gym 맵 (m). vehicle = 후륜축, x 전방, y 좌측. image = OpenCV (u 우, v 아래).
`H_g2i`: ground (x,y,1) → image. pitch 0이면 지평선은 이미지 세로 중앙.

## 코랩 주의
- 코랩 Python(3.13)에는 numpy 1.22 wheel이 없다. 그래서 코랩에서는 numpy를 코랩 기본(2.x) 그대로 두고 f110_gym을
  `pip install --no-deps -e .` 로 설치한다 (numpy 2.5에서 테스트 60개 통과 확인). 노트북 첫 셀이 이 순서대로 되어 있다.
- numpy를 설치 중에 바꾸면 런타임을 재시작해야 한다. 재시작 없이 진행하면 `numpy.dtype size changed` 오류가 난다.
- `pyglet` import 에러가 나면 `!apt-get install -y libgl1` 후 재시도.
- 노트북 첫 코드 셀의 `REPO_URL` 기본값은 조교 fork(`https://github.com/minjai345/f1tenth_gym.git`, `main`)다.
  다른 fork를 쓰려면 그 줄만 바꾸면 된다. clone이 실패하면 `%cd f1tenth_gym`부터 전부 깨지므로 주소를 먼저 확인할 것.
