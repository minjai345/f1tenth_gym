#!/usr/bin/env bash
# gym 0.19.0의 setup.py에 있는 'opencv-python>=3.' (잘못된 버전 표기) 때문에 최신 setuptools에서 설치가 실패한다.
# sdist를 받아 한 글자 고친 뒤 설치한다. 로컬(uv venv)과 코랩 모두 이 스크립트를 쓴다.
# 사용법: bash camsim/scripts/install_gym019.sh [python 실행 파일]   (기본: python)
set -euo pipefail
PY="${1:-python}"
TMP="$(mktemp -d)"
curl -sL https://files.pythonhosted.org/packages/source/g/gym/gym-0.19.0.tar.gz -o "$TMP/gym.tgz"
tar xzf "$TMP/gym.tgz" -C "$TMP"
sed -i 's/opencv-python>=3\./opencv-python>=3/' "$TMP/gym-0.19.0/setup.py"
"$PY" -m pip install --no-build-isolation "$TMP/gym-0.19.0" 2>/dev/null \
  || uv pip install --python "$PY" --no-build-isolation "$TMP/gym-0.19.0"
rm -rf "$TMP"
