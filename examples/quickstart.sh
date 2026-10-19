#!/bin/bash
# 저장소에 든 K7 HEV 데이터로 처음부터 끝까지: 커버리지 → 고른 설정의 교차 검증 → 전체 주행으로 학습해 모델 파일.
#   examples/quickstart.sh            (저장소 루트에서, .venv 준비 뒤)
# 결과는 runs/(추적하지 않음)에 쓴다. MPS(Apple GPU)에서 교차 검증 2~3분, 학습 30초쯤.
set -euo pipefail
cd "$(dirname "$0")/.."
py=${PY:-.venv/bin/python}
data=${1:-dataset/k7_hev}
"$py" scripts/coverage.py "$data"
chosen=$("$py" -c 'from nnlc.configs import CHOSEN; print(CHOSEN)')  # 고른 설정(운용 입력이라 .inputs.csv.gz가 쓰인다)
"$py" scripts/experiments.py "$data" runs/quickstart_cv.json "$chosen"
"$py" scripts/train_final.py "$data" runs/quickstart_model.json --no-cv
