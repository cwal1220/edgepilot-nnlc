#!/usr/bin/env bash
# 녹화마다 edgepilot이 그 순간 학습 모델에 넣었을 입력(운용 입력)을 만들어 데이터 폴더에 <경로>.inputs.csv.gz로 둔다.
# 계획 조건 학습(nnlc.runtime)의 입력이다. edgepilot의 replay_controls를 열린 루프로 돌린다: 녹화된 모델 출력과 차
# 상태로 플래너와 컨트롤러가 계획 궤적·이력을 만들고, 모델을 계산한 틱마다 그 입력을 쓴다.
#
#   scripts/dump_runtime_inputs.sh <edgepilot 저장소> <녹화 폴더(경로마다 하위 폴더)> <데이터 폴더> [경로 이름...]
#
# - 경로 이름을 주지 않으면 데이터 폴더의 CSV(<경로>.csv, .csv.gz)가 있는 경로를 모두 돈다.
# - 설정은 edgepilot 저장소의 params/steering.json(기본값)이고, 학습기는 녹화 당시 params/live_parameters.json에서
#   시작한다(있으면). 녹화 전에 결합해 둔 경로가 있어 언제나 버튼 없이 결합한다(--force-engaged). 학습 표본은 그 순간
#   실제로 조향했는지(CSV active)로 고르므로 결합 상태가 녹화와 달라도 된다.
# - 입력 이름은 k7_hev_v3의 33개다. 그보다 적은 입력을 쓰는 설정은 이름으로 골라 쓴다.
# - 20 Hz(5틱에 한 번)로 쓴다. 학습 표본 간격(dataset.Spec.stride 5)과 같다.
set -euo pipefail
[ $# -ge 3 ] || { sed -n 2,13p "$0"; exit 2; }
edgepilot=$(cd "$1" && pwd)
records=$(cd "$2" && pwd)
out=$(cd "$3" && pwd)
shift 3
replay="$edgepilot/build-host/bin/replay_controls"
[ -x "$replay" ] || { echo "$replay missing: build edgepilot's host tools (scripts/run_host_tests.sh)"; exit 1; }
model="$edgepilot/models/lateral_nn/k7_hev_v3.json"
routes=("$@")
if [ ${#routes[@]} -eq 0 ]; then
  for csv in "$out"/*.csv "$out"/*.csv.gz; do
    [ -e "$csv" ] || continue
    name=$(basename "$csv"); name=${name%.gz}; name=${name%.csv}
    case "$name" in *.inputs) continue ;; esac
    routes+=("$name")
  done
fi
for route in $(printf '%s\n' "${routes[@]}" | sort -u); do
  dir="$records/$route"
  [ -d "$dir/events" ] || { echo "skip $route: no $dir/events"; continue; }
  args=(--force-engaged --steering "$edgepilot/params/steering.json" --lateral-nn "$model")
  [ -f "$dir/params/live_parameters.json" ] && args+=(--vehicle-json "$dir/params/live_parameters.json")
  tmp="$out/$route.inputs.csv"
  "$replay" "${args[@]}" --nn-inputs "$tmp" --nn-inputs-every 5 "$dir"/events/*.bin > /dev/null 2>&1
  gzip -9 -f "$tmp"
  echo "$route: $(($(gzip -dc "$tmp.gz" | wc -l) - 1)) rows"
done
