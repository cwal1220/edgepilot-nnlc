#!/bin/bash
# edgepilot extract_lateral_dataset로 녹화 경로마다 CSV 하나를 만든다.
#   scripts/extract_routes.sh <edgepilot 빌드 dir> <녹화 경로들이 있는 dir> <출력 dir>
# 예: scripts/extract_routes.sh ~/Documents/ax630/edgepilot/build-host ~/Documents/ax630/experiments data
# 이벤트 파일 목록은 배열로 넘긴다(zsh에서 변수로 넘기면 한 인자로 붙어 0행이 된다).
set -euo pipefail
tool="$1/bin/extract_lateral_dataset"
routes="$2"
out="$3"
[ -x "$tool" ] || { echo "no $tool (cmake --build <build> --target extract_lateral_dataset)" >&2; exit 2; }
mkdir -p "$out"
for route in "$routes"/20*/; do
    name=$(basename "$route")
    if [ -d "$route/events" ]; then
        files=("$route"/events/*.bin)
    elif [ -f "$route/events.bin" ]; then
        files=("$route/events.bin")
    else
        continue
    fi
    "$tool" "$out/$name.csv" "${files[@]}" 2>&1 | grep -v truncated | tail -1
done
