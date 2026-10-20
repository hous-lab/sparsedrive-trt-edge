#!/bin/bash
# Power-mode sweep on Orin: for each nvpmodel power mode, run the engine
# latency bench + trtexec/tegrastats power sampling. Results per mode land in
# arm64/eval/results/power/<mode>/. Accuracy is mode-independent, so only
# latency/power are re-measured per mode.
#
# Usage (on Orin): sudo bash arm64/eval/run_power_modes.sh
# Env: MODES (default "MAXN:0 50W:1 30W:2 15W:3" — check `nvpmodel -q` on your
#      board; IDs differ between AGX Orin / Orin NX / Orin Nano)
set -e
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_ROOT="$REPO/arm64/eval/results/power"
MODES="${MODES:-MAXN:0 50W:1 30W:2 15W:3}"

for m in $MODES; do
  name="${m%%:*}"; id="${m##*:}"
  echo "===== power mode: $name (nvpmodel $id) ====="
  sudo nvpmodel -m "$id"
  if [ "$name" = "MAXN" ]; then
    sudo jetson_clocks
  fi
  sleep 5  # settle
  SD_POWER_OUT="$OUT_ROOT/$name" bash arm64/eval/bench_power_orin.sh
  SD_RESULTS_DIR="$OUT_ROOT/$name" python3 arm64/eval/bench_engines.py
done

echo "===== power sweep done; results in $OUT_ROOT ====="
