#!/bin/bash
# DLA power experiment (C1/C2/C3): run a backbone engine continuously under
# trtexec while tegrastats samples power. Latency from trtexec, power from
# tegrastats logs.
#
# Usage (on Orin): bash arm64/eval/bench_power_orin.sh
# Env overrides:
#   TRTEXEC            trtexec binary
#   SD_ENGINE_DIR      FP16 engines dir
#   SD_ENGINE_INT8_DIR INT8 engines dir
#   SD_DLA_ENGINE      DLA INT8 backbone engine (default .../backbone_dla0.engine)
#   SD_POWER_OUT       output dir for logs/results
set -e
TRTEXEC="${TRTEXEC:-/usr/src/TensorRT-10.4.0.26/targets/aarch64-linux-gnu/bin/trtexec}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FP16_DIR="${SD_ENGINE_DIR:-$REPO/arm64/trt_compile/fp16/output}"
INT8_DIR="${SD_ENGINE_INT8_DIR:-$REPO/arm64/trt_compile/int8/output}"
DLA_ENGINE="${SD_DLA_ENGINE:-$INT8_DIR/backbone_dla0.engine}"
OUT="${SD_POWER_OUT:-$REPO/arm64/eval/results/power}"
mkdir -p "$OUT"

bench_one() {
  local name=$1 engine=$2 iters=$3
  echo "===== [$name] ====="
  # trtexec in background (enough iterations to keep the SoC busy), tegrastats samples
  "$TRTEXEC" --loadEngine="$engine" --warmUp=2000 --iterations=$iters > "$OUT/${name}_trt.log" 2>&1 &
  local PID=$!
  sleep 3  # let it reach steady state
  timeout 12 tegrastats --interval 1000 > "$OUT/${name}_tegrastats.log" 2>/dev/null || true || true
  wait $PID 2>/dev/null || true
  local lat=$(grep -oP "Latency: min = [0-9.]+ ms, max = [0-9.]+ ms, mean = \K[0-9.]+" "$OUT/${name}_trt.log" | head -1)
  echo "  ${name}: latency=${lat}ms"
  # mean power from tegrastats (VIN_SYS_5V0 total-board, VDD_GPU_SOC SoC rail)
  python3 - "$OUT/${name}_tegrastats.log" "$name" <<'EOF'
import re, sys, statistics
log, name = sys.argv[1], sys.argv[2]
vin, gpu = [], []
for line in open(log):
    m = re.search(r'VIN_SYS_5V0 (\d+)mW/(\d+)mW', line)
    if m: vin.append(int(m[1]))
    m2 = re.search(r'VDD_GPU_SOC (\d+)mW/(\d+)mW', line)
    if m2: gpu.append(int(m2[1]))
if vin:
    if gpu:
        print(f"  {name}: VIN_SYS_5V0 avg={statistics.mean(vin):.0f}mW  VDD_GPU_SOC avg={statistics.mean(gpu):.0f}mW")
    else:
        print(f"  {name}: VIN_SYS_5V0 AGX avg={statistics.mean(vin):.0f}mW")
else:
    print(f"  {name}: (no power data)")
EOF
}

bench_one "C1_gpu_fp16" "$FP16_DIR/backbone_fp16.engine" 400
bench_one "C2_gpu_int8" "$INT8_DIR/backbone_int8.engine" 400
bench_one "C3_dla_int8" "$DLA_ENGINE" 400

echo "===== DONE ====="
