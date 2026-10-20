#!/bin/bash
# Orin-side driver: build plugin -> build FP16 (+INT8) engines -> accuracy vs
# the x86 reference -> latency + power benches. Run on the Orin device after
# scripts/sync_to_orin.sh:
#
#   bash scripts/run_orin_pipeline.sh
#
# Optional env:
#   SD_BUILD_INT8=1   also build INT8 engines (calib data synced from x86)
#   SD_SKIP_ACC=1     skip accuracy gates (no x86_reference.npz)
#   SD_SKIP_BENCH=1   skip latency/power benches
set -e
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

echo "===== [1/5] Orin plugin build (sm_87) ====="
bash arm64/plugins_build/build_orin.sh

echo "===== [2/5] FP16 engines ====="
bash arm64/trt_compile/fp16/scripts/run_build_fp16_orin.sh

echo "===== [3/5] INT8 engines (optional) ====="
if [ "${SD_BUILD_INT8:-0}" = "1" ]; then
  bash arm64/trt_compile/int8/scripts/run_build_int8_orin.sh
else
  echo "skip (set SD_BUILD_INT8=1; calib npz synced from x86)"
fi

echo "===== [4/5] Accuracy gates vs x86 reference ====="
if [ "${SD_SKIP_ACC:-0}" = "1" ]; then
  echo "skip (SD_SKIP_ACC=1; needs data/x86_reference.npz)"
else
  python3 arm64/eval/eval_accuracy_orin.py
fi

echo "===== [5/5] Latency + power benches ====="
if [ "${SD_SKIP_BENCH:-0}" = "1" ]; then
  echo "skip (SD_SKIP_BENCH=1)"
else
  SD_RESULTS_DIR=arm64/eval/results python3 arm64/eval/bench_engines.py
  bash arm64/eval/run_power_modes.sh
fi

echo "===== Orin pipeline done ====="
