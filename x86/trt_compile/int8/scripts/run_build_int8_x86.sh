#!/bin/bash
# x86 INT8 tier end-to-end: calib gen -> engine builds -> INT8 cascade
# reference -> gates vs the FP16 cascade reference -> latency bench.
# Env: one conda env serves all steps (torch + TRT + mmcv), same as
# run_pipeline.sh. Expects data/x86_reference.npz (FP16 walk-through).
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO"

# [1/5] Calibration data (real val frames through the training repo).
if [ ! -f x86/trt_compile/int8/calib_data/backbone_calib.npz ] || \
   [ ! -f x86/trt_compile/int8/calib_data/head_calib.npz ]; then
  python tools/gen_calib_backbone.py
  python tools/gen_calib_head.py
fi

# [2/5] INT8 engines.
python x86/trt_compile/int8/scripts/build_backbone_int8.py
python x86/trt_compile/int8/scripts/build_head_int8.py

# [3/5] Cascade dir: INT8 bb/head under the fp16 names the eval scripts
# expect, map/motion stay FP16 (this repo's INT8 tier).
mkdir -p x86/trt_compile/int8/output/cascade
cd x86/trt_compile/int8/output/cascade
ln -sf ../backbone_int8.engine backbone_fp16.engine
ln -sf ../head_int8.engine head_fp16.engine
ln -sf ../../../fp16/output/map_fp16.engine map_fp16.engine
ln -sf ../../../fp16/output/motion_plan_fp16.engine motion_plan_fp16.engine
cd "$REPO"

# [4/5] INT8 cascade reference + gates vs the FP16 reference.
SD_X86_ENGINE_DIR="$REPO/x86/trt_compile/int8/output/cascade" \
SD_REF_NPZ="$REPO/x86/trt_compile/int8/output/x86_reference_int8.npz" \
  python tools/gen_x86_reference.py
python tools/compare_npz_gates.py \
    --ref data/x86_reference.npz \
    --cand x86/trt_compile/int8/output/x86_reference_int8.npz \
    --json x86/eval/results/int8_vs_fp16_gates.json

# [5/5] Latency (uses the shared bench script; INT8 rows skip if absent).
SD_PLUGIN_LIB="$REPO/plugins/build/libplugins.so" \
SD_ENGINE_DIR="$REPO/x86/trt_compile/fp16/output" \
SD_ENGINE_INT8_DIR="$REPO/x86/trt_compile/int8/output" \
SD_RESULTS_DIR="$REPO/x86/eval/results" \
  python arm64/eval/bench_engines.py
