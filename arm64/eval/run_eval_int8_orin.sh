#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

export LD_LIBRARY_PATH="$DEPLOY_DIR/arm64/plugins_build/build:$LD_LIBRARY_PATH"
export PYTHONPATH="$DEPLOY_DIR/..:$PYTHONPATH"

python "$SCRIPT_DIR/eval_orin_fp16_vs_int8.py" \
    --fp16 "$DEPLOY_DIR/arm64/trt_compile/fp16/output/head_fp16.engine" \
    --int8 "$DEPLOY_DIR/arm64/trt_compile/int8/output/head_int8.engine" \
    --plugin "$DEPLOY_DIR/arm64/plugins_build/build/libplugins_orin.so" \
    --num-frames 10
echo "✅ Orin INT8 vs FP16 comparison complete"
