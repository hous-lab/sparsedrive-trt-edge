#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

export LD_LIBRARY_PATH="$DEPLOY_DIR/arm64/plugins_build/build:$LD_LIBRARY_PATH"
export PYTHONPATH="$DEPLOY_DIR/..:$PYTHONPATH"

python "$SCRIPT_DIR/eval_orin.py" \
    --bb-engine "$DEPLOY_DIR/arm64/trt_compile/fp16/output/backbone_fp16.engine" \
    --head-engine "$DEPLOY_DIR/arm64/trt_compile/fp16/output/head_fp16.engine" \
    --plugin "$DEPLOY_DIR/arm64/plugins_build/build/libplugins_orin.so" \
    --num-frames 50
echo "✅ Orin FP16 eval complete"
