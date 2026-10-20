#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$DEPLOY_DIR"

export LD_LIBRARY_PATH="$DEPLOY_DIR/arm64/plugins_build/build:$LD_LIBRARY_PATH"
export PYTHONPATH="$DEPLOY_DIR/..:$PYTHONPATH"

python "$SCRIPT_DIR/build_backbone_fp16_orin.py" \
    --onnx "$DEPLOY_DIR/onnx_export/output/sparsedrive_backbone.onnx" \
    --engine "$DEPLOY_DIR/arm64/trt_compile/fp16/output/backbone_fp16.engine"

python "$SCRIPT_DIR/build_head_fp16_orin.py" \
    --onnx "$DEPLOY_DIR/onnx_export/output/sparsedrive_head_sim.onnx" \
    --engine "$DEPLOY_DIR/arm64/trt_compile/fp16/output/head_fp16.engine" \
    --plugin "$DEPLOY_DIR/arm64/plugins_build/build/libplugins_orin.so"


python "$SCRIPT_DIR/build_map_motion_fp16_orin.py" \
    --plugin "$DEPLOY_DIR/arm64/plugins_build/build/libplugins_orin.so"
echo "OK: Orin FP16 engines (backbone/head/map/motion_plan)"
