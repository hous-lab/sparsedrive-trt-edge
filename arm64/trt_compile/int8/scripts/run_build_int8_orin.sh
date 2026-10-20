#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

export LD_LIBRARY_PATH="$DEPLOY_DIR/arm64/plugins_build/build:$LD_LIBRARY_PATH"
export PYTHONPATH="$DEPLOY_DIR/..:$PYTHONPATH"

# Calibration data: link the x86-generated npz into the arm64 calib dir
# (generated once on x86 via tools/gen_calib_*.py, synced with the repo).
ln -sf "$DEPLOY_DIR/x86/trt_compile/int8/calib_data/backbone_calib.npz" \
       "$DEPLOY_DIR/arm64/trt_compile/int8/calib_data/" 2>/dev/null || true
ln -sf "$DEPLOY_DIR/x86/trt_compile/int8/calib_data/head_calib.npz" \
       "$DEPLOY_DIR/arm64/trt_compile/int8/calib_data/" 2>/dev/null || true

# Backbone INT8: reuse the x86 builder (TRT-API-generic; runs identically on
# Orin with the same FP32 fallback list). Head INT8 uses the Orin variant
# (TRT-10-aware) below.
python "$DEPLOY_DIR/x86/trt_compile/int8/scripts/build_backbone_int8.py" \
    --engine "$DEPLOY_DIR/arm64/trt_compile/int8/output/backbone_int8.engine"

python "$SCRIPT_DIR/build_head_int8_orin.py" \
    --onnx "$DEPLOY_DIR/onnx_export/output/sparsedrive_head_sim.onnx" \
    --engine "$DEPLOY_DIR/arm64/trt_compile/int8/output/head_int8.engine" \
    --plugin "$DEPLOY_DIR/arm64/plugins_build/build/libplugins_orin.so"

echo "✅ Orin INT8 engines built"
