#!/bin/bash
# SparseDriveV2 TRT edge deployment — x86 one-shot driver.
#
# Runs the full chain: plugin -> ONNX export -> FP16 engines -> demo data ->
# x86 reference -> accuracy gates -> renders. Run inside the conda env with
# torch+TRT+mmcv (sparsedrive env):
#
#   conda activate sparsedrive
#   bash run_pipeline.sh
#
# Optional env:
#   SD_BUILD_INT8=1     also build INT8 engines (calib data required)
#   SD_SKIP_EXPORT=1    skip ONNX export if output/*.onnx already exist
set -e
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

echo "===== [1/8] kmeans anchor symlink ====="
mkdir -p data
ln -sfn ../configs/anchors data/kmeans
ls -l data/kmeans/ | head -3

echo "===== [2/8] plugin build (x86, sm_89) ====="
if [ ! -f plugins/build/libplugins.so ]; then
  bash plugins/build_x86.sh
else
  echo "skip (plugins/build/libplugins.so exists)"
fi

echo "===== [3/8] ONNX export (training-repo env) ====="
mkdir -p onnx_export/output
export_sd() {
  local name=$1 script=$2
  if [ -f "onnx_export/output/sparsedrive_${name}.onnx" ]; then
    echo "skip ${name} (onnx exists)"
    return
  fi
  python onnx_export/scripts/${script} \
    --output "$REPO/onnx_export/output/sparsedrive_${name}.onnx"
}
export_sd backbone export_backbone.py
export_sd head export_head.py
export_sd map export_map_head.py
export_sd motion_plan export_motion_plan.py

# fold the shape-guarded If subgraphs so the same ONNX builds on TRT 10.3
# (Jetson) as well as 10.4 (x86); see tools/fold_if_subgraphs.py + docs/lessons
if [ ! -f onnx_export/output/sparsedrive_motion_plan_sim.onnx ]; then
  python tools/fold_if_subgraphs.py
else
  echo "skip fold (sim onnx exists)"
fi

echo "===== [4/8] FP16 engine build ====="
mkdir -p x86/trt_compile/fp16/output
P=plugins/build/libplugins.so
O=x86/trt_compile/fp16/output
[ -f "$O/backbone_fp16.engine" ] || python x86/trt_compile/fp16/scripts/build_backbone_fp16.py
[ -f "$O/head_fp16.engine" ] || python x86/trt_compile/fp16/scripts/build_head_fp16.py
[ -f "$O/map_fp16.engine" ] || python x86/trt_compile/fp16/scripts/build_onnx_engine.py \
  --onnx onnx_export/output/sparsedrive_map_sim.onnx \
  --engine $O/map_fp16.engine \
  --plugin $P
[ -f "$O/motion_plan_fp16.engine" ] || python x86/trt_compile/fp16/scripts/build_onnx_engine.py \
  --onnx onnx_export/output/sparsedrive_motion_plan_sim.onnx \
  --engine $O/motion_plan_fp16.engine \
  --plugin $P
echo "===== [5/8] demo data + x86 reference ====="
if [ ! -f data/demo_data/sample0.npz ]; then
  python tools/gen_demo_data.py
else
  echo "skip demo data (exists)"
fi
python tools/gen_x86_reference.py

echo "===== [6/8] INT8 tier (optional) ====="
if [ "${SD_BUILD_INT8:-0}" = "1" ]; then
  bash x86/trt_compile/int8/scripts/run_build_int8_x86.sh
else
  echo "skip (set SD_BUILD_INT8=1 for the full INT8 chain)"
fi

echo "===== [7/8] accuracy gates (PT vs engine cascade) ====="
python x86/eval/eval_motion_v2_x86.py

echo "===== [8/8] renders ====="
python x86/eval/render_outputs.py
