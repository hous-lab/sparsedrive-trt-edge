#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# TRT root: tarball layout first, else Jetson dpkg multiarch via symlink stub
TRT_ROOT=""
for c in /usr/src/TensorRT-* /usr/local/TensorRT-*; do
  if [ -f "$c/include/NvInfer.h" ] && grep -q "define NV_TENSORRT_MAJOR 10" "$c/include/NvInfer.h"; then
    TRT_ROOT="$c"
  fi
done
if [ ! -f "$TRT_ROOT/include/NvInfer.h" ]; then
  if [ -f /usr/include/aarch64-linux-gnu/NvInfer.h ]; then
    STUB="$SCRIPT_DIR/build/trt-root"
    mkdir -p "$STUB"
    ln -sfn /usr/include/aarch64-linux-gnu "$STUB/include"
    ln -sfn /usr/lib/aarch64-linux-gnu "$STUB/lib"
    TRT_ROOT="$STUB"
  else
    echo "ERROR: no TensorRT dev headers found (tarball + dpkg both missing)" >&2
    exit 1
  fi
fi
CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda}"
ARCH="${ARCH:-87}"
echo "Building DeformableAggregation Plugin for Orin (ARM64, sm_${ARCH})"
echo "TRT_ROOT=${TRT_ROOT}"
echo "CUDA_ROOT=${CUDA_ROOT}"

mkdir -p "$SCRIPT_DIR/build" && cd "$SCRIPT_DIR/build"
# fresh configure: stale CMakeCache from a failed run poisons CMAKE_CUDA_ARCHITECTURES
rm -f CMakeCache.txt
rm -rf CMakeFiles

cmake -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER="$CUDA_ROOT/bin/nvcc" \
    -DTENSORRT_ROOT="$TRT_ROOT" \
    -DCUDA_ROOT="$CUDA_ROOT" \
    -DARCH="$ARCH" \
    "$DEPLOY_DIR/plugins/"

make -j4

cp libplugins.so libplugins_orin.so
echo "✅ Orin plugin built: $(pwd)/libplugins_orin.so"
ls -lh libplugins_orin.so
