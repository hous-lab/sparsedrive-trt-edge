#!/bin/bash
# Build the DeformableAggregation plugin for x86 (default sm_89, e.g. RTX 4090 Laptop).
set -e
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO/plugins"
TRT_ROOT="${TRT_ROOT:-}"
if [ -z "$TRT_ROOT" ]; then
  # Prefer the tarball matching the python tensorrt wheel version.
  V="$(python -c 'import tensorrt; print(tensorrt.__version__)' 2>/dev/null || true)"
  for d in ${V:+/usr/src/TensorRT-"$V"*/} /usr/src/TensorRT-* /usr/local/tensorrt; do
    [ -d "$d/include" ] && TRT_ROOT="$d" && break
  done
fi
if [ -z "$TRT_ROOT" ] || [ ! -d "$TRT_ROOT/include" ] 2>/dev/null; then
  echo "ERROR: TensorRT root with headers not found. Set TRT_ROOT=/path/to/TensorRT" >&2
  exit 1
fi
CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda}"
ARCH="${ARCH:-89}"

echo "Building plugin (x86_64, sm_${ARCH}) -> plugins/build/libplugins.so"
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DTENSORRT_ROOT="${TRT_ROOT}" \
      -DCUDA_ROOT="${CUDA_ROOT}" \
      -DARCH="${ARCH}" ..
make -j$(nproc)
echo "OK: plugins/build/libplugins.so"
