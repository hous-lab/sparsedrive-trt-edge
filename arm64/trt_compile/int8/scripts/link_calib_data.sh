#!/bin/bash
# Symlink calibration data from x86 (assumes rsync was done)
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CALIB_DIR="$(cd "$SCRIPT_DIR/../calib_data" && pwd)"

cd "$CALIB_DIR"
ln -sf ../../../../x86/trt_compile/int8/calib_data/backbone_calib.npz .
ln -sf ../../../../x86/trt_compile/int8/calib_data/head_calib.npz .
echo "✅ Calibration data linked"
ls -la .
