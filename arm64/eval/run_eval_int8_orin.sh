#!/bin/bash
# Orin INT8 accuracy: run the multi-config accuracy gates (FP16 baseline,
# BB INT8, BB+Head INT8) against the x86 reference on the Orin.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

export LD_LIBRARY_PATH="$DEPLOY_DIR/arm64/plugins_build/build:$LD_LIBRARY_PATH"

python "$SCRIPT_DIR/eval_accuracy_orin.py"
echo "[int8-eval] results: $DEPLOY_DIR/arm64/eval/results/accuracy_comparison.json"
