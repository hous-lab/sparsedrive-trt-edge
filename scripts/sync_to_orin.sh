#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

ORIN_USER="${ORIN_USER:-hous}"
ORIN_HOST="${ORIN_HOST:-orin}"
ORIN_PATH="${ORIN_PATH:-/home/$ORIN_USER/sparsedrive-trt-edge}"

echo "Syncing SparseDriveV2 deploy pipeline to Orin..."
echo "  Host: $ORIN_USER@$ORIN_HOST"
echo "  Path: $ORIN_PATH"

rsync -avz --progress \
    --exclude='.git/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='build/' \
    --exclude='*.engine' \
    "$DEPLOY_DIR/" \
    "$ORIN_USER@$ORIN_HOST:$ORIN_PATH/"

echo "✅ Sync complete"
echo "Next steps on Orin:"
echo "  1. cd $ORIN_PATH"
echo "  2. bash arm64/plugins_build/build_orin.sh"
echo "  3. bash arm64/trt_compile/fp16/scripts/run_build_fp16_orin.sh"
