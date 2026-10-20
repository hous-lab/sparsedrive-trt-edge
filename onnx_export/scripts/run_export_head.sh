#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "Exporting SparseDriveV2 detection head to ONNX..."
python "$SCRIPT_DIR/export_head.py" \
    --output "$REPO/onnx_export/output/sparsedrive_head.onnx"

echo "OK: $REPO/onnx_export/output/sparsedrive_head.onnx"
