#!/bin/bash
# Detect TensorRT versions and paths on the current platform
echo "=== TensorRT Version Detection ==="
echo "Platform: $(uname -m)"

# Check common TensorRT installation paths
for dir in /usr/src/TensorRT* /usr/lib/x86_64-linux-gnu/cmake/tensorrt* /opt/TensorRT*; do
    if [ -d "$dir" ]; then
        echo "Found: $dir"
        if [ -f "$dir/VERSION" ]; then
            echo "  Version: $(cat $dir/VERSION)"
        fi
    fi
done

# Check dpkg / pip
if command -v dpkg &> /dev/null; then
    dpkg -l 2>/dev/null | grep tensorrt || true
fi
pip show tensorrt 2>/dev/null | grep -E "^(Name|Version)" || echo "pip: tensorrt not found"

# Check nvinfer library
for libpath in /usr/lib/aarch64-linux-gnu/libnvinfer.so* /usr/lib/x86_64-linux-gnu/libnvinfer.so*; do
    if [ -f "$libpath" ]; then
        echo "Library: $libpath"
    fi
done

if command -v trtexec &> /dev/null; then
    echo "trtexec: $(which trtexec)"
    trtexec --version 2>&1 | head -1 || true
else
    echo "trtexec: not found in PATH"
fi

echo "=== Detection complete ==="
