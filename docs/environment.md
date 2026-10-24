# 1. Environment

## 1.1 x86 host

| Item | Value |
|---|---|
| GPU | RTX 4090 Laptop GPU, 16 GB GDDR6 |
| OS | Ubuntu 22.04, Linux 6.5 kernel |
| Python | 3.8 (conda env `sparsedrive`) |
| torch | 2.0.1+cu118 |
| TensorRT (python) | 10.4.0.26 |
| TRT tarball used for the plugin | /usr/src/TensorRT-10.4.0.26 |
| CUDA (nvcc) | 11.8 (plugin), runtime libs for TRT 10.4 include libcudart.so.12 |
| mmcv / mmdet | 1.7.1 / 3.x-compatible (training repo pins) |
| onnx / gs / onnxruntime | 1.15 / graphsurgeon 0.4.x |

**RTX 4090 Laptop disclaimer.** The RTX 4090 Laptop GPU is not a desktop
RTX 4090: 5,888 cores vs 16,384, 16 GB vs 24 GB, and roughly **57% of the desktop part's memory bandwidth**. Any latency or throughput numbers in this repo were measured on the laptop part. Do not benchmark against desktop-4090 results and do not extrapolate linearly to other laptop GPUs.

## 1.2 Multiple TRT installs (the trap)

The host has four TRT tarballs under /usr/src (8.6.1.6, 10.4.0.26, 10.7.0.23, 10.16.1.11) plus a stale apt TRT 8.6.1 under /usr/include/x86_64-linux-gnu. If CMake picks the stale headers, the plugin compiles against TRT 8 API and fails engine build with "could not find any supported formats". [plugins/build_x86.sh](../plugins/build_x86.sh) auto-detects the tarball matching the python `tensorrt` wheel version; override with `TRT_ROOT=...`.

## 1.3 numpy 2.x pickles on numpy 1.x

Info pickles written by numpy 2.x fail to unpickle on numpy 1.x (`No module named 'numpy._core'`). Every script that loads B2D info pickles registers a `numpy._core` module alias before loading (see [tools/gen_demo_data.py](../tools/gen_demo_data.py)).

## 1.4 Which python

[run_pipeline.sh](../run_pipeline.sh) calls bare `python`; activate the `sparsedrive` conda env first. Running with system python 3.10 + user-site mmcv fails with `No module named 'mmcv._ext'`.
