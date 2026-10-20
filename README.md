# sparsedrive-trt-edge

TensorRT end-side deployment of SparseDriveV2 (detection + online mapping +
motion/planning) for automotive SoCs and laptop-class GPUs. Companion to
[vad-trt-edge](https://github.com/hous-lab/vad-trt-edge); shares the same
pipeline layout: plugin -> ONNX (4 sub-models) -> FP16 engines -> demo data ->
reference -> accuracy gates -> renders.

## Hardware note (important)

All reported x86 numbers were measured on an **RTX 4090 Laptop GPU** (16 GB).
The 4090 Laptop is a different silicon than the desktop RTX 4090: it has
5,888 CUDA cores, 16 GB GDDR6, and about **57% of the desktop 4090's memory
bandwidth**. Do not compare these numbers against desktop-4090 results. Orin
results are measured on Jetson AGX Orin 64GB.

## Results summary

| Metric | Value |
|---|---|
| x86 FP16 cascade latency (backbone+head+map+motion, RTX 4090 Laptop) | **p50 16.15 ms / p90 16.52 ms** (~62 FPS) |
| x86 accuracy gates (PT vs engine cascade, single frame) | see `x86/eval/results/motion_v2_gate.json` |
| Engine sizes (FP16) | backbone 53.1 MB, head 55.6 MB, map 56.1 MB, motion_plan 39.3 MB |
| ONNX param count (incl. baked constants) | backbone 26.8 M, head 20.0 M, map 26.9 M, motion_plan 16.7 M |
| Motion engine I/O | 22 inputs / 5 outputs (queue-tensor convention, see docs/pipeline_guide.md) |

Accuracy gates (cosine similarity, engine cascade vs PT reference):
motion_cls 0.9965, motion_traj 0.9895, traj_cls 0.9856, lat_anchor 0.9991,
lon_anchor 0.9742 (WARN, see docs/lessons.md).

## Repo layout

```
plugins/     DeformableAggregation TRT plugin (x86 + Orin)
onnx_export/ export scripts + patches + output ONNX
x86/         FP16 engine builds + eval (gates, renders)
arm64/       Orin: plugin build, TRT compile, accuracy/latency/power eval
tools/       demo data + x86 reference generation
configs/     vendored training config + kmeans anchors + ego query
scripts/     cross-cutting drivers (run_orin_pipeline.sh, sync_to_orin.sh)
data/        demo inputs + x86 reference (generated; see data/README.md)
ckpts/       no weights stored; see ckpts/README.md for the weight policy
docs/        environment, pipeline guide, profiling, lessons
```

## Quick start (x86)

```bash
conda activate sparsedrive      # torch 2.0.1+cu118, TRT 10.4.0.26, mmcv 1.7.1
bash run_pipeline.sh             # 8 steps, idempotent; see docs/pipeline_guide.md
```

For Orin: `scripts/sync_to_orin.sh`, then on the device
`bash scripts/run_orin_pipeline.sh` (5 steps; `SD_BUILD_INT8=1` adds INT8).

## Weights policy

Per `ckpts/README.md`: upstream checkpoints are linked (not redistributed);
models obtained indirectly (QAT or similar) can be published to Hugging Face
via CLI upload.

## License

Apache-2.0 with third-party attribution in [NOTICE](NOTICE).
