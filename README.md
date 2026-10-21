# SparseDriveV2 TensorRT Edge Deployment

End-to-end TensorRT deployment of [SparseDriveV2](https://github.com/swc-17/SparseDriveV2/tree/bench2drive) (3D detection + mapping + motion prediction & planning) on NVIDIA Jetson AGX Orin (`sm_87`) and an x86_64 dev host (RTX 4090 Laptop, `sm_89`).

## Latency on Jetson Orin and 4090 Laptop

| Platform | Config | BB | Head | Map | Motion | Cascade |
|---|---|---|---|---|---|---|
| x86 (4090 Laptop) | FP16 | 5.6 | 4.4 | 6.1 | 5.0 | 21.1 ms |
| x86 (4090 Laptop) | INT8 | 2.7 | 4.4 | 6.1 | 5.0 | 18.2 ms |
| Orin MAXN | FP16 | 22.1 | 17.3 | 26.1 | 20.7 | **86.2 ms** |
| Orin MAXN | INT8 | **11.3** | 17.4 | 26.1 | 20.7 | 75.5 ms |

> Per-engine and cascade latency (200-iter mean per engine, `bench_engines.py`)
> The x86 p50/p90 (16.15/16.52 ms) are a separate 50-iter p50 measurement on the FP16 chain.

- Backbone INT8 is the only lever on both platforms: 2.1x on x86 (cascade 21.1 -> 18.2 ms) and 1.96x on Orin MAXN (cascade 86.2 -> 75.4 ms, ~12%). 
- Head INT8 buys nothing - the deformable aggregation plugin (the head's dominant cost) is forced FP16, so quantizing the MatMuls around it never moves the critical path.

### Accuracy

Cosine similarity, engine cascade vs PyTorch reference, single real demo frame (gates per docs/lessons.md §7.17):

| Output | x86 FP16 | x86 INT8 | Orin FP16 | Orin INT8 |
|---|---|---|---|---|
| bb_feat_0..3 | 0.9999+ | 0.981-0.989 | 0.9999+ | 0.981-0.989 |
| hd_det_cls / hd_det_reg | 0.9999+ | 0.9998 | 0.9999+ | 0.9998 |
| hd_det_quality | 0.9999 | 0.9871 | 0.9999 | 0.9863 |
| map_cls | 0.9999 | 0.9999 (map stays FP16) | 0.9999+ | 0.9999+ (FP16) |
| mp_motion_cls | 0.9965 | 0.979 | 0.9987 | 0.9785 |
| mp_motion_traj | 0.9895 | 0.959 | 0.9971 | 0.9615 |
| mp_traj_cls | 0.9856 | 0.998 | 0.9998 | 0.9981 |
| mp_lat_anchor | 0.9991 | 0.9982 | 0.9988 | 0.9979 |
| mp_lon_anchor | 0.9742 (WARN) | 0.9589 | 0.9742 | 0.9635 |

The Orin INT8 degradation is bit-consistent with x86 INT8 (bb_feat 0.981-0.989 on both platforms; mp_lon_anchor 0.9635 Orin vs 0.9589 x86) - the INT8 tier is cross-platform reproducible. Full per-tensor tables: `arm64/eval/results/accuracy_comparison.json` (Orin, three configs) and `x86/eval/results/int8_vs_fp16_gates.json` (x86). The lon_anchor WARN reflects argmax-sensitive velocity-profile selection near ties under reduced precision (docs/lessons.md §7.6), not a deployment defect.

## Get started

### 1. Environment

The conda environment is built exactly as the upstream training repo instructs - follow the official [SparseDriveV2 quick start](https://github.com/swc-17/SparseDriveV2/blob/bench2drive/docs/quick_start.md), then `conda activate sparsedrive` (torch 2.0.1+cu118, mmcv 1.7.1, TensorRT python 10.4.0.26, onnx 1.15, numpy<2). x86-side specifics (multiple TRT installs, numpy 2.x pickle shims, which python) are in [docs/environment.md](docs/environment.md).

### 2. Weights, ONNX, and data inputs

- Download the SparseDriveV2-Small stage2 checkpoint per the link table in [ckpts/README.md](ckpts/README.md) (upstream weights are linked, never redistributed). 
- To skip the ONNX-export step, the four deployment ONNX are published at [hous-lab/sparsedrive-trt-edge on Hugging Face](https://huggingface.co/hous-lab/sparsedrive-trt-edge)
    - place them under `onnx_export/output/` and continue from the engine-build step (the HF repo hosts only our own export artifacts; weights stay upstream-link-only). 
- The demo inputs (3 val frames used for demo data + calibration + reference) come from the Bench2Drive val split, and `run_pipeline.sh` step 5 generates demo npz from them. 
- Set the env vars in `run_pipeline.sh` (SD_CKPT, dataset roots) if your paths differ.

### 3. x86 pipeline

```bash
conda activate sparsedrive
bash run_pipeline.sh                # 8 idempotent steps: plugin -> ONNX -> FP16 engines -> demo data -> gates -> renders
SD_BUILD_INT8=1 bash run_pipeline.sh  # adds the INT8 tier (calib -> INT8 engines -> gates + bench)
```

See [docs/pipeline_guide.md](docs/pipeline_guide.md) for the step table and [docs/engine_profiling.md](docs/engine_profiling.md) for what each stage produces.

### 4. Orin pipeline

```bash
# on the x86 host:
bash scripts/sync_to_orin.sh
# on the Orin:
bash scripts/run_orin_pipeline.sh   # 5 steps; SD_BUILD_INT8=1 adds the INT8 tier
```

Accuracy on Orin is engine-vs-engine against the x86 reference (the training env is impractical on the device): `arm64/eval/eval_accuracy_orin.py` compares all four engines against `data/x86_reference.npz` with cosine/MSE gates. Latency/power: `arm64/eval/bench_engines.py` (per-engine, any power mode) and `arm64/eval/run_power_modes.sh` (per-mode sweep with tegrastats).

## Hardware platforms

### NVIDIA Jetson AGX Orin 64 GB

| Component | Specification |
|---|---|
| SoC | AGX Orin 64 GB, Ampere GPU (**sm_87**), 2048 CUDA cores |
| Memory | 64 GB LPDDR5, 204.8 GB/s |
| Power modes | MAXN / 50 W / 30 W / 15 W (see [docs/power_modes.md](docs/power_modes.md)) |
| TensorRT | 10.3 (JetPack, dpkg layout; 10.4 is measurably slower on this SoC, see docs/trt_versions.md) |
| OS | JetPack 6.2.2, L4T 36.5.0, Linux aarch64 |

### x86_64 workstation - RTX 4090 Laptop GPU

| Component | Specification |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 **Laptop** GPU (Ada, **sm_89**) |
| VRAM | 16 GB GDDR6 |
| Memory bandwidth | ~576 GB/s (~57% of the desktop 4090's 1008 GB/s) |
| TGP | up to 175 W (incl. Dynamic Boost) |
| CUDA / driver | 11.8 / 580.95.05 |
| TensorRT | 10.4.0.26 (pip wheel + matching tarball for the plugin) |
| OS | Ubuntu 22.04 (Linux 6.5) |

**Not a desktop RTX 4090.** The laptop part has 5,888 CUDA cores vs 16,384 on desktop, 16 GB vs 24 GB VRAM, and ~57% of the desktop part's memory bandwidth. If you reproduce on a desktop 4090 (or any other GPU), expect lower latencies than this README reports - that difference is hardware, not software. Accuracy results are unaffected, since they are properties of the engine builds, not of the GPU speed.

## Acknowledgements

- [SparseDriveV2](https://github.com/swc-17/SparseDriveV2/tree/bench2drive) - the upstream model, training configs and checkpoints; the deployed variant is SparseDriveV2-Small (B2D stage2).
- [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive) - the closed-loop benchmark whose val split supplies demo/calib frames.
- [vad-trt-edge](https://github.com/hous-lab/vad-trt-edge) - the companion VAD deployment repo that established the pipeline layout.
- [NVIDIA TensorRT](https://github.com/NVIDIA/TensorRT) - inference engine and plugin SDK.

## License

Apache-2.0 with third-party attribution in [NOTICE](NOTICE).
