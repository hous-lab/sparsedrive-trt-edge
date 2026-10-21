---
license: apache-2.0
tags:
- tensorrt
- autonomous-driving
- onnx
- edge-deployment
- sparsedrive
---

# SparseDriveV2 TensorRT deployment ONNX

The four deployment ONNX exported by the [sparsedrive-trt-edge](https://github.com/hous-lab/sparsedrive-trt-edge) deployment pipeline (this HF repo is the companion of the code repo, same name). Exported 2026-10-20 from the official SparseDriveV2-Small (B2D stage2) checkpoint (sha256 f15c889b759734894e6ac906403df55a9973d7e60e3101da3ac71e9091f7b130).

## Files

| File | Size | What it is |
|---|---|---|
| sparsedrive_backbone.onnx | 107.3 MB | shared CNN backbone (ResNet-34 + FPN), raw export - no If-folding applies, this is the deployment ONNX |
| sparsedrive_head_sim.onnx | 81.3 MB | 3D detection head, If-folded deployment export (the `_sim` suffix) |
| sparsedrive_map_sim.onnx | 108.3 MB | online mapping head, If-folded deployment export |
| sparsedrive_motion_plan_sim.onnx | 68.3 MB | motion prediction + planning head, If-folded deployment export |

The `_sim` suffix marks the If-folded deployment export: the training graph contains If subgraphs that TensorRT 10.3 cannot parse unfolded (see the code repo docs/lessons.md 7.16), so the pipeline folds them before building engines. Backbone has no conditional subgraphs and needs no folding.

## Usage
The pipeline normally exports these itself (run_pipeline.sh step 2); downloading from here skips the export step - place the ONNX under onnx_export/output/ and continue from the engine-build step.

Build TRT engines with trtexec plus the repo's DeformableAggregation plugin:

```bash
trtexec --onnx=sparsedrive_motion_plan_sim.onnx --fp16 --staticPlugins=<path to libplugins built per the code repo>
```

TensorRT version pairing matters: x86 uses TRT 10.4; Orin/JetPack must use the JetPack-matched TRT 10.3 - 10.4 on Orin is measurably slower on the plugin-bearing graphs (docs/trt_versions.md in the code repo).

Full I/O contracts, accuracy gates and latency numbers: the code repo (docs/pipeline_guide.md, docs/engine_profiling.md).

## Provenance and attribution

Exported from the official SparseDriveV2-Small (B2D stage2) checkpoint. Weights are NOT rehosted here - download from the upstream HuggingFace: https://huggingface.co/wenchaosun/SparseDriveV2 (upstream code MIT). This repo hosts only our own pipeline export artifacts, released under Apache-2.0 (pipeline: Apache-2.0).

## Citation

If these artifacts help your work, please cite/acknowledge the upstream SparseDriveV2 work: https://github.com/swc-17/SparseDriveV2
