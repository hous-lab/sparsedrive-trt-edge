# 4. Model variants and real-time capability

## 4.1 Deployed sub-models

| Sub-model | Params (ONNX initializers) | ONNX size | FP16 engine | Inputs |
|---|---|---|---|---|
| backbone | 26.8 M | 107.3 MB | 53.1 MB | img [6,3,384,704] |
| head | 20.0 M | 81.3 MB | 55.6 MB | feat_0..3 + proj + wh |
| map | 26.9 M | 108.3 MB | 56.1 MB | feat_0..3 + proj + wh |
| motion_plan | 16.7 M | 68.3 MB | 39.3 MB | 22 inputs (§2.3) |

## 4.2 Structure notes

- The deployed variant is SparseDriveV2-Small (B2D stage2 checkpoint).
- backbone = ResNet-34 + FPN (4 levels, 256 ch); per tick 6 cameras at
  384x704; features enter head/map engines as 6-camera tensors, and enter
  the motion engine pre-flattened (feat_flat [1,134640,256]).
- head/map engines take features as INPUT (backbone runs once, shared);
  the deployed cascade runs backbone once per tick and feeds both heads.
- head: 900 det anchors, 9 classes, 11-dim box reg
  (x,y,z,w,l,h,sin,cos,vx,vy,vz), 6 deformable layers.
- map: 100 map anchors, 6 classes, 20-point polylines.
- motion_plan: det bridge + map bridge -> motion head (900 inst, 6 modes,
  6-step @0.5 s) + planning head (64x16 lat/lon anchor grid, 15-pt path,
  6-step velocity profile), temporal queue 21 slots x 901 instances
  (900 det + ego query).

## 4.3 Real-time capable variant

The deployed Small variant reaches p50 16.15 ms on RTX 4090 Laptop
(§3.2). On Orin the same FP16 cascade targets the 100 ms (10 Hz) budget;
measure with `arm64/eval/bench_engines.py` under each power mode
(docs/power_modes.md).

## 4.4 Weight links (policy per ckpts/README.md)

- SparseDriveV2-Small stage2 checkpoint: linked from
  [wenchaosun/SparseDriveV2 on Hugging Face](https://huggingface.co/wenchaosun/SparseDriveV2);
  not redistributed here.
- If we later publish a QAT variant, upload via `huggingface-cli upload`
  to the maintainer's HF account (see ckpts/README.md).
