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

## 4.5 INT8 tier (backbone + head)

The default deployment tier is FP16 for all four engines. An optional INT8
tier quantizes the two cheapest-to-quantize sub-models:

| Sub-model | Precision | Why |
|---|---|---|
| backbone | INT8 | 26.8 M conv weights; pure CNN, convs quantize cleanly |
| head | INT8 | MatMul layers quantized; deformable plugin + sensitive ops forced FP16 |
| map | FP16 | shares the same feature inputs; kept FP16 to cap accuracy risk |
| motion_plan | FP16 | argmax/selection-heavy; lon_anchor is already at the WARN edge in FP16 (§7.6) |

Calibration uses entropy calibrator v2 over 64 real val frames stored as
stacked float16 npz (`x86/trt_compile/int8/calib_data/*.npz`; the
calibrators cast back to float32 per batch). `tools/gen_calib_backbone.py`
saves raw `[64,6,3,384,704]` images; `tools/gen_calib_head.py` saves
`feat_0..3 [64,6,256,h,w]` + projection_mat/image_wh. The motion engine
has no INT8 calib data in this tier.

**x86 results (RTX 4090 Laptop, TRT 10.4):**

| Engine | FP16 size | INT8 size | FP16 ms | INT8 ms |
|---|---|---|---|---|
| backbone | 55.5 MB | 30.9 MB | 5.6 | **2.7** (2.1x) |
| head | 49.5 MB | ~51 MB | 4.4 | 4.4 (no gain) |

Head INT8 gives no latency gain: the deformable aggregation plugin (the
head's dominant cost) is forced FP16, so quantizing MatMuls around it does
not move the critical path. INT8 still shrinks the engine and the accuracy
cost is confined to head outputs (det_quality 0.987).

Cascade gates (INT8 cascade vs PT reference, all PASS):
bb_feat 0.981-0.989, hd_det_cls 0.9998, hd_det_reg 0.9998,
hd_det_quality 0.9871, map_cls 0.9999 (map is FP16), mp_motion_cls 0.979,
mp_motion_traj 0.959, mp_traj_cls 0.998, mp_lon_anchor 0.959,
mp_lat_anchor 0.998.

**Orin results (AGX Orin 64GB, MODE_30W, TRT 10.3):**

| Engine | FP16 size | INT8 size | FP16 ms | INT8 ms |
|---|---|---|---|---|
| backbone | 63.9 MB | 28.1 MB | 66.5 | **39.4** (1.69x) |
| head | 44.7 MB | 45.6 MB | 52.1 | 52.8 (no gain) |

Same conclusion as x86, now measured at the capped 30W mode: INT8
backbone is the only lever (1.69x), head INT8 quantizes MatMuls around
the FP16 deformable plugin and does not move the critical path. INT8
cascade (BB INT8 + rest FP16) 271.1 -> 244.0 ms (~10%).

Cascade gates vs the x86 FP16 reference, three configs (all PASS; gates
bb_feat 0.98 / hd_det_quality 0.98 / motion 0.95 - see docs/lessons.md 7.17):
| Config | bb_feat | hd_det_quality | mp_motion_traj | mp_lon_anchor |
|---|---|---|---|---|
| All FP16 | 0.9999+ | 0.999896 | 0.997095 | 0.974231 (x86: 0.9742) |
| BB INT8 | 0.981-0.989 | 0.986335 | 0.961489 | 0.963534 |
| BB+Head INT8 | 0.981-0.989 | 0.986270 | 0.961453 | 0.963534 |

Orin INT8 degradation is bit-consistent with x86 INT8 (bb_feat 0.981-0.989
on both platforms, mp_lon_anchor 0.9635 Orin vs 0.9589 x86) - the INT8
tier is cross-platform reproducible. Full per-tensor tables:
`arm64/eval/results/accuracy_comparison.json`; raw bench rows:
`arm64/eval/results/int8/bench_results.json`.
