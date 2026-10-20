# 2. Pipeline guide

## 2.1 x86 driver: `run_pipeline.sh` (8 steps)

| Step | What it does | Skip condition |
|---|---|---|
| 1 | symlink `data/kmeans -> ../configs/anchors` | always re-links |
| 2 | plugin build (`plugins/build_x86.sh`) | `plugins/build/libplugins.so` exists |
| 3 | ONNX export (backbone/head/map/motion_plan) | per-model ONNX exists |
| 4 | FP16 engine build | per-engine file exists |
| 5 | `tools/gen_demo_data.py` + `tools/gen_x86_reference.py` | demo npz exists; ref always regenerated |
| 6 | INT8 tier (optional, `SD_BUILD_INT8=1` -> `x86/trt_compile/int8/scripts/run_build_int8_x86.sh`: calib -> engines -> INT8 cascade gates + bench) | off by default |
| 7 | `x86/eval/eval_motion_v2_x86.py` gates | always runs |
| 8 | `x86/eval/render_outputs.py` renders | always runs |

## 2.2 Sub-model I/O

**backbone** engine: `img [6,3,384,704] f32 -> feat_0..feat_3 [6,256,h,w]`.

**head** engine (6 in / 4 out):
`feat_0..3, projection_mat [6,4,4], image_wh [6,2] -> det_cls [1,900,9],
det_reg [1,900,11], det_quality [1,900,2], det_instance_feat [1,900,256]`.

**map** engine: same 6 inputs as head -> map_cls [1,100,6],
map_instance_feature [1,100,256], map_anchor_embed [1,100,256] ->
map_reg zeros bridge [1,100,2] (see docs/lessons.md).

## 2.3 motion engine (22 in / 5 out)

Inputs: feat_flat [1,134640,256], spat_shape [4,2] i32, lvl_start [4] i32,
det_cls/reg/quality/instance_feat/anchor_embed (det bridge),
map_cls/reg/instance_feature/anchor_embed (map bridge, map_reg zeros),
projection_mat [6,4,4], image_wh [6,2], timestamp [1], ego_status [10],
tp_near [2], gt_ego_fut_cmd [6] i64, mask [900], queue_temp_feat
[21,1,901,256], queue_temp_anchor [21,1,901,256, period [1,901] i64.
Outputs: motion_cls [1,900,6], motion_traj [1,900,6,6,2], traj_cls [1,1024],
lat_anchor [1,64,15,2], lon_anchor [1,16,6].

## 2.4 Idempotency and env vars

Every stage checks its outputs and skips when they exist. Force a full rebuild
by deleting the artifacts (ONNX, engines, demo npz) or pointing env vars
elsewhere. Env vars: SD_CONFIG, SD_CKPT, SD_PLUGIN_LIB, SD_ENGINE_DIR,
SD_REF_NPZ, SD_DEMO_NPZ, SD_BUILD_INT8, SD_SKIP_ACC, SD_SKIP_BENCH,
SPARSEDRIVE_PROJECT_ROOT.
