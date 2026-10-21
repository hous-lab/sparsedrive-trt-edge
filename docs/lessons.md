# 7. Lessons (debugging log)

## 7.1 The kINT64 / TRT-headers trap

Symptom: engine build fails with "could not find any supported formats" on the deformable node. Root cause chain:

1. CMake had no valid TENSORRT_ROOT and fell back to stale apt TRT 8.6.1 headers; the plugin compiled against TRT 8 API.
2. TRT 10.4 DOES have `DataType::kINT64`. The first "fix" (a `#if NV_TENSORRT_MAJOR >= 10` guard that dropped the int64 branch) was wrong and *caused* the failure: the plugin must ACCEPT int64 plugin inputs, because spatial_shape/level_start_index arrive from ONNX as int64 constants.
3. The real fix: build the plugin with `TRT_ROOT=/usr/src/TensorRT-10.4.0.26` (build_x86.sh now auto-detects the tarball matching the python wheel) and keep the plugin accepting kINT32|kINT64, casting to int32 inside enqueue.

## 7.2 chdir poisoning

Scripts that `os.chdir(project_root)` at import time poison every later relative-path computation. Rules (all export scripts follow them):

- compute REPO/ROOT absolute constants BEFORE any chdir;
- make --output/--config/--checkpoint defaults absolute at parse time;
- abspath any relative --output argument against REPO, not CWD.

## 7.3 Double-unsqueeze metas bug

The collated batch tensors are already batch-first: `projection_mat [1,6,4,4]`. Adding another unsqueeze(0) produced [1,1,6,4,4] and a confusing matmul error ("size of tensor a (6) vs b (13)"). When calling sub-heads directly, build metas as `dict(collated)` — no extra batch dim.

## 7.4 Engine I/O conventions (squeezed metas)

Engines take SQUEEZED per-sample metas: projection_mat [6,4,4], image_wh [6,2], ego_status [10], tp_near [2], gt_ego_fut_cmd [6] i64, timestamp [1]. The demo npz saves exactly these shapes (§2.2-2.3).

## 7.5 map_reg zeros bridge

The map engine has no regression output, so the deployed chain feeds map_reg = zeros [1,100,2] into the motion engine; gen_demo_data mirrors this on the PT side (zeroed before motion_plan_head) so gates compare like with like.

## 7.6 lon_anchor WARN (0.9742)

cos-sim 0.9742 < 0.99 gate but > gate-0.02 (WARN band). Mechanism: the planning head selects a velocity profile by argmax over 1024 (64x16) anchor scores; near-ties flip under FP16. Perception outputs stay >0.98. Treat as a known FP16 artifact of argmax-style selection, documented in x86/eval/results/motion_v2_gate.json.

## 7.7 TRT-8.6.1 apt headers vs tarballs

If `bash -n` builds succeed but engine builds fail, check which TRT headers the plugin linked. `objdump -d libplugins.so` plus a python `tensorrt.int64 == 8` check settles it in one minute.

## 7.8 numpy 2.x pickles on numpy 1.x

Info pickles from numpy 2.x envs need the `numpy._core` shim (§1.3).

## 7.9 Building on Jetson (TRT 10.3, dpkg layout)

JetPack's TRT differs from the x86 tarball in three ways; build_orin.sh handles all three:

1. **Headers/libs live in multiarch dirs** (/usr/include/aarch64-linux-gnu, /usr/lib/aarch64-linux-gnu), not in a $TRT_ROOT/{include,lib} tree. The script builds a symlink-stub TRT root pointing at both.
2. **An old tarball may shadow the dpkg TRT**: /usr/src/TensorRT-8.6.1.6 ships with JetPack and its headers are TRT 8. Probing for "any tarball" picks it and the plugin builds against the WRONG headers. The script only accepts tarballs whose NvInfer.h defines NV_TENSORRT_MAJOR 10.
3. **nvcc is not on the non-interactive PATH**: CMake needs -DCMAKE_CUDA_COMPILER=$CUDA_ROOT/bin/nvcc, and a stale CMakeCache.txt from a failed run poisons CMAKE_CUDA_ARCHITECTURES — build_orin.sh clears the cache before configuring.

With TRT 10.3's builder, the exported ONNX is also rejected: "IIfConditionalOutputLayer inputs must have the same shape ([1,900,13,3] vs [1,900,13,3,1])". These are shape-guarded squeezes (x.squeeze(-1) if x.shape[-1] == 1 else x) that torch.onnx.export emits for no-arg .squeeze() calls; TRT 10.4 folds them at build time, 10.3 does not. tools/fold_if_subgraphs.py inlines the Squeeze branch statically and writes <name>_sim.onnx; both x86 and Orin now build engines from the _sim files, so one set of ONNX serves both.

Folding has its own traps (each cost a debugging round-trip):
1. **Topological order**: the branch nodes must be spliced at the If node's own position, not appended at the graph end — consumers (TopK/Slice) would otherwise precede their new input ("Nodes in a graph must be topologically sorted" / TRT "Trying to access weights or a null tensor").
2. **Protobuf insert copies**: `g.node.insert(idx, nb)` copies the message, so rename branch outputs to the If's output names BEFORE splicing. Renaming after the splice silently mutates a discarded copy.
3. **Friendly output names**: torch.onnx.export may name the If's output directly (traj_cls, not If_output_0); renaming the spliced Squeeze's output to the If's output name keeps consumers, graph outputs, and every downstream eval script (which index outputs by name) unchanged.

## 7.10 nvpmodel boot mode reads the status file, not conf DEFAULT

To apply a power mode across a reboot on this AGX Orin, write /var/lib/nvpmodel/status (e.g. `pmode:0003`) AND reboot (`sudo systemctl reboot -i`). Setting /etc/nvpmodel.conf DEFAULT or running `nvpmodel -m` at runtime does not survive the boot-time apply, and `nvpmodel -m` itself refuses while a CUDA golden context exists.

Mode IDs come from THIS board's /etc/nvpmodel.conf: 0=MAXN, 1=MODE_15W, 2=MODE_30W, 3=MODE_50W. Stock tables differ per board (AGX Orin stock maps 4=MAXN); always read the table from the box's conf.

## 7.11 RTC is stuck; verify reboots via /proc/uptime

The on-board RTC is dead: `uptime -s` always prints 2026-08-12 13:32 regardless of reboots. Use `cat /proc/uptime` (seconds since boot) to verify a reboot landed (e.g. 96 s right after boot).

## 7.12 pgrep -f self-match via the launch line

A pgrep -f guard placed in the same ssh command line as the launch it guards will self-match: the remote bash cmdline contains the full string, including the plain-text launch path. The `[a]rm64` bracket trick only protects the PATTERN occurrence, not other occurrences of the same text in the same cmdline. If each sweep step reboots first, the guard is redundant anyway - drop it.

## 7.13 INT8 calib npz convention

Calibration data is a stacked npz: d[k][i] = input k of frame i. Storage is float16 (the calibrators astype float32 per batch); uncompressed np.savez - compressed writes of multi-GB float32 arrays run at ~5 MB/s. Backbone calib holds imgs [64,6,3,384,704] f32 (raw image bytes compress well, so it uses savez_compressed); head calib holds feat_0..3 [64,6,256,h,w] f16 + projection_mat + image_wh, NO feat_flat (the motion engine stays FP16 in this tier).

## 7.14 15W Myelin context-creation failure

Under MODE_15W (2 CPU cores, low clocks) the Map FP16 engine's create_execution_context() failed with "Myelin cask_fusion unknown internal error" once; the first bench crashed on ctx.set_input_shape with a None context while BB/Head had already been measured. The retry in bench_engines.py does NOT clear it - on the re-run Map still failed (it is deterministic on this board at 15W). bench_engines.py isolates per-engine failures into a "failed" list in bench_results.json instead of dying, so one engine cannot void the whole mode's data. Treat 15W Map latency as unavailable on this board and report the 3-engine sum.

## 7.15 lvl_start / feat_flat feed contract (motion engine)

The motion ONNX declares lvl_start as [4] (per-cam level start offsets, cumsum of per-cam h*w) and feat_flat as (1, n_cam, sum_hw, 256). Early scripts fed the cumsum's 5-element version and a 3-D (1, 6*sum_hw, 256) cat - both feed the same bytes as the 4-D view, so outputs were always correct, but TRT rejects setInputShape on the nbDims mismatch and prints an error per call. Fixed by slicing [:-1] on lvl_start and viewing the cat output as (1, n_cam, sum_hw, 256) (metadata-only reshape, memory untouched). Rule: match the ONNX tensor declarations exactly even when the byte counts coincide - the error spam hides real errors.

## 7.16 Any head engine build must parse sparsedrive_head_sim.onnx

build_head_int8_orin.py defaulted to the UNFOLDED sparsedrive_head.onnx; TRT fails it at parse with IIfConditionalOutputLayer shape conflict ([1,900,13,3] vs [1,900,13,3,1]) in /det_head/layers.7/kps_generator/If. The FP16 builder already used the If-folded *_sim.onnx - all head engine builds (FP16 or INT8) must parse sparsedrive_head_sim.onnx. Rule: an INT8 variant of an existing builder inherits the builder's onnx source, not the raw export.

## 7.17 One gate policy, one table, both platforms

x86 (tools/compare_npz_gates.py GATE_OVERRIDES) and Orin (arm64/eval/eval_accuracy_orin.py GATES) drifted apart: the Orin table gated mp_lon_anchor at 0.99 (measured 0.9742 on pure FP16 - unmeetable), bb_feat at 0.9999 (INT8 backbone lands ~0.98 by design - unmeetable). Result: an INT8 run that is bit-consistent with x86 reported GATES PASS: False. The x86 table is the source of truth; the Orin table now mirrors it (mp_lon_anchor 0.95, bb_feat 0.98, hd_det_quality 0.98, mp_traj_cls 0.95; reg/instance_feature/map_* 0.99). Before editing either table, check the other.
