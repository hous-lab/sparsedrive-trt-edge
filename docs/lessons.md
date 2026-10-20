# 7. Lessons (debugging log)

## 7.1 The kINT64 / TRT-headers trap

Symptom: engine build fails with "could not find any supported formats" on
the deformable node. Root cause chain:

1. CMake had no valid TENSORRT_ROOT and fell back to stale apt TRT 8.6.1
   headers; the plugin compiled against TRT 8 API.
2. TRT 10.4 DOES have `DataType::kINT64`. The first "fix" (a
   `#if NV_TENSORRT_MAJOR >= 10` guard that dropped the int64 branch) was
   wrong and *caused* the failure: the plugin must ACCEPT int64 plugin
   inputs, because spatial_shape/level_start_index arrive from ONNX as
   int64 constants.
3. The real fix: build the plugin with
   `TRT_ROOT=/usr/src/TensorRT-10.4.0.26` (build_x86.sh now auto-detects
   the tarball matching the python wheel) and keep the plugin accepting
   kINT32|kINT64, casting to int32 inside enqueue.

## 7.2 chdir poisoning

Scripts that `os.chdir(project_root)` at import time poison every later
relative-path computation. Rules (all export scripts follow them):

- compute REPO/ROOT absolute constants BEFORE any chdir;
- make --output/--config/--checkpoint defaults absolute at parse time;
- abspath any relative --output argument against REPO, not CWD.

## 7.3 Double-unsqueeze metas bug

The collated batch tensors are already batch-first: `projection_mat
[1,6,4,4]`. Adding another unsqueeze(0) produced [1,1,6,4,4] and a
confusing matmul error ("size of tensor a (6) vs b (13)"). When calling
sub-heads directly, build metas as `dict(collated)` — no extra batch dim.

## 7.4 Engine I/O conventions (squeezed metas)

Engines take SQUEEZED per-sample metas: projection_mat [6,4,4], image_wh
[6,2], ego_status [10], tp_near [2], gt_ego_fut_cmd [6] i64, timestamp
[1]. The demo npz saves exactly these shapes (§2.2-2.3).

## 7.5 map_reg zeros bridge

The map engine has no regression output, so the deployed chain feeds
map_reg = zeros [1,100,2] into the motion engine; gen_demo_data mirrors
this on the PT side (zeroed before motion_plan_head) so gates compare
like with like.

## 7.6 lon_anchor WARN (0.9742)

cos-sim 0.9742 < 0.99 gate but > gate-0.02 (WARN band). Mechanism: the
planning head selects a velocity profile by argmax over 1024 (64x16)
anchor scores; near-ties flip under FP16. Perception outputs stay >0.98.
Treat as a known FP16 artifact of argmax-style selection, documented in
x86/eval/results/motion_v2_gate.json.

## 7.7 TRT-8.6.1 apt headers vs tarballs

If `bash -n` builds succeed but engine builds fail, check which TRT headers
the plugin linked. `objdump -d libplugins.so` plus a python
`tensorrt.int64 == 8` check settles it in one minute.

## 7.8 numpy 2.x pickles on numpy 1.x

Info pickles from numpy 2.x envs need the `numpy._core` shim (§1.3).

## 7.9 Building on Jetson (TRT 10.3, dpkg layout)

JetPack's TRT differs from the x86 tarball in three ways; build_orin.sh
handles all three:

1. **Headers/libs live in multiarch dirs** (/usr/include/aarch64-linux-gnu,
   /usr/lib/aarch64-linux-gnu), not in a $TRT_ROOT/{include,lib} tree. The
   script builds a symlink-stub TRT root pointing at both.
2. **An old tarball may shadow the dpkg TRT**: /usr/src/TensorRT-8.6.1.6
   ships with JetPack and its headers are TRT 8. Probing for "any tarball"
   picks it and the plugin builds against the WRONG headers. The script
   only accepts tarballs whose NvInfer.h defines NV_TENSORRT_MAJOR 10.
3. **nvcc is not on the non-interactive PATH**: CMake needs
   -DCMAKE_CUDA_COMPILER=$CUDA_ROOT/bin/nvcc, and a stale CMakeCache.txt
   from a failed run poisons CMAKE_CUDA_ARCHITECTURES — build_orin.sh
   clears the cache before configuring.

With TRT 10.3's builder, the exported ONNX is also rejected:
"IIfConditionalOutputLayer inputs must have the same shape
([1,900,13,3] vs [1,900,13,3,1])". These are shape-guarded squeezes
(x.squeeze(-1) if x.shape[-1] == 1 else x) that torch.onnx.export emits
for no-arg .squeeze() calls; TRT 10.4 folds them at build time, 10.3 does
not. tools/fold_if_subgraphs.py inlines the Squeeze branch statically and
writes <name>_sim.onnx; both x86 and Orin now build engines from the _sim
files, so one set of ONNX serves both.

Folding has its own traps (each cost a debugging round-trip):
1. **Topological order**: the branch nodes must be spliced at the If node's
   own position, not appended at the graph end — consumers (TopK/Slice)
   would otherwise precede their new input ("Nodes in a graph must be
   topologically sorted" / TRT "Trying to access weights or a null tensor").
2. **Protobuf insert copies**: `g.node.insert(idx, nb)` copies the message,
   so rename branch outputs to the If's output names BEFORE splicing.
   Renaming after the splice silently mutates a discarded copy.
3. **Friendly output names**: torch.onnx.export may name the If's output
   directly (traj_cls, not If_output_0); renaming the spliced Squeeze's
   output to the If's output name keeps consumers, graph outputs, and every
   downstream eval script (which index outputs by name) unchanged.
