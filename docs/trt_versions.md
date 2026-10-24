# 8. Orin TensorRT version study (10.3 vs 10.4)

The Orin deployment is pinned to TensorRT 10.3, the JetPack-matched release. This doc records why: on this SoC, the 10.4.0.26 tarball builds engines that run measurably slower for this model family, the gap is localized to a handful of layers around the custom plugin, and none of the public build-time knobs - or a plugin rebuilt against 10.4 headers - closes it. x86 is unaffected; the repo keeps 10.4 there.

## 8.1 Headline numbers

Per-engine trtexec GPU compute time (mean), paired builds in one session at MAXN with jetson_clocks pinned, identical flags (--fp16 + --staticPlugins, everything else default):

| Sub-model | TRT 10.3 | TRT 10.4 | Delta |
|---|---|---|---|
| backbone (no plugin) | 23.25 ms | 23.29 ms | +0.2% (flat) |
| head_sim | 17.82 ms | 22.13 ms | +24% |
| map_sim | 22.72 ms | 35.30 ms | +55% |
| motion_plan_sim | 23.03 / 22.98 ms (2 builds) | 31.58 / 32.14 / 32.50 ms (3 builds) | +37-40% |

End-to-end trtexec latency (includes H2D of fp32 inputs) tells the same story: backbone +0.3%, head +18%, map +41%, motion +23%. Throughput on motion_plan falls from 43.1 to 31.3 qps. Cascade engine-sum: 86.8 ms (10.3) vs 112.3 ms (10.4) - the 10.3 sum cross-checks against the published bench-harness MAXN number (86.2 ms, [docs/engine_profiling.md](engine_profiling.md) 3.6), and the 10.4 sum would push the FP16 cascade past the 100 ms real-time budget. Engine sizes differ by less than 0.7% across versions: the gap is kernel choice, not memory footprint.

These trtexec numbers are within-harness only: they are not comparable to the [arm64/eval/bench_engines.py](../arm64/eval/bench_engines.py) figures elsewhere in docs (different harness, fp16 vs fp32 input feeds). Every statement in this doc compares like with like.

## 8.2 Methodology

Both TRTs live on the same box: 10.3 is the JetPack dpkg (/usr/src/tensorrt), 10.4.0.26 is the tarball (/usr/src/TensorRT-10.4.0.26). The deployable plugin .so was built against the dpkg 10.3 headers (the same libplugins_orin.so served both versions). Each comparison run is a paired build: 10.3 and 10.4 compile the same ONNX back to back with identical flags; on motion_plan the pair was run in both orders (10.3-first and 10.4-first) to rule out drift, and three independent 10.4 builds agree within 0.9 ms while the two 10.3 builds agree within 0.05 ms. The trtexec benchmark uses its defaults (warmup + a few-second window); with the pinned MAXN clocks the percentiles are flat (see 8.7 for the exact commands).

## 8.3 Mechanism: per-layer profiles

Per-layer profiles (--exportProfile) locate every missing millisecond:

- backbone: 70 layers on both versions, sum 23.226 vs 23.220 ms, top layers identical to the 0.05 ms level. In a pure-conv graph 10.4 changes nothing.
- The plugin kernel itself is bit-identical across versions (motion_plan: 4.04 / 1.75 / 1.00 ms, equal to 0.01 ms; same in head and map). The plugin binary is not the problem.
- The regression is the MatMuls feeding DeformableAggregationPlugin (the sampling_locations / weights producers). 10.3 fuses them into __myl_* kernel groups; 10.4 emits them standalone: six x 0.82 ms in head, six x 2.50 ms in map (+15 ms), and in motion_plan a 6.08 / 2.34 / 1.43 ms top-3 alone accounts for 9.85 ms. The same work costs ~0.7 ms inside a 10.3 fused group.
- Fused-layer counts drop across versions: 469 to 437 (motion_plan), 859 to 823 (head), 536 to 458 (map). 10.4 still runs its fusion engine (layers keep _myl suffixes), but groups differently around the plugin.
- The 10.3 engine inspector shows dedicated Cast layers flanking the plugin (layers.18/Cast_2/3/4); 10.4 absorbs them. Corroborating clue: an unfolded-ONNX build on 10.4 skipped one tactic with "Mismatched type for tensor traj_cls f16 vs f32". Type/layout propagation across plugin boundaries is where 10.4 changed.

Unrelated but recorded here: the plugin enqueue does a synchronous D2H int64-to-int32 conversion of the constant index tensors on every call - identical cost on both versions, a separate optimization item.

## 8.4 Fix attempts (all failed)

Target: motion_plan 10.4 from 31.6-32.1 ms back to the 10.3 level (~23 ms).

| Attempt | What it changes | GPU compute |
|---|---|---|
| stock 10.4 | baseline | 31.6-32.1 ms |
| --builderOptimizationLevel=4 | deeper tactic search | 31.64 ms (no change) |
| --builderOptimizationLevel=5 | exhaustive search | >21 min build, terminated - impractical |
| --preview=+/-aliasedPluginIO1003 | the only 10.4 preview feature (plugin I/O semantics, introduced in 10.3) | 31.65 / 31.64 ms (no change either way) |
| plugin rebuilt with 10.4 headers | 10.4-native plugin binary | 31.55 ms (no change; .so byte-size identical to the 10.3 build) |
| --tacticSources=-cublasLt | switch GEMM backend heuristic | 32.15 ms (no change) |
| --layerPrecisions fp16 on the 3 regressed MatMuls + --precisionConstraints=obey | force kernel precision | 32.72 ms (no change) |

The layerPrecisions result is the informative one: the regressed MatMuls are not running a wrong precision - the fusion grouping itself is lost, and precision cannot be forced back into a fusion that no longer exists.

## 8.5 Interplay with If-folding (_sim.onnx)

TRT 10.3 rejects the unfolded ONNX at parse (IIfConditionalOutputLayer shape conflict, [docs/lessons.md](lessons.md) 7.16). TRT 10.4 folds the If subgraphs at build: the unfolded sparsedrive_motion_plan.onnx compiles fine, with one benign skipped tactic (traj_cls f16 vs f32) and 33.07 ms GPU compute - marginally slower than the folded 31.6-32.1 ms. So 10.4 would remove the parse blocker, but on Orin the _sim.onnx folding remains required anyway (10.3 needs it) and stays marginally faster on 10.4.

## 8.6 Conclusions

1. Orin stays on the JetPack-matched TensorRT 10.3 for this model family. The gap (+24-55% GPU compute on the three plugin-bearing sub-models) is too large to buy back, and a 10.4 cascade would miss the 100 ms real-time budget. Re-evaluate at the next JetPack TRT bump.
2. The plugin does not need per-version rebuilds here: the same .so built against 10.3 headers runs identically under 10.4 (interface ABI is stable; kernel cost is bit-identical).
3. Same-TRT-across-platforms is a non-goal for this repo: x86 (sm_89) on 10.4 shows no such regression - the gap is specific to 10.4 on sm_87. The validated pairing is x86 10.4 / Orin 10.3.
4. The failed-fix log doubles as the map of public knobs that do NOT matter for this failure mode. Remaining option if 10.4 is ever needed on Orin: report upstream to NVIDIA with the minimal repro (one ONNX + the plugin .so + one trtexec command, 8.7).

## 8.7 Reproducing / artifacts

Paired build command (substitute the TRT root and ONNX name; the sim variants are required for head/map/motion_plan):

```bash
LD_LIBRARY_PATH=<trt>/lib <trt>/bin/trtexec --onnx=onnx_export/output/sparsedrive_motion_plan_sim.onnx --saveEngine=/tmp/mp.engine --staticPlugins=arm64/plugins_build/build/libplugins_orin.so --fp16
```

Optional evidence flags: --exportProfile=<file>.json (per-layer times), --exportLayerInfo=<file>.json (fused layer names/counts). Profile the built engines by adding --loadEngine plus --exportProfile. All artifacts from the study (about 15 engines, per-layer profiles for all four models on both versions, inspector exports, logs and driver scripts) live on the Orin device under ~/exp_trt104_unfolded/ and are regenerable; nothing is committed to the repo.
