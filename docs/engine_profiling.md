# 3. Engine profiling

## 3.1 Measured x86 latency

All numbers below were measured on the RTX 4090 Laptop GPU (16 GB). The 4090 Laptop has ~57% of the desktop 4090's memory bandwidth; do not compare with desktop results.

## 3.2 Cascade latency (FP16, batch 1)

| Metric | Value |
|---|---|
| full cascade p50 | 16.15 ms |
| full cascade p90 | 16.52 ms |
| ~FPS | ~62 FPS |

Measurement: 50 iterations after 5 warmups, `torch.cuda.synchronize()` around each iteration, TRT 10.4.0.26 engines, plugin built from TRT-10.4.0.26 headers. Feed: real demo frame I/O for gate eval; the bench feeds random-normal inputs with identical shapes (latency is shape-bound, not value-bound).

## 3.3 Cascade latency vs the 100 ms budget

Real-time ADAS budget is 100 ms (10 Hz). At p50 16.15 ms the full cascade uses 16% of budget on a laptop GPU — real-time capable with headroom for pre/post-processing. The bottleneck is the backbone (6 images at 384x704 per tick).

## 3.4 Accuracy vs speed tradeoffs measured

Gates (cosine, engine cascade vs PT): motion_cls 0.9965, motion_traj 0.9895, traj_cls 0.9856, lat_anchor 0.9991, lon_anchor 0.9742 (WARN). FP16 is effectively lossless for perception; the lon_anchor WARN reflects argmax-sensitive velocity-profile selection near ties, see [docs/lessons.md](lessons.md).

## 3.5 INT8 tier on x86

Same methodology as the FP16 bench (same script, same run):

| Engine | FP16 ms | INT8 ms | Gain |
|---|---|---|---|
| backbone | 5.6 | 2.7 | 2.1x |
| head | 4.4 | 4.4 | none |
| map | 6.1 | - | n/a |
| motion_plan | 5.0 | - | n/a |

Engine-sum cascade: 21.1 ms (all FP16) -> 18.2 ms (BB+Head INT8), a 14% reduction ([bench_engines.py](../arm64/eval/bench_engines.py) 200-iter means; §3.2's 16.15 ms is a separate p50 measurement on the FP16 chain). Head INT8 buys nothing: the deformable aggregation plugin (the head's dominant cost) runs FP16 by design, so INT8 MatMuls around it never become the critical path. Backbone INT8 is the win: 2.1x on the single largest engine, engine size 55.5 -> 30.9 MB, with bb_feat cosine 0.981-0.989 vs PT (all cascade gates PASS, per-key gates as §3.4; det_cls 0.9998, det_quality 0.9871, lon_anchor 0.9589 vs 0.9742 on the all-FP16 chain - still above its 0.95 gate).

## 3.6 INT8 tier on Orin (MAXN, TRT 10.3)

Built via `SD_BUILD_INT8=1 scripts/run_orin_pipeline.sh` (calib npz synced from x86, entropy calibrator v2, 64 frames). Same FP16-forced plugin / FP32-forced sensitive ops as x86. MAXN is this repo's Orin reporting standard:

| Engine | FP16 ms | INT8 ms | Gain |
|---|---|---|---|
| backbone | 22.1 | **11.3** | 1.96x |
| head | 17.3 | 17.4 | none |
| map | 26.1 | - | n/a |
| motion_plan | 20.7 | - | n/a |

Engine-sum cascade: 86.2 ms (FP16) -> 75.4 ms with BB INT8 (~12%). The INT8 gain is larger at MAXN than under the capped 30W mode (30W: BB 66.5 -> 39.4 ms, 1.69x; cascade 271.2 -> 244.1 ms, ~10%) - the quantized backbone exploits the higher clocks better. Same conclusion at any power mode: backbone INT8 is the only lever; head INT8 quantizes MatMuls around the FP16 deformable plugin and never becomes the critical path. Accuracy: three-config gates all PASS (bb_feat 0.981-0.989, hd_det_quality 0.9863, mp_lon_anchor 0.9635 - all matching x86 INT8 within 0.005 cos, [docs/model_variants.md](model_variants.md) §4.5); the INT8 tier is cross-platform reproducible. Results: [arm64/eval/results/accuracy_comparison.json](../arm64/eval/results/accuracy_comparison.json) and [arm64/eval/results/maxn_int8/bench_results.json](../arm64/eval/results/maxn_int8/bench_results.json) (MAXN standard), [arm64/eval/results/int8/bench_results.json](../arm64/eval/results/int8/bench_results.json) (30W reference).
