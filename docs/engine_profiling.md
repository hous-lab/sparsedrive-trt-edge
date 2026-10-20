# 3. Engine profiling

## 3.1 Measured x86 latency

All numbers below were measured on the RTX 4090 Laptop GPU
(16 GB). The 4090 Laptop has ~57% of the desktop 4090's memory bandwidth;
do not compare with desktop results.

## 3.2 Cascade latency (FP16, batch 1)

| Metric | Value |
|---|---|
| full cascade p50 | 16.15 ms |
| full cascade p90 | 16.52 ms |
| ~FPS | ~62 FPS |

Measurement: 50 iterations after 5 warmups, `torch.cuda.synchronize()`
around each iteration, TRT 10.4.0.26 engines, plugin built from
TRT-10.4.0.26 headers. Feed: real demo frame I/O for gate eval; the bench
feeds random-normal inputs with identical shapes (latency is shape-bound,
not value-bound).

## 3.3 Cascade latency vs the 100 ms budget

Real-time ADAS budget is 100 ms (10 Hz). At p50 16.15 ms the full cascade
uses 16% of budget on a laptop GPU — real-time capable with headroom for
pre/post-processing. The bottleneck is the backbone (6 images at 384x704
per tick). 

## 3.4 Accuracy vs speed tradeoffs measured

Gates (cosine, engine cascade vs PT): motion_cls 0.9965, motion_traj
0.9895, traj_cls 0.9856, lat_anchor 0.9991, lon_anchor 0.9742 (WARN).
FP16 is effectively lossless for perception; the lon_anchor WARN reflects
argmax-sensitive velocity-profile selection near ties, see docs/lessons.md.
