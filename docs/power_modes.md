# 5. Orin power modes

## 5.1 Modes under test

Mode IDs are board-specific - read the table from the box's /etc/nvpmodel.conf before trusting any mapping (this AGX Orin's table below):

| Mode | ID on this board | Description |
|---|---|---|
| MAXN | 0 | all cores online, no capping |
| 50W | 3 | capped at 50 W |
| 30W | 2 | capped at 30 W |
| 15W | 1 | capped at 15 W |

On stock AGX Orin the table differs (4=MAXN); Orin NX/Nano differ again.

## 5.2 How to measure (on the Orin device)

Preferred on a healthy box:

```bash
sudo nvpmodel -m <id> && sudo jetson_clocks
bash arm64/eval/run_power_modes.sh      # per-mode bench + tegrastats
```

On our box, runtime `nvpmodel -m` refuses while a CUDA golden context exists and does not survive reboot (docs/lessons.md §7.10): the sweep driver writes /var/lib/nvpmodel/status (`pmode:000N`) and reboots with `sudo systemctl reboot -i` per mode, then verifies the active mode via `nvpmodel -q` before benching. `jetson_clocks` is applied only at MAXN (run_power_modes.sh) to pin clocks; capped modes keep their caps.

## 5.3 Results table (fill on device)

Per-engine latency from bench_engines.py (200-iter mean per engine):

| Mode | BB | Head | Map | Motion | Sum | VDD_GPU_SOC (C1) |
|---|---|---|---|---|---|---|
| MAXN | 23.7 | 17.3 | 26.5 | 20.7 | 88.2 ms | 39.0 W |
| 50W | 32.8 | 24.8 | 38.6 | 28.7 | 124.9 ms | 23.7 W |
| 30W | 74.3 | 52.5 | 88.4 | 64.9 | 280.1 ms | 11.5 W |
| 15W | 145.7 | 101.3 | n/a (ctx fail, §7.14) | 126.3 | 373.3 ms* | 6.8 W |

\*15W sum is the 3-engine sum; the Map engine cannot create its execution
context on this board at 15W (deterministic TRT cask_fusion failure, docs/lessons.md §7.14).

Power is the tegrastats VDD_GPU_SOC average over the C1 (backbone FP16 trtexec) window - VIN_SYS_5V0 reads ~9 W even at MAXN on this devkit and is not trusted.

The FP16 cascade only meets the 100 ms budget at MAXN. At 50W it is 125 ms (8 Hz); at 30W/15W it falls to 3.8/2.7 Hz. The INT8 tier helps exactly where it should. Under MAXN (the repo's reporting standard), BB INT8 22.1 -> 11.3 ms (1.96x) and the cascade 86.2 -> 75.4 ms (~12%, docs/engine_profiling.md §3.6). Under the capped 30W mode the same tier moves BB 66.5 -> 39.4 ms (1.69x), cascade 271.2 -> 244.1 ms (~10%) - backbone INT8 lifts every mode; head INT8 does not move the plugin-bound critical path on any power mode.

## 5.4 Notes

- Always run `jetson_clocks` at MAXN before benching; otherwise clocks ramp slowly and p50 is meaningless. Capped modes keep their caps.
- tegrastats reports instantaneous rail power; the parser averages over the bench window only.
- DLA off by default (docs/dla_notes.md); the C1/C2/C3 trtexec experiment (arm64/eval/bench_power_orin.sh) samples power under continuous load.
- Accuracy is mode-independent - gates are re-checked per config, not per power mode.
- nvpmodel writes power caps into the boot-time status file on this board; the sweep driver restores mode 2 (30W) after the sweep.
