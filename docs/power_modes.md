# 5. Orin power modes

## 5.1 Modes under test

| Mode | nvpmodel ID | Description |
|---|---|--- |
| MAXN | 0 | all cores online, no capping |
| 50W | 2 | capped at 50 W |
| 30W | 2/3 | capped at 30 W production mode |
| 15W | 5 | two cores, capped |

## 5.2 How to measure (on the Orin device)

```bash
sudo nvpmodel -m <id> && sudo jetson_clocks
bash arm64/eval/run_power_modes.sh      # tegrastats capture + p50/p90
```

`run_power_modes.sh` runs `arm64/eval/bench_power_orin.sh` per mode:
1. `sudo nvpmodel -m <id>`, `sudo jetson_clocks` to pin clocks
2. runs the engine cascade via `arm64/eval/bench_engines.py`
3. captures tegrastats in the background, stops after the bench
4. parses the log into p50/p90 latency and p50/p90 power draw

## 5.3 Results table (fill on device)

| Mode | p50 lat (ms) | p90 lat (ms) | p50 power (W) | p90 power (W) |
|---|---|---|---|---|
| MAXN | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 50W | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 30W | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 15W | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

## 5.4 Notes

- Always run `jetson_clocks` after switching nvpmodel; otherwise clocks
  ramp slowly and p50 is meaningless.
- tegrastats reports instantaneous power; the parser averages over the
  bench window only.
- DLA off by default (docs/dla_notes.md).
