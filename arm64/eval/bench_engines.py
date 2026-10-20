"""Engine-level latency benchmark on Orin (BB / Head / Map / Motion engines).

Benches individual engines with random inputs (latency only, no accuracy).
Writes arm64/eval/results/bench_results.json (regenerable).

Usage (on Orin):
    python arm64/eval/bench_engines.py

Env overrides:
    SD_ENGINE_DIR       FP16 engines dir (default arm64/trt_compile/fp16/output)
    SD_ENGINE_INT8_DIR  INT8 engines dir (default arm64/trt_compile/int8/output)
    SD_PLUGIN_LIB       plugin lib (default arm64/plugins_build/build/libplugins_orin.so)
    SD_RESULTS_DIR      results dir (default arm64/eval/results)
"""
import os, ctypes, time, json
import numpy as np
import tensorrt as trt
import torch

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
ENGINE_FP16_DIR = os.environ.get('SD_ENGINE_DIR', os.path.join(REPO, 'arm64/trt_compile/fp16/output'))
ENGINE_INT8_DIR = os.environ.get('SD_ENGINE_INT8_DIR', os.path.join(REPO, 'arm64/trt_compile/int8/output'))
PLUGIN = os.environ.get('SD_PLUGIN_LIB', os.path.join(REPO, 'arm64/plugins_build/build/libplugins_orin.so'))
OUT_DIR = os.environ.get('SD_RESULTS_DIR', os.path.join(REPO, 'arm64/eval/results'))

FEAT_SHAPES = [(96, 176), (48, 88), (24, 44), (12, 22)]
LEVEL_SIZES = [h * w for h, w in FEAT_SHAPES]
N_TOKENS = sum(LEVEL_SIZES)  # 22440


def load_engine(path, plugin=None):
    if plugin and os.path.exists(plugin):
        ctypes.CDLL(plugin, mode=ctypes.RTLD_GLOBAL)
    trt.init_libnvinfer_plugins(TRT_LOGGER, "")
    with open(path, "rb") as f:
        return trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(f.read())


def bench(eng, feeds, x86=0, warmup_sec=3, iters=200):
    ctx = eng.create_execution_context()
    if ctx is None:
        # 15W caps have been seen to fail Myelin context creation (cask_fusion
        # internal error) transiently; one retry usually clears it.
        time.sleep(2)
        ctx = eng.create_execution_context()
    if ctx is None:
        raise RuntimeError('execution context creation failed (Myelin/cask), see docs/lessons.md 7.14')
    s = torch.cuda.Stream()
    for i in range(eng.num_io_tensors):
        name = eng.get_tensor_name(i)
        if eng.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            ctx.set_input_shape(name, tuple(feeds[name].shape))
            ctx.set_tensor_address(name, feeds[name].data_ptr())
        else:
            shape = list(eng.get_tensor_shape(name))
            for j, d in enumerate(shape):
                if d == -1:
                    shape[j] = ctx.get_tensor_shape(name)[j]
            ctx.set_tensor_address(name, torch.empty(shape, device="cuda").data_ptr())
    # wall-clock warmup, then CUDA-event timed loop
    n = 0
    t0 = time.time()
    while time.time() - t0 < warmup_sec:
        ctx.execute_async_v3(s.cuda_stream)
        s.synchronize()
        n += 1
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        ctx.execute_async_v3(s.cuda_stream)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters, n / warmup_sec


def motion_feeds():
    """Random feeds matching sparsedrive_motion_plan.onnx I/O (latency only)."""
    d = torch.device('cuda')
    return {
        'feat_flat': torch.randn(1, 6, N_TOKENS, 256, device=d) * 0.1,
        'spat_shape': torch.tensor([[h, w] for h, w in FEAT_SHAPES], device=d, dtype=torch.int32),
        # lvl_start is [num_levels]=4 - drop the cumsum trailing sentinel.
        'lvl_start': torch.tensor(np.cumsum([0] + LEVEL_SIZES)[:-1], device=d, dtype=torch.int32),
        'det_cls': torch.randn(1, 900, 9, device=d),
        'det_reg': torch.randn(1, 900, 11, device=d),
        'det_quality': torch.randn(1, 900, 2, device=d),
        'det_instance_feat': torch.randn(1, 900, 256, device=d),
        'det_anchor_embed': torch.randn(1, 900, 256, device=d),
        'map_cls': torch.randn(1, 100, 6, device=d),
        'map_reg': torch.randn(1, 100, 2, device=d),
        'map_instance_feat': torch.randn(1, 100, 256, device=d),
        'map_anchor_embed': torch.randn(1, 100, 256, device=d),
        'projection_mat': torch.eye(4, device=d).unsqueeze(0).expand(6, -1, -1).contiguous(),
        'image_wh': torch.tensor([[704, 384]], device=d).float().expand(6, -1).contiguous(),
        'timestamp': torch.zeros(1, device=d),
        'ego_status': torch.randn(10, device=d),
        'tp_near': torch.zeros(2, device=d),
        'gt_ego_fut_cmd': torch.zeros(6, device=d, dtype=torch.int64),
        'mask': torch.full((900,), -1.0, device=d),
        'queue_temp_feat': torch.randn(21, 1, 901, 256, device=d) * 0.1,
        'queue_temp_anchor': torch.randn(21, 1, 901, 256, device=d) * 0.1,
        'period': torch.full((1, 901), 20, device=d, dtype=torch.int64),
    }


def main():
    tables = [
        ("BB_FP16", os.path.join(ENGINE_FP16_DIR, "backbone_fp16.engine"), None,
         {"img": torch.randn(6, 3, 384, 704, device="cuda")}),
        ("BB_INT8", os.path.join(ENGINE_INT8_DIR, "backbone_int8.engine"), None,
         {"img": torch.randn(6, 3, 384, 704, device="cuda")}),
        ("Head_FP16", os.path.join(ENGINE_FP16_DIR, "head_fp16.engine"), PLUGIN,
         {"feat_0": torch.randn(6, 256, 96, 176, device="cuda"),
          "feat_1": torch.randn(6, 256, 48, 88, device="cuda"),
          "feat_2": torch.randn(6, 256, 24, 44, device="cuda"),
          "feat_3": torch.randn(6, 256, 12, 22, device="cuda"),
          "projection_mat": torch.eye(4, device="cuda").unsqueeze(0).expand(6, -1, -1),
          "image_wh": torch.tensor([[704, 384]], device="cuda").float().expand(6, -1)}),
        ("Head_INT8", os.path.join(ENGINE_INT8_DIR, "head_int8.engine"), PLUGIN,
         {"feat_0": torch.randn(6, 256, 96, 176, device="cuda"),
          "feat_1": torch.randn(6, 256, 48, 88, device="cuda"),
          "feat_2": torch.randn(6, 256, 24, 44, device="cuda"),
          "feat_3": torch.randn(6, 256, 12, 22, device="cuda"),
          "projection_mat": torch.eye(4, device="cuda").unsqueeze(0).expand(6, -1, -1),
          "image_wh": torch.tensor([[704, 384]], device="cuda").float().expand(6, -1)}),
        ("Map_FP16", os.path.join(ENGINE_FP16_DIR, "map_fp16.engine"), PLUGIN,
         {"feat_0": torch.randn(6, 256, 96, 176, device="cuda"),
          "feat_1": torch.randn(6, 256, 48, 88, device="cuda"),
          "feat_2": torch.randn(6, 256, 24, 44, device="cuda"),
          "feat_3": torch.randn(6, 256, 12, 22, device="cuda"),
          "projection_mat": torch.eye(4, device="cuda").unsqueeze(0).expand(6, -1, -1),
          "image_wh": torch.tensor([[704, 384]], device="cuda").float().expand(6, -1)}),
        ("Motion_FP16", os.path.join(ENGINE_FP16_DIR, "motion_plan_fp16.engine"), PLUGIN,
         motion_feeds()),
    ]
    results = []
    missing = []
    failed = []
    for name, path, plugin, feeds in tables:
        if not os.path.exists(path):
            missing.append({"name": name, "path": path})
            print(f"[skip] {name}: engine not found: {path}")
            continue
        eng = load_engine(path, plugin)
        try:
            ms, fps = bench(eng, feeds)
        except Exception as e:
            failed.append({"name": name, "path": path, "error": str(e)[:200]})
            print(f"[fail] {name}: {e}")
            del eng
            torch.cuda.empty_cache()
            continue
        results.append({"name": name, "ms": round(ms, 1), "fps": round(1000.0 / ms, 1)})
        print(f"{name}: {ms:.1f} ms ({1000.0/ms:.1f} FPS)")
        del eng
        torch.cuda.empty_cache()

    by = {r["name"]: r["ms"] for r in results}
    def _total(*names):
        return sum(by[n] for n in names if n in by)
    fp16_total = _total("BB_FP16", "Head_FP16", "Map_FP16", "Motion_FP16")
    print(f"\nTotal (BB+Head+Map+Motion, FP16): {fp16_total:.1f} ms")
    if "BB_INT8" in by:
        i8_total = _total("BB_INT8", "Head_FP16", "Map_FP16", "Motion_FP16")
        print(f"Total (BB_INT8+Head+Map+Motion, FP16 rest): {i8_total:.1f} ms")

    out = os.path.join(OUT_DIR, "bench_results.json")
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"engines": results, "missing": missing, "failed": failed}, f, indent=2)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
