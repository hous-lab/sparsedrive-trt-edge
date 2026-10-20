#!/usr/bin/env python
"""
Compare TRT FP16 vs INT8 output similarity on Orin.
Uses cosine similarity (same methodology as VAD Orin eval).
"""
import os
import sys
import ctypes
import argparse
import json
import numpy as np
import torch
import tensorrt as trt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from eval_orin import build_engine_from_file


def cos_sim(a, b):
    a_f = a.flatten().float().cpu().numpy()
    b_f = b.flatten().float().cpu().numpy()
    return np.dot(a_f, b_f) / (np.linalg.norm(a_f) * np.linalg.norm(b_f) + 1e-12)


def compare_fp16_int8(fp16_engine_path, int8_engine_path, plugin_path,
                      calib_path, num_frames=10):
    fp16_engine = build_engine_from_file(fp16_engine_path, plugin_path)
    int8_engine = build_engine_from_file(int8_engine_path, plugin_path)

    fp16_ctx = fp16_engine.create_execution_context()
    int8_ctx = int8_engine.create_execution_context()
    stream = torch.cuda.Stream()

    calib = np.load(calib_path, mmap_mode='r')
    frames = calib['frames']

    similarities = {}
    count = 0

    for i in range(min(num_frames, len(frames))):
        dummy = torch.randn(6, 3, 384, 704, device='cuda')

        # Run both engines
        for ctx, engine in [(fp16_ctx, fp16_engine), (int8_ctx, int8_engine)]:
            bindings = []
            for j in range(engine.num_bindings):
                if engine.binding_is_input(j):
                    t = dummy.contiguous()
                    ctx.set_input_shape(engine.get_binding_name(j), tuple(t.shape))
                else:
                    shape = tuple(engine.get_binding_shape(j))
                    if any(d == -1 for d in shape):
                        shape = ctx.get_binding_shape(j)
                    t = torch.empty(shape, device='cuda')
                bindings.append(t.data_ptr())
            ctx.execute_async_v3(binding_data=bindings, stream_handle=stream.cuda_stream)
        stream.synchronize()

        count += 1

    sims = []
    for name in sims:
        avg = np.mean(sims[name]) if sims[name] else 0
        status = "GOOD" if avg > 0.997 else "WARN" if avg > 0.99 else "FAIL"
        print(f"  {name}: cos_sim={avg:.6f} [{status}]")

    overall = np.mean([np.mean(v) for v in similarities.values()]) if similarities else 0
    print(f"\n  Overall cos_sim avg = {overall:.6f}")
    print(f"  PASS" if overall > 0.99 else "  FAIL")

    result = {
        'overall_cos_sim': float(overall),
        'num_frames': count,
        'per_output': {k: float(np.mean(v)) for k, v in similarities.items()}
    }

    out_path = os.path.join(os.path.dirname(__file__), '..', 'results',
                            'fp16_vs_int8_comparison.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp16', default='arm64/trt_compile/fp16/output/head_fp16.engine')
    parser.add_argument('--int8', default='arm64/trt_compile/int8/output/head_int8.engine')
    parser.add_argument('--plugin', default='arm64/plugins_build/build/libplugins_orin.so')
    parser.add_argument('--calib', default='arm64/trt_compile/int8/calib_data/head_calib.npz')
    parser.add_argument('--num-frames', type=int, default=10)
    args = parser.parse_args()
    compare_fp16_int8(args.fp16, args.int8, args.plugin, args.calib, args.num_frames)
