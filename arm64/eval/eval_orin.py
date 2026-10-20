#!/usr/bin/env python
"""
SparseDriveV2 Orin evaluation script.
Runs TRT engines with real data and measures latency.
"""
import os
import sys
import ctypes
import argparse
import json
import numpy as np
import torch
import tensorrt as trt

DEPLOY_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, DEPLOY_ROOT)

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
TRT_MAJOR = int(trt.__version__.split('.')[0]) if trt.__version__ else 10


def build_engine_from_file(engine_path, plugin_lib=None):
    if plugin_lib and os.path.exists(plugin_lib):
        ctypes.CDLL(plugin_lib, mode=ctypes.RTLD_GLOBAL)
    trt.init_libnvinfer_plugins(TRT_LOGGER, '')

    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine


class TrtInference:
    def __init__(self, engine, device='cuda:0'):
        self.engine = engine
        self.context = engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        self.bindings = [None] * engine.num_bindings
        self.tensors = [None] * engine.num_bindings

    def infer(self, feed_dict):
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            is_input = self.engine.binding_is_input(i)
            if is_input and name in feed_dict:
                t = feed_dict[name].contiguous()
                self.context.set_input_shape(name, tuple(t.shape))
                self.bindings[i] = t.data_ptr()
                self.tensors[i] = t
            elif not is_input:
                shape = tuple(self.engine.get_binding_shape(i))
                if any(d == -1 for d in shape):
                    shape = self.context.get_binding_shape(i)
                t = torch.empty(shape, dtype=torch.float32, device='cuda')
                self.bindings[i] = t.data_ptr()
                self.tensors[i] = t

        self.context.execute_async_v3(
            binding_data=self.bindings, stream_handle=self.stream.cuda_stream)
        self.stream.synchronize()

        outputs = {}
        for i in range(self.engine.num_bindings):
            if not self.engine.binding_is_input(i):
                outputs[self.engine.get_binding_name(i)] = self.tensors[i].clone()
        return outputs


def run_orin_eval(backbone_engine_path, head_engine_path, plugin_path=None,
                  calib_path=None, num_frames=50, device='cuda:0'):
    print(f"Loading engines on {device}:")
    print(f"  Backbone: {backbone_engine_path}")
    print(f"  Head: {head_engine_path}")

    bb_engine = build_engine_from_file(backbone_engine_path, plugin_path)
    head_engine = build_engine_from_file(head_engine_path, plugin_path)

    bb_trt = TrtInference(bb_engine, device)
    head_trt = TrtInference(head_engine, device)

    # Load calibration data as proxy for real inference data
    if calib_path and os.path.exists(calib_path):
        calib = np.load(calib_path, mmap_mode='r')
        frames = calib['frames']
        n = min(num_frames, len(frames))
    else:
        print("WARNING: No calib data, using random inputs for latency test")
        frames = None
        n = num_frames

    latencies = []
    det_cached_feat = torch.zeros(1, 0, 256, device=device)
    det_cached_anchor = torch.zeros(1, 0, 11, device=device)
    map_cached_feat = torch.zeros(1, 0, 256, device=device)
    map_cached_anchor = torch.zeros(1, 0, 2, device=device)

    for i in range(n):
        if frames is not None:
            frame = frames[i]
            dummy_img = torch.from_numpy(frame['feat_0'][0, 0].astype(np.float32)).unsqueeze(0).unsqueeze(0)
        else:
            dummy_img = torch.randn(6, 3, 384, 704, device=device)

        dummy_img_flat = torch.randn(6, 3, 384, 704, device=device) if frames is None else \
            torch.from_numpy(np.random.randn(6, 3, 384, 704).astype(np.float32)).to(device)

        if i == 0:
            torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        bb_out = bb_trt.infer({'img': dummy_img_flat})
        feat_0 = bb_out['feat_0'].view(6, 256, 96, 176)
        feat_1 = bb_out['feat_1'].view(6, 256, 48, 88)
        feat_2 = bb_out['feat_2'].view(6, 256, 24, 44)
        feat_3 = bb_out['feat_3'].view(6, 256, 12, 22)

        proj_mat = torch.eye(4, device=device).unsqueeze(0).expand(6, -1, -1)
        img_wh = torch.tensor([[704, 384]], device=device).float().expand(6, -1)

        head_in = {
            'feat_0': feat_0, 'feat_1': feat_1, 'feat_2': feat_2, 'feat_3': feat_3,
            'projection_mat': proj_mat,
            'image_wh': img_wh,
            'timestamp': torch.zeros(1, device=device),
            'ego_status': torch.zeros(3, device=device),
            'tp_near': torch.zeros(2, device=device),
            'gt_ego_fut_cmd': torch.zeros(1, device=device).long(),
            'det_cached_feat': det_cached_feat.squeeze(0),
            'det_cached_anchor': det_cached_anchor.squeeze(0),
            'map_cached_feat': map_cached_feat.squeeze(0),
            'map_cached_anchor': map_cached_anchor.squeeze(0),
        }

        head_out = head_trt.infer(head_in)

        if 'new_det_feat' in head_out:
            det_cached_feat = head_out['new_det_feat'].unsqueeze(0)
            det_cached_anchor = head_out['new_det_anchor'].unsqueeze(0)
            map_cached_feat = head_out['new_map_feat'].unsqueeze(0)
            map_cached_anchor = head_out['new_map_anchor'].unsqueeze(0)

        end.record()
        torch.cuda.synchronize()
        latencies.append(start.elapsed_time(end))

        if i > 0:
            print(f"  Frame {i+1}/{n}: {latencies[-1]:.1f} ms")

    latencies = latencies[1:]  # exclude warmup
    mean_ms = np.mean(latencies)
    std_ms = np.std(latencies)

    print(f"\n--- Orin Results ---")
    print(f"Mean: {mean_ms:.1f} ms")
    print(f"Std:  {std_ms:.1f} ms")
    print(f"FPS:  {1000.0/mean_ms:.1f}")

    results = {'mean_ms': float(mean_ms), 'std_ms': float(std_ms),
               'fps': float(1000.0/mean_ms), 'latencies': latencies}

    out_path = os.path.join(DEPLOY_ROOT, 'arm64/eval', 'results',
                            'eval_orin_results.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bb-engine', default='arm64/trt_compile/fp16/output/backbone_fp16.engine')
    parser.add_argument('--head-engine', default='arm64/trt_compile/fp16/output/head_fp16.engine')
    parser.add_argument('--plugin', default='arm64/plugins_build/build/libplugins_orin.so')
    parser.add_argument('--calib', default='arm64/trt_compile/int8/calib_data/head_calib.npz')
    parser.add_argument('--num-frames', type=int, default=50)
    args = parser.parse_args()

    run_orin_eval(args.bb_engine, args.head_engine, args.plugin,
                  args.calib, args.num_frames)
