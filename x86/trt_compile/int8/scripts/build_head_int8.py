import os
import sys
import ctypes
import argparse
import numpy as np
import tensorrt as trt

DEPLOY_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, DEPLOY_ROOT)

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

FP32_LAYER_KEYWORDS = ['layernorm', 'layer_norm', 'softmax', 'sigmoid',
                        'sin', 'cos', 'pow', 'sqrt', 'erf',
                        'reducesum', 'reducemean', 'reducemax', 'reducemin']
PLUGIN_LAYER_KEYWORDS = ['DeformableAggregation', 'deformable_aggregation']


class HeadRealCalibrator(trt.IInt8EntropyCalibrator2):
    """Stacked-npz calibrator: d[k][i] = frame i input k (feat_0..3, metas)."""

    def __init__(self, calib_path, batch_size=1):
        super().__init__()
        self.batch_size = batch_size
        self.d = np.load(calib_path)
        self.num_frames = self.d['feat_0'].shape[0]
        self.batch_idx = 0
        self._batch = None

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.batch_idx >= self.num_frames:
            return None
        import torch
        self.batch_idx += 1
        tensors = [torch.from_numpy(np.ascontiguousarray(
            self.d[name][self.batch_idx - 1], dtype=np.float32)).float().cuda() for name in names]
        self._device_mems = tensors
        return [int(t.data_ptr()) for t in tensors]

    def read_calibration_cache(self):
        return None

    def write_calibration_cache(self, cache):
        pass


def build_engine(builder, network, config):
    if hasattr(builder, 'build_serialized_network'):
        return builder.build_serialized_network(network, config)
    return builder.build_cuda_engine(network, config).serialize()


def set_workspace(config, size_bytes):
    if hasattr(config, 'set_memory_pool_limit'):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, size_bytes)
    else:
        config.max_workspace_size = size_bytes


def force_layer_precision(network):
    """FIX 2: MatMul FP16, plugins FP16, sensitive ops FP32."""
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        name = layer.name.lower()

        if layer.type == trt.LayerType.MATRIX_MULTIPLY:
            layer.precision = trt.float16
            for j in range(layer.num_outputs):
                layer.set_output_type(j, trt.float16)
            continue

        if any(kw in name for kw in PLUGIN_LAYER_KEYWORDS):
            layer.precision = trt.float16
            for j in range(layer.num_outputs):
                layer.set_output_type(j, trt.float16)
            continue

        if any(kw in name for kw in FP32_LAYER_KEYWORDS):
            layer.precision = trt.float32
            for j in range(layer.num_outputs):
                layer.set_output_type(j, trt.float32)


def build_head_int8(onnx_path, engine_path, calib_path, plugin_lib=None):
    if plugin_lib and os.path.exists(plugin_lib):
        ctypes.CDLL(plugin_lib, mode=ctypes.RTLD_GLOBAL)
        print(f"[build_head_int8] Loaded plugin: {plugin_lib}")

    trt.init_libnvinfer_plugins(TRT_LOGGER, '')

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    print(f"[build_head_int8] Parsing ONNX: {onnx_path}")
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    # FIX 3: Use INT8 + FP16 + PREFER (not OBEY)
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
    set_workspace(config, 8 * 1024 * 1024 * 1024)

    # FIX 1: Real data calibrator
    calibrator = HeadRealCalibrator(calib_path)
    config.int8_calibrator = calibrator

    # FIX 2: Force layer precisions
    force_layer_precision(network)

    serialized = build_engine(builder, network, config)
    with open(engine_path, 'wb') as f:
        f.write(serialized)

    print(f"[build_head_int8] Done: {engine_path}")
    print(f"[build_head_int8] Size: {os.path.getsize(engine_path) / 1024 / 1024:.1f} MB")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', default='onnx_export/output/sparsedrive_head_sim.onnx')
    parser.add_argument('--engine', default='x86/trt_compile/int8/output/head_int8.engine')
    parser.add_argument('--calib', default='x86/trt_compile/int8/calib_data/head_calib.npz')
    parser.add_argument('--plugin', default='plugins/build/libplugins.so')
    args = parser.parse_args()
    build_head_int8(args.onnx, args.engine, args.calib, args.plugin)
