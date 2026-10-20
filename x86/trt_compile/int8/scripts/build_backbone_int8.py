import os
import sys
import ctypes
import argparse
import numpy as np
import tensorrt as trt

DEPLOY_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, DEPLOY_ROOT)

TRT_LOGGER = trt.Logger(trt.Logger.INFO)


class BackboneCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, calib_path, batch_size=1, input_shape=(6, 3, 384, 704)):
        super().__init__()
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.calib_data = np.load(calib_path)['imgs']
        self.num_frames = len(self.calib_data)
        self.batch_idx = 0
        self.device_input = None

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.batch_idx >= self.num_frames:
            return None
        import torch
        data = self.calib_data[self.batch_idx].astype(np.float32)
        self._device_mem = torch.from_numpy(data).float().cuda()
        self.batch_idx += 1
        return [int(self._device_mem.data_ptr())]

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


def build_backbone_int8(onnx_path, engine_path, calib_path, plugin_lib=None):
    if plugin_lib and os.path.exists(plugin_lib):
        ctypes.CDLL(plugin_lib, mode=ctypes.RTLD_GLOBAL)

    trt.init_libnvinfer_plugins(TRT_LOGGER, '')

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    print(f"[build_backbone_int8] Parsing ONNX: {onnx_path}")
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
    set_workspace(config, 8 * 1024 * 1024 * 1024)

    calibrator = BackboneCalibrator(calib_path)
    config.int8_calibrator = calibrator

    serialized = build_engine(builder, network, config)
    with open(engine_path, 'wb') as f:
        f.write(serialized)

    print(f"[build_backbone_int8] Done: {engine_path}")
    print(f"[build_backbone_int8] Size: {os.path.getsize(engine_path) / 1024 / 1024:.1f} MB")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', default='onnx_export/output/sparsedrive_backbone.onnx')
    parser.add_argument('--engine', default='x86/trt_compile/int8/output/backbone_int8.engine')
    parser.add_argument('--calib', default='x86/trt_compile/int8/calib_data/backbone_calib.npz')
    parser.add_argument('--plugin', default=None)
    args = parser.parse_args()
    build_backbone_int8(args.onnx, args.engine, args.calib, args.plugin)
