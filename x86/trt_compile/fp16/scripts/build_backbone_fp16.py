import os
import sys
import ctypes
import argparse
import numpy as np
import tensorrt as trt

DEPLOY_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, DEPLOY_ROOT)

TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def build_engine(builder, network, config):
    if hasattr(builder, 'build_serialized_network'):
        return builder.build_serialized_network(network, config)
    return builder.build_cuda_engine(network, config).serialize()


def set_workspace(config, size_bytes):
    if hasattr(config, 'set_memory_pool_limit'):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, size_bytes)
    else:
        config.max_workspace_size = size_bytes


def build_backbone_fp16(onnx_path, engine_path, plugin_lib=None):
    if plugin_lib and os.path.exists(plugin_lib):
        ctypes.CDLL(plugin_lib, mode=ctypes.RTLD_GLOBAL)
        print(f"[build_backbone_fp16] Loaded plugin: {plugin_lib}")

    trt.init_libnvinfer_plugins(TRT_LOGGER, '')

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    print(f"[build_backbone_fp16] Parsing ONNX: {onnx_path}")
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
    set_workspace(config, 8 * 1024 * 1024 * 1024)

    serialized = build_engine(builder, network, config)
    with open(engine_path, 'wb') as f:
        f.write(serialized)

    print(f"[build_backbone_fp16] Done: {engine_path}")
    print(f"[build_backbone_fp16] Size: {os.path.getsize(engine_path) / 1024 / 1024:.1f} MB")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', type=str,
                        default='onnx_export/output/sparsedrive_backbone.onnx')
    parser.add_argument('--engine', type=str,
                        default='x86/trt_compile/fp16/output/backbone_fp16.engine')
    parser.add_argument('--plugin', type=str, default=None)
    args = parser.parse_args()
    build_backbone_fp16(args.onnx, args.engine, args.plugin)
