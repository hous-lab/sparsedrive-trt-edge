#!/usr/bin/env python
"""Generic ONNX -> TRT FP16 engine build (x86), mirrors build_head_fp16.py.

Usage:
  python build_onnx_engine.py --onnx <model.onnx> --engine <out.engine> \
      [--plugin <libplugins.so>] [--fp32-keywords ...]
"""
import os, sys, ctypes, argparse, tensorrt as trt
DEPLOY_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, DEPLOY_ROOT)
TRT_LOGGER = trt.Logger(trt.Logger.INFO)

FP32_LAYER_KW = ['layernorm', 'layer_norm', 'softmax', 'sigmoid',
                 'sin', 'cos', 'pow', 'sqrt', 'erf',
                 'reducesum', 'reducemean', 'reducemax', 'reducemin']
PLUGIN_KW = ['DeformableAggregation', 'deformable_aggregation']


def set_workspace(config, size):
    if hasattr(config, 'set_memory_pool_limit'):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, size)
    else:
        config.max_workspace_size = size


def force_precision(network):
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        name = layer.name.lower()
        if layer.type == trt.LayerType.MATRIX_MULTIPLY:
            layer.precision = trt.float16
            for j in range(layer.num_outputs):
                layer.set_output_type(j, trt.float16)
        elif any(kw in name for kw in PLUGIN_KW):
            layer.precision = trt.float16
            for j in range(layer.num_outputs):
                layer.set_output_type(j, trt.float16)
        elif any(kw in name for kw in FP32_LAYER_KW):
            layer.precision = trt.float32
            for j in range(layer.num_outputs):
                layer.set_output_type(j, trt.float32)


def build(onnx_path, engine_path, plugin_lib):
    if plugin_lib and os.path.exists(plugin_lib):
        ctypes.CDLL(plugin_lib, mode=ctypes.RTLD_GLOBAL)
    trt.init_libnvinfer_plugins(TRT_LOGGER, '')
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX parse failed")
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
    force_precision(network)
    set_workspace(config, 8 * 1024 * 1024 * 1024)
    if hasattr(builder, 'build_serialized_network'):
        serialized = builder.build_serialized_network(network, config)
    else:
        serialized = builder.build_cuda_engine(network, config).serialize()
    if serialized is None:
        raise RuntimeError("TRT build failed (returned None)")
    os.makedirs(os.path.dirname(os.path.abspath(engine_path)), exist_ok=True)
    with open(engine_path, 'wb') as f:
        f.write(serialized)
    print(f"Engine: {engine_path} ({os.path.getsize(engine_path) / 1024 / 1024:.1f} MB)")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--onnx', required=True)
    p.add_argument('--engine', required=True)
    p.add_argument('--plugin', default='')
    args = p.parse_args()
    build(args.onnx, args.engine, args.plugin)
