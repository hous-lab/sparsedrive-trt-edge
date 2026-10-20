import os
import sys
import ctypes
import argparse
import tensorrt as trt

DEPLOY_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, DEPLOY_ROOT)

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
TRT_VERSION = trt.__version__
TRT_MAJOR = int(TRT_VERSION.split('.')[0]) if TRT_VERSION else 10

FP32_LAYER_KEYWORDS = ['layernorm', 'layer_norm', 'softmax', 'sigmoid',
                        'sin', 'cos', 'pow', 'sqrt', 'erf',
                        'reducesum', 'reducemean', 'reducemax', 'reducemin']
PLUGIN_LAYER_KEYWORDS = ['DeformableAggregation', 'deformable_aggregation']


def build_engine(builder, network, config):
    if TRT_MAJOR >= 10:
        return builder.build_serialized_network(network, config)
    return builder.build_cuda_engine(network, config).serialize()


def set_workspace(config, size_bytes):
    if TRT_MAJOR >= 10:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, size_bytes)
    else:
        config.max_workspace_size = size_bytes


def force_layer_precision(network, config):
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        name = layer.name.lower()
        if layer.type == trt.LayerType.MATRIX_MULTIPLY:
            layer.precision = trt.float16
            for j in range(layer.num_outputs):
                layer.set_output_type(j, trt.float16)
        elif any(kw in name for kw in PLUGIN_LAYER_KEYWORDS):
            layer.precision = trt.float16
            for j in range(layer.num_outputs):
                layer.set_output_type(j, trt.float16)
        elif any(kw in name for kw in FP32_LAYER_KEYWORDS):
            layer.precision = trt.float32
            for j in range(layer.num_outputs):
                layer.set_output_type(j, trt.float32)


def build_head_fp16_orin(onnx_path, engine_path, plugin_lib=None):
    if plugin_lib and os.path.exists(plugin_lib):
        ctypes.CDLL(plugin_lib, mode=ctypes.RTLD_GLOBAL)

    trt.init_libnvinfer_plugins(TRT_LOGGER, '')

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    print(f"[Orin Head FP16] Parsing ONNX: {onnx_path}")
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
    set_workspace(config, 8 * 1024 * 1024 * 1024)

    force_layer_precision(network, config)

    serialized = build_engine(builder, network, config)
    with open(engine_path, 'wb') as f:
        f.write(serialized)
    print(f"[Orin Head FP16] Done: {engine_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', default='onnx_export/output/sparsedrive_head.onnx')
    parser.add_argument('--engine', default='arm64/trt_compile/fp16/output/head_fp16.engine')
    parser.add_argument('--plugin', default='arm64/plugins_build/build/libplugins_orin.so')
    args = parser.parse_args()
    build_head_fp16_orin(args.onnx, args.engine, args.plugin)
