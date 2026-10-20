#!/usr/bin/env python
"""Build map + motion_plan FP16 engines on Orin (sm_87).

The Orin needs the re-exported engines (motion_plan_v2 with real planning
outputs, and the map head) — the old motion engine cannot plan. Sync the
ONNX files to Orin first (they are in step2_onnx_export/output/).

Run on Orin:
  python3 build_map_motion_fp16_orin.py
"""
import os, sys, argparse, ctypes, tensorrt as trt

DEPLOY_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, DEPLOY_ROOT)

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
FP32_KW = ['layernorm', 'layer_norm', 'softmax', 'sigmoid', 'sin', 'cos', 'pow', 'sqrt', 'erf',
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
        elif any(kw in name for kw in FP32_KW):
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
            raise RuntimeError("ONNX parse failed: " + onnx_path)
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
        raise RuntimeError("TRT build failed: " + engine_path)
    os.makedirs(os.path.dirname(os.path.abspath(engine_path)), exist_ok=True)
    with open(engine_path, 'wb') as f:
        f.write(serialized)
    print(f"Engine: {engine_path} ({os.path.getsize(engine_path) / 1024 / 1024:.1f} MB)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plugin', default=os.path.join(DEPLOY_ROOT, 'arm64/plugins_build/build/libplugins_orin.so'))
    args = p.parse_args()

    out_dir = os.path.join(DEPLOY_ROOT, 'arm64/trt_compile/fp16/output')
    plugin = os.path.join(DEPLOY_ROOT, args.plugin)

    # motion plan v2 (needs plugin: DeformableAggregation)
    build(
        os.path.join(DEPLOY_ROOT, 'onnx_export/output/sparsedrive_motion_plan_sim.onnx'),
        os.path.join(out_dir, 'motion_plan_fp16.engine'),
        plugin,
    )
    # map head (needs plugin: DeformableAggregation)
    build(
        os.path.join(DEPLOY_ROOT, 'onnx_export/output/sparsedrive_map_sim.onnx'),
        os.path.join(out_dir, 'map_fp16.engine'),
        plugin,
    )
    print("Done.")


if __name__ == '__main__':
    main()
