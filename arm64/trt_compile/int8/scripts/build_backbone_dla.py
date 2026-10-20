import os
import sys
import argparse
import numpy as np
import tensorrt as trt

DEPLOY_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, DEPLOY_ROOT)

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
TRT_MAJOR = int(trt.__version__.split('.')[0])


class BackboneCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, calib_path, batch_size=1, input_shape=(6, 3, 384, 704)):
        super().__init__()
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.calib_data = np.load(calib_path)['imgs']
        self.num_frames = len(self.calib_data)
        self.batch_idx = 0

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
    if TRT_MAJOR >= 10:
        return builder.build_serialized_network(network, config)
    return builder.build_cuda_engine(network, config).serialize()


def set_workspace(config, size_bytes):
    if TRT_MAJOR >= 10:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, size_bytes)
    else:
        config.max_workspace_size = size_bytes


def force_layers_to_dla(builder, network, config, dla_core=0):
    """Try to assign every layer to DLA. Report layers that cannot run on DLA.

    TRT >= 10: setting config.DLA_core alone enables DLA (no BuilderFlag.DLA).
    """
    config.DLA_core = dla_core
    config.default_device_type = trt.DeviceType.DLA

    n_on_dla = 0
    n_gpu = 0
    dla_fail = []
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        if layer.type in (trt.LayerType.PLUGIN, trt.LayerType.PLUGIN_V2):
            continue
        try:
            ok = config.can_run_on_DLA(layer)
        except Exception as e:
            ok = False
        if ok:
            config.set_device_type(layer, trt.DeviceType.DLA)
            n_on_dla += 1
        else:
            config.set_device_type(layer, trt.DeviceType.GPU)
            dla_fail.append(layer.name)
            n_gpu += 1
    print(f"[DLA] {n_on_dla} layers on DLA, {n_gpu} on GPU")
    for n in dla_fail[:50]:
        print(f"  [DLA GPU fallback] {n}")


def build_backbone_dla(onnx_path, engine_path, calib_path, dla_core=0):
    trt.init_libnvinfer_plugins(TRT_LOGGER, '')

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    print(f"[build_backbone_dla] Parsing ONNX: {onnx_path}")
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
    # DLA 默认要求所有计算层都在 DLA 上；输入 Reformat/Shape 等层必须允许回退 GPU
    config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
    set_workspace(config, 8 * 1024 * 1024 * 1024)

    calibrator = BackboneCalibrator(calib_path)
    config.int8_calibrator = calibrator

    force_layers_to_dla(builder, network, config, dla_core=dla_core)

    print(f"[build_backbone_dla] Building engine on DLA core {dla_core} ...")
    serialized = build_engine(builder, network, config)
    with open(engine_path, 'wb') as f:
        f.write(serialized)

    print(f"[build_backbone_dla] Done: {engine_path}")
    print(f"[build_backbone_dla] Size: {os.path.getsize(engine_path) / 1024 / 1024:.1f} MB")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', default='onnx_export/output/sparsedrive_backbone.onnx')
    parser.add_argument('--engine', default='arm64/trt_compile/int8/output/backbone_dla0.engine')
    parser.add_argument('--calib', default='arm64/trt_compile/int8/calib_data/backbone_calib.npz')
    parser.add_argument('--dla-core', type=int, default=0)
    args = parser.parse_args()
    build_backbone_dla(args.onnx, args.engine, args.calib, args.dla_core)
