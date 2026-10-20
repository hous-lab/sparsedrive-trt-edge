"""TRT inference wrapper for TRT 10.x API (I/O tensor based)."""
import os, sys, ctypes, torch, numpy as np, tensorrt as trt, json, time

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class TrtEngine:
    def __init__(self, engine_path, plugin_lib=None):
        if plugin_lib and os.path.exists(plugin_lib):
            ctypes.CDLL(plugin_lib, mode=ctypes.RTLD_GLOBAL)
        trt.init_libnvinfer_plugins(TRT_LOGGER, '')

        runtime = trt.Runtime(TRT_LOGGER)
        with open(engine_path, 'rb') as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.input_names = []
        self.output_names = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

    def infer(self, feed_dict):
        for name in self.input_names:
            t = feed_dict[name].contiguous()
            self.context.set_input_shape(name, tuple(t.shape))
            self.context.set_tensor_address(name, t.data_ptr())

        output_tensors = {}
        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            vol = int(np.prod(shape))
            t = torch.empty(vol, dtype=torch.float32, device='cuda').reshape(shape)
            self.context.set_tensor_address(name, t.data_ptr())
            output_tensors[name] = t

        self.context.execute_async_v3(0)
        torch.cuda.current_stream().synchronize()
        return output_tensors


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--bb-engine', default=None)
    parser.add_argument('--head-engine', default=None)
    parser.add_argument('--plugin', default=None)
    parser.add_argument('--num-samples', type=int, default=50)
    args = parser.parse_args()

    # Simple latency benchmark only
    plugin = args.plugin
    bb_engine = TrtEngine(args.bb_engine, plugin) if args.bb_engine else None
    head_engine = TrtEngine(args.head_engine, plugin) if args.head_engine else None

    print("Engines loaded")

    # Warmup
    if bb_engine and head_engine:
        for _ in range(3):
            d = torch.randn(6, 3, 384, 704, device='cuda')
            bo = bb_engine.infer({'img': d})
            f0 = bo['feat_0'].view(6, 256, 96, 176)
            f1 = bo['feat_1'].view(6, 256, 48, 88)
            f2 = bo['feat_2'].view(6, 256, 24, 44)
            f3 = bo['feat_3'].view(6, 256, 12, 22)
            hi = {'feat_0':f0,'feat_1':f1,'feat_2':f2,'feat_3':f3,
                  'projection_mat':torch.eye(4,device='cuda').unsqueeze(0).expand(6,-1,-1),
                  'image_wh':torch.tensor([[704,384]],device='cuda').float().expand(6,-1)}
            head_engine.infer(hi)
        torch.cuda.synchronize()

        # Measure
        N = args.num_samples
        lats = []
        for i in range(N):
            d = torch.randn(6, 3, 384, 704, device='cuda')
            t0 = time.perf_counter()
            bo = bb_engine.infer({'img': d})
            f0 = bo['feat_0'].view(6, 256, 96, 176)
            f1 = bo['feat_1'].view(6, 256, 48, 88)
            f2 = bo['feat_2'].view(6, 256, 24, 44)
            f3 = bo['feat_3'].view(6, 256, 12, 22)
            hi = {'feat_0':f0,'feat_1':f1,'feat_2':f2,'feat_3':f3,
                  'projection_mat':torch.eye(4,device='cuda').unsqueeze(0).expand(6,-1,-1),
                  'image_wh':torch.tensor([[704,384]],device='cuda').float().expand(6,-1)}
            head_engine.infer(hi)
            torch.cuda.synchronize()
            lats.append((time.perf_counter() - t0) * 1000)

        lats = np.array(lats)
        print(f"\nResults (N={N}):")
        print(f"  Backbone + Head FP16: {lats.mean():.1f} ± {lats.std():.1f} ms")
        print(f"  FPS: {1000/lats.mean():.1f}")
        print(f"  Min: {lats.min():.1f} ms, Max: {lats.max():.1f} ms")
