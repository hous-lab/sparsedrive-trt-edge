#!/usr/bin/env python
"""Evaluate both PT baseline and TRT hybrid on a subset, with NDS + mAP only."""
import sys, os, types, numpy

# numpy._core compatibility shim
if not hasattr(numpy, '_core'):
    nc = types.ModuleType('numpy._core')
    nc.__path__ = []
    sys.modules['numpy._core'] = nc
    for sub in ['multiarray', 'umath', 'numerictypes']:
        old = getattr(numpy.core, sub, None)
        if old:
            setattr(nc, sub, old)
            sys.modules[f'numpy._core.{sub}'] = old

import argparse, copy, json, ctypes, time
import torch
import tensorrt as trt

# Training repo root (contains projects/, ckpt/, data/). Override with
# SPARSEDRIVE_PROJECT_ROOT when the deploy repo is checked out elsewhere.
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# Training repo = two levels above this repo (dev layout). Override with
# SPARSEDRIVE_PROJECT_ROOT when cloned as a sibling of SparseDriveV2.
ROOT = os.environ.get('SPARSEDRIVE_PROJECT_ROOT',
                      os.path.abspath(os.path.join(REPO, '..', '..')))
CKPT = os.environ.get('SD_CKPT', os.path.join(ROOT, 'ckpt/sparsedrive_small_b2d_stage2.pth'))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import projects.mmdet3d_plugin
from mmcv import Config
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet.models import build_detector
from mmdet.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmcv.parallel import MMDataParallel, DataContainer
from projects.mmdet3d_plugin.ops.deformable_aggregation import deformable_format

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
FEAT_SHAPES = [(96,176),(48,88),(24,44),(12,22)]

class TrtEngine:
    def __init__(self, ep, pl=None):
        if pl and os.path.exists(pl): ctypes.CDLL(pl, mode=ctypes.RTLD_GLOBAL)
        trt.init_libnvinfer_plugins(TRT_LOGGER,'')
        with open(ep,'rb') as f: self.eng = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(f.read())
        self.ctx = self.eng.create_execution_context()

    def infer(self, d):
        s = torch.cuda.current_stream()
        for i in range(self.eng.num_io_tensors):
            n = self.eng.get_tensor_name(i)
            if self.eng.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
                t = d[n].contiguous()
                self.ctx.set_input_shape(n, tuple(t.shape))
                self.ctx.set_tensor_address(n, t.data_ptr())
        outs = {}
        for i in range(self.eng.num_io_tensors):
            n = self.eng.get_tensor_name(i)
            if self.eng.get_tensor_mode(n) != trt.TensorIOMode.INPUT:
                sh = tuple(self.ctx.get_tensor_shape(n))
                t = torch.empty(sh, device='cuda')
                self.ctx.set_tensor_address(n, t.data_ptr())
                outs[n] = t
        self.ctx.execute_async_v3(s.cuda_stream)
        s.synchronize()
        return outs


def run_eval(cfg_path, ckpt_path, num_samples, out_dir, bb_eng=None, head_eng=None, plugin=None):
    os.makedirs(out_dir, exist_ok=True)
    cfg = Config.fromfile(cfg_path)
    cfg.data.test.test_mode = True

    ds = build_dataset(cfg.data.test)
    ds.work_dir = out_dir
    if num_samples: ds.data_infos = ds.data_infos[:num_samples]
    print(f'Dataset: {len(ds)} samples')

    dl = build_dataloader(ds, 1, 2, dist=False, shuffle=False)

    # Build model
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, ckpt_path, map_location='cpu')
    if cfg.get('fp16'): wrap_fp16_model(model)
    model.cuda().eval()

    # Determine config name
    if bb_eng:
        config_name = 'trt_hybrid'
        bb_trt = TrtEngine(bb_eng, plugin)
        hd_trt = TrtEngine(head_eng, plugin) if head_eng else None

        def trt_extract_feat(img, metas=None):
            bs, nc = img.shape[0], img.shape[1]
            bo = bb_trt.infer({'img': img.flatten(end_dim=1)})
            return deformable_format([bo[f'feat_{i}'].reshape(bs,nc,256,*FEAT_SHAPES[i]) for i in range(4)]), None

        def trt_det_forward(self, fm, metas, depth_prob):
            ff = fm[0]
            off = 0
            fl = []
            for i in range(4):
                h,w=FEAT_SHAPES[i]
                n=h*w
                t = ff[:,:,off:off+n,:].transpose(-1,-2).reshape(1,6,256,h,w).squeeze(0)
                fl.append(t)
                off+=n
            pj = metas.get('projection_mat',torch.eye(4,device=ff.device).unsqueeze(0)).squeeze(0).float()
            iw = metas.get('image_wh',torch.tensor([[704,384]],device=ff.device).float()).squeeze(0)
            hi = {f'feat_{i}':fl[i] for i in range(4)}; hi.update({'projection_mat':pj,'image_wh':iw})
            ho = hd_trt.infer(hi)
            self.instance_bank.mask = torch.ones(1, dtype=torch.bool, device=ff.device)
            return {'classification':[ho['det_cls']]*6, 'prediction':[ho['det_reg']]*6,
                    'quality':[ho['det_quality']]*6,
                    'instance_feature': ho.get('x.607', torch.zeros(1,900,256,device=ff.device)),
                    'anchor_embed': torch.zeros(1,900,256,device=ff.device),
                    'instance_id': -torch.ones(1,900,device=ff.device).long()}

        inner = model
        inner.extract_feat = trt_extract_feat
        inner.head.det_head.forward = trt_det_forward.__get__(inner.head.det_head, type(inner.head.det_head))

        mp = MMDataParallel(inner, device_ids=[0]); mp.eval()
        from mmdet.apis import single_gpu_test
        outputs = single_gpu_test(mp, dl)
    else:
        config_name = 'pytorch_baseline'
        mp = MMDataParallel(model, device_ids=[0]); mp.eval()
        from mmdet.apis import single_gpu_test
        outputs = single_gpu_test(mp, dl)

    # Evaluate results - only detection metrics, skip planning
    eval_cfg = copy.deepcopy(cfg.evaluation.get('eval_mode', {}))
    eval_cfg['with_planning'] = False  # skip planning eval
    results_dict = ds.evaluate(outputs, eval_cfg, jsonfile_prefix=os.path.join(out_dir, config_name))

    print(f'\n=== {config_name} ===')
    for k, v in results_dict.items():
        print(f'  {k}: {v:.4f}' if isinstance(v, float) else f'  {k}: {v}')

    out_path = os.path.join(out_dir, f'{config_name}_results.json')
    with open(out_path, 'w') as f:
        json.dump({k: float(v) if hasattr(v, '__float__') else str(v) for k, v in results_dict.items()}, f, indent=2)
    print(f'Results saved: {out_path}')
    return results_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bb-engine')
    parser.add_argument('--head-engine')
    parser.add_argument('--plugin')
    parser.add_argument('--num-samples', type=int, default=50)
    parser.add_argument('--output-dir', default='work_dirs/trt_eval')
    args = parser.parse_args()

    cfg_path = os.environ.get(
        'SD_CONFIG',
        os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'configs/sparsedrive_stage2.py')))
    ckpt_path = CKPT
    out_dir = os.path.join(ROOT, args.output_dir)

    # 1. PT Baseline
    pt_results = run_eval(cfg_path, ckpt_path, args.num_samples, out_dir)

    # 2. TRT Hybrid (BB INT8 or FP16 + Head TRT, Motion PT)
    trt_results = run_eval(cfg_path, ckpt_path, args.num_samples, out_dir,
                          bb_eng=args.bb_engine, head_eng=args.head_engine, plugin=args.plugin)

    # Delta
    print('\n' + '='*60)
    print('Delta (TRT - PT)')
    print('='*60)
    for k in ['NDS', 'mAP']:
        pt_v = float(pt_results.get(k, 0))
        trt_v = float(trt_results.get(k, 0))
        d = trt_v - pt_v
        pct = d / pt_v * 100 if pt_v else 0
        print(f'  {k}: {pt_v:.4f} -> {trt_v:.4f} ({d:+.4f}, {pct:+.2f}%)')
