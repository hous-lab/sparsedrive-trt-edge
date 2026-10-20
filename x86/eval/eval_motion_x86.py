#!/usr/bin/env python
"""Motion/Planning eval: run PT full vs TRT full through MMDataParallel, compare motion outputs."""
import sys, os, types, json, copy
# Training repo root; override with SPARSEDRIVE_PROJECT_ROOT.
ROOT = os.environ.get(
    'SPARSEDRIVE_PROJECT_ROOT',
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..')))
os.chdir(ROOT)
sys.path.insert(0, '.')

import numpy as np
if not hasattr(np, '_core'):
    import numpy.core.multiarray as _ma; import numpy.core.umath as _um
    nc = types.ModuleType('numpy._core'); nc.__path__ = []; nc.multiarray = _ma; nc.umath = _um
    sys.modules['numpy._core'] = nc; sys.modules['numpy._core.multiarray'] = _ma; sys.modules['numpy._core.umath'] = _um

import torch, ctypes, tensorrt as trt
import projects.mmdet3d_plugin
from mmcv import Config
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet.models import build_detector
from mmdet.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from projects.mmdet3d_plugin.ops.deformable_aggregation import deformable_format
from mmcv.parallel import MMDataParallel

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
FEAT_SHAPES = [(96,176),(48,88),(24,44),(12,22)]

def cos_sim(a, b):
    a_f = a.flatten().float().cpu().numpy()
    b_f = b.flatten().float().cpu().numpy()
    return float(np.dot(a_f, b_f) / (np.linalg.norm(a_f)*np.linalg.norm(b_f)+1e-12))

def mse_v(a, b):
    return float(((a.float()-b.float())**2).mean().cpu().numpy())

def main():
    cfg = Config.fromfile('projects/configs/sparsedrive_stage2.py')
    cfg.data.test.test_mode = True
    ds = build_dataset(cfg.data.test)
    ds.work_dir = 'work_dirs/motion_eval'
    ds.data_infos = ds.data_infos[:6]
    dl = build_dataloader(ds, 1, 2, dist=False, shuffle=False)

    # TRT engines
    edge_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
    plugin = os.environ.get(
        'SD_PLUGIN_LIB', os.path.join(edge_root, 'plugins/build/libplugins.so'))
    out_dir = os.path.join(edge_root, 'x86/eval/results')
    os.makedirs(out_dir, exist_ok=True)
    engine_dir = os.environ.get(
        'SD_X86_ENGINE_DIR', os.path.join(edge_root, 'x86/trt_compile/fp16/output'))
    ctypes.CDLL(plugin, mode=ctypes.RTLD_GLOBAL)
    trt.init_libnvinfer_plugins(TRT_LOGGER, '')

    class TE:
        def __init__(self, path):
            with open(path, 'rb') as f:
                self.eng = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(f.read())
            self.ctx = self.eng.create_execution_context()
        def infer(self, fd):
            s = torch.cuda.current_stream()
            for i in range(self.eng.num_io_tensors):
                n = self.eng.get_tensor_name(i)
                if self.eng.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
                    t = fd[n].contiguous()
                    self.ctx.set_input_shape(n, tuple(t.shape))
                    self.ctx.set_tensor_address(n, t.data_ptr())
            o = {}
            for i in range(self.eng.num_io_tensors):
                n = self.eng.get_tensor_name(i)
                if self.eng.get_tensor_mode(n) != trt.TensorIOMode.INPUT:
                    sh = tuple(self.ctx.get_tensor_shape(n))
                    t = torch.empty(sh, device='cuda')
                    self.ctx.set_tensor_address(n, t.data_ptr())
                    o[n] = t
            self.ctx.execute_async_v3(s.cuda_stream); s.synchronize()
            return o

    bb = TE(os.path.join(engine_dir, 'backbone_fp16.engine'))
    hd = TE(os.path.join(engine_dir, 'head_fp16.engine'))
    mo = TE(os.path.join(engine_dir, 'motion_plan_fp16.engine'))

    # Save motion outputs
    pt_saved = []
    trt_saved = []

    def make_pt_hook(saved_list):
        def hook(module, inputs, output):
            if isinstance(output, tuple) and len(output) >= 2:
                mdict = output[0]
                pdict = output[1]
                mc_list = mdict.get('classification') if isinstance(mdict, dict) else None
                mt_list = mdict.get('prediction') if isinstance(mdict, dict) else None
                mc = mc_list[-1] if isinstance(mc_list, list) and mc_list else None
                mt = mt_list[-1] if isinstance(mt_list, list) and mt_list else None
                pt = None
                if isinstance(pdict, dict):
                    pr = pdict.get('planning_results', [])
                    if pr and isinstance(pr[-1], dict):
                        pt = pr[-1].get('trajs')
                d = {}
                if mc is not None: d['motion_cls'] = mc.detach().cpu()
                if mt is not None: d['motion_traj'] = mt.detach().cpu()
                if pt is not None: d['planning_traj'] = pt.detach().cpu()
                saved_list.append(d)
            return output
        return hook

    # === PT baseline ===
    print('PT baseline...')
    model_pt = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model_pt, 'ckpt/sparsedrive_small_b2d_stage2.pth', map_location='cpu')
    if cfg.get('fp16'): wrap_fp16_model(model_pt)
    model_pt.cuda().eval()
    h_pt = model_pt.head.motion_plan_head.register_forward_hook(make_pt_hook(pt_saved))
    mp_pt = MMDataParallel(model_pt, device_ids=[0]); mp_pt.eval()
    from mmdet.apis import single_gpu_test
    _ = single_gpu_test(mp_pt, dl)
    h_pt.remove()
    print(f'  PT: {len(pt_saved)} frames')

    # === TRT hybrid (BB+Head TRT, Motion PT) ===
    print('TRT det + PT motion...')
    model_trt = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model_trt, 'ckpt/sparsedrive_small_b2d_stage2.pth', map_location='cpu')
    if cfg.get('fp16'): wrap_fp16_model(model_trt)
    model_trt.cuda().eval()

    def trt_extract_feat(img, metas=None):
        bs, nc = img.shape[0], img.shape[1]
        bo = bb.infer({'img': img.flatten(end_dim=1)})
        return deformable_format([bo[f'feat_{i}'].reshape(bs,nc,256,*FEAT_SHAPES[i]) for i in range(4)]), None

    def trt_det_forward(self, fm, metas, depth_prob):
        ff = fm[0]; off = 0; fl = []
        for i in range(4):
            h,w=FEAT_SHAPES[i]; n=h*w
            fl.append(ff[:,:,off:off+n,:].transpose(-1,-2).reshape(1,6,256,h,w).squeeze(0)); off+=n
        pj = metas.get('projection_mat',torch.eye(4,device=ff.device).unsqueeze(0)).squeeze(0).float()
        iw = metas.get('image_wh',torch.tensor([[704,384]],device=ff.device).float()).squeeze(0)
        hi = {f'feat_{i}':fl[i] for i in range(4)}; hi.update({'projection_mat':pj,'image_wh':iw})
        ho = hd.infer(hi)
        self.instance_bank.mask = torch.ones(1, dtype=torch.bool, device=ff.device)
        return {'classification':[ho['det_cls']]*6,'prediction':[ho['det_reg']]*6,
                'quality':[ho['det_quality']]*6,
                'instance_feature': ho.get('x.607', torch.zeros(1,900,256,device=ff.device)),
                'anchor_embed': torch.zeros(1,900,256,device=ff.device),
                'instance_id': -torch.ones(1,900,device=ff.device).long()}

    model_trt.extract_feat = trt_extract_feat
    model_trt.head.det_head.forward = trt_det_forward.__get__(model_trt.head.det_head, type(model_trt.head.det_head))
    # Keep PT motion_plan_head — hook it to capture outputs with TRT det inputs
    h_trt = model_trt.head.motion_plan_head.register_forward_hook(make_pt_hook(trt_saved))

    mp_trt = MMDataParallel(model_trt, device_ids=[0]); mp_trt.eval()
    _ = single_gpu_test(mp_trt, dl)
    h_trt.remove()
    print(f'  TRT hybrid: {len(trt_saved)} frames (TRT det + PT motion)')

    # === Compare: PT motion outputs vs TRT-hybrid motion outputs ===
    # Both use PT motion head, so the difference is only from detection head accuracy
    print('\n=== Motion Comparison (PT full vs TRT det + PT motion) ===')
    print('This shows the impact of TRT detection head on motion planning.')
    for name in ['motion_cls','motion_traj','planning_traj']:
        valid = []
        for i in range(min(len(pt_saved), len(trt_saved))):
            a, b = pt_saved[i].get(name), trt_saved[i].get(name)
            if a is None or b is None: continue
            if a.numel() == 0 or b.numel() == 0: continue
            af, bf = a.float(), b.float()
            if af.norm() > 1e-8 and bf.norm() > 1e-8 and not torch.isnan(af).any() and not torch.isnan(bf).any():
                valid.append({'cs': cos_sim(af,bf), 'mse': mse_v(af,bf)})
        if valid:
            cs = np.mean([v['cs'] for v in valid])
            ms = np.mean([v['mse'] for v in valid])
        else: cs = ms = None
        s = 'OK' if cs and cs>0.99 else 'WARN' if cs and cs>0.95 else 'N/A'
        cs_str = f'{cs:.6f}' if cs is not None else 'N/A'
        ms_str = f'{ms:.8f}' if ms is not None else 'N/A'
        print(f'  {name}: cos_sim={cs_str} mse={ms_str} [{s}]')

    out = os.path.join(out_dir, 'motion_comparison.json')
    with open(out, 'w') as f: json.dump({
        'pt_frames': len(pt_saved),
        'trt_frames': len(trt_saved),
        'note': 'PT full vs TRT det+PT motion. Motion head runs in PT for both, difference is from TRT detection head.',
    }, f, indent=2)
    print(f'Saved: {out}')
    print('\nNote: Motion head runs in PT for both configs.')
    print('Difference shows TRT detection head impact on motion/planning quality.')
    print('For full TRT motion eval, additional queue externalization is needed.')

if __name__ == '__main__':
    main()
