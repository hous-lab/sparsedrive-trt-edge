#!/usr/bin/env python
"""SparseDriveV2 NDS: TRT backbone + TRT det_head + PT motion/planning."""
import os, sys, copy, argparse, time, ctypes, json, numpy as np, torch
import tensorrt as trt
PROJECT = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')
sys.path.insert(0, PROJECT)
from mmcv import Config; from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet.models import build_detector; from mmdet.datasets import build_dataset
from mmcv.parallel import MMDataParallel, DataContainer
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from projects.mmdet3d_plugin.ops.deformable_aggregation import deformable_format
import projects.mmdet3d_plugin
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
FEAT_SHAPES = [(96,176),(48,88),(24,44),(12,22)]

class TrtEngine:
    def __init__(self, ep, pl=None):
        if pl and os.path.exists(pl): ctypes.CDLL(pl, mode=ctypes.RTLD_GLOBAL)
        trt.init_libnvinfer_plugins(TRT_LOGGER,'')
        with open(ep,'rb') as f: self.eng = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(f.read())
        self.ctx = self.eng.create_execution_context()
        self.ins = []; self.outs = []
        for i in range(self.eng.num_io_tensors):
            n = self.eng.get_tensor_name(i)
            if self.eng.get_tensor_mode(n) == trt.TensorIOMode.INPUT: self.ins.append(n)
            else: self.outs.append(n)
    def infer(self, d):
        for n in self.ins: t=d[n].contiguous();self.ctx.set_input_shape(n,tuple(t.shape));self.ctx.set_tensor_address(n,t.data_ptr())
        o={}
        for n in self.outs: s=tuple(self.ctx.get_tensor_shape(n));t=torch.empty(s,device='cuda');self.ctx.set_tensor_address(n,t.data_ptr());o[n]=t
        self.ctx.execute_async_v3(0);torch.cuda.current_stream().synchronize();return o

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='projects/configs/sparsedrive_stage2.py')
    p.add_argument('--checkpoint', default='ckpt/sparsedrive_small_b2d_stage2.pth')
    p.add_argument('--bb-engine'); p.add_argument('--head-engine')
    p.add_argument('--plugin'); p.add_argument('--num-samples', type=int, default=None)
    p.add_argument('--output-dir', default='work_dirs/trt_eval'); p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    def rp(x): return os.path.join(PROJECT,'sparsedrive_deploy_pipeline',x) if x and not os.path.isabs(x) else x
    bb_eng = rp(args.bb_engine); hd_eng = rp(args.head_engine); plugin = rp(args.plugin)
    cfg_path = os.path.join(PROJECT, args.config); ckpt_path = os.path.join(PROJECT, args.checkpoint)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load config & dataset
    cfg = Config.fromfile(cfg_path); cfg.data.test.test_mode = True
    ds = build_dataset(cfg.data.test); ds.work_dir = args.output_dir
    if args.num_samples and hasattr(ds,'data_infos'): ds.data_infos = ds.data_infos[:args.num_samples]
    dl = build_dataloader(ds, 1, 2, dist=False, shuffle=False)
    print(f"Dataset: {len(ds)} samples")

    # ============= Build TRT engines =============
    bb_trt = TrtEngine(bb_eng, plugin)
    hd_trt = TrtEngine(hd_eng, plugin)
    
    # Load model once for PT motion/planning
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, ckpt_path, map_location='cpu')
    if cfg.get('fp16'): wrap_fp16_model(model)
    model.to(args.device); model.eval()

    def trt_extract_feat(img, metas=None):
        bs, nc = img.shape[0], img.shape[1]
        bo = bb_trt.infer({'img': img.flatten(end_dim=1)})
        return deformable_format([bo[f'feat_{i}'].reshape(bs,nc,256,*FEAT_SHAPES[i]) for i in range(4)]), None

    def trt_det_forward(self, fm, metas, depth_prob):
        ff = fm[0]; off = 0; fl = []
        for i in range(4): h,w=FEAT_SHAPES[i]; n=h*w; fl.append(ff[:,:,off:off+n,:].transpose(-1,-2).reshape(1,6,256,h,w).squeeze(0)); off+=n
        pj = metas.get('projection_mat',torch.eye(4,device=ff.device).unsqueeze(0)).squeeze(0).float()
        iw = metas.get('image_wh',torch.tensor([[704,384]],device=ff.device).float()).squeeze(0)
        hi = {f'feat_{i}':fl[i] for i in range(4)}; hi.update({'projection_mat':pj,'image_wh':iw})
        ho = hd_trt.infer(hi)
        # Initialize InstanceBank mask for motion head
        self.instance_bank.mask = torch.ones(1, dtype=torch.bool, device=ff.device)
        return {'classification':[ho['det_cls']]*6, 'prediction':[ho['det_reg']]*6,
                'quality':[ho['det_quality']]*6,
                'instance_feature': ho.get('x.607', torch.zeros(1,900,256,device=ff.device)),
                'anchor_embed': torch.zeros(1,900,256,device=ff.device),
                'instance_id': -torch.ones(1,900,device=ff.device).long()}

    # ============= Run evaluation =============
    from mmdet.apis import single_gpu_test
    results = {}

    # 1. PyTorch baseline
    print("\n1/2 PyTorch baseline")
    model2 = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model2, ckpt_path, map_location='cpu')
    if cfg.get('fp16'): wrap_fp16_model(model2)
    model2.to(args.device); model2.eval()
    ds2 = build_dataset(cfg.data.test); ds2.work_dir = args.output_dir
    if args.num_samples and hasattr(ds2,'data_infos'): ds2.data_infos = ds2.data_infos[:args.num_samples]
    dl2 = build_dataloader(ds2, 1, 2, dist=False, shuffle=False)
    mp2 = MMDataParallel(model2, device_ids=[0]); mp2.eval()
    out_pt = single_gpu_test(mp2, dl2)
    em = copy.deepcopy(cfg.evaluation.get('eval_mode',{'with_det':True}))
    r_pt = ds2.evaluate(out_pt, em, jsonfile_prefix=os.path.join(args.output_dir,'pytorch'))
    results['pytorch'] = {k:float(v) if hasattr(v,'__float__') else str(v) for k,v in r_pt.items()}
    print(f"  NDS={r_pt.get('NDS',0):.4f}  mAP={r_pt.get('mAP',0):.4f}")

    # 2. TRT + PT hybrid
    print("\n2/2 TRT det + PT motion/planning")
    # Build model with TRT backbone + det_head
    inner = model
    inner.extract_feat = trt_extract_feat
    inner.head.det_head.forward = trt_det_forward.__get__(inner.head.det_head, type(inner.head.det_head))
    # Keep original motion_plan_head and post_process

    ds3 = build_dataset(cfg.data.test); ds3.work_dir = args.output_dir
    if args.num_samples and hasattr(ds3,'data_infos'): ds3.data_infos = ds3.data_infos[:args.num_samples]
    dl3 = build_dataloader(ds3, 1, 2, dist=False, shuffle=False)
    mp3 = MMDataParallel(inner, device_ids=[0]); mp3.eval()
    out_trt = single_gpu_test(mp3, dl3)
    r_trt = ds3.evaluate(out_trt, em, jsonfile_prefix=os.path.join(args.output_dir,'trt_hybrid'))
    results['trt_hybrid'] = {k:float(v) if hasattr(v,'__float__') else str(v) for k,v in r_trt.items()}
    print(f"  NDS={r_trt.get('NDS',0):.4f}  mAP={r_trt.get('mAP',0):.4f}")

    # Delta
    for m in ['NDS','mAP']:
        pt_v = float(r_pt.get(f'bbox_NuScenes/{m}',r_pt.get(m,0)))
        tt_v = float(r_trt.get(f'bbox_NuScenes/{m}',r_trt.get(m,0)))
        d = tt_v - pt_v
        print(f"  Δ{m}: {pt_v:.4f} → {tt_v:.4f} ({d:+.4f}, {d/pt_v*100:+.2f}%)")
        results[f'delta_{m}'] = d

    with open(os.path.join(args.output_dir, 'comparison.json'),'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {os.path.join(args.output_dir,'comparison.json')}")

if __name__ == '__main__': main()
