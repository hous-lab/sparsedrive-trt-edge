#!/usr/bin/env python
"""Generate the x86 engine-cascade reference (data/x86_reference.npz).

Runs the x86 FP16 engine cascade (backbone -> head -> map -> motion) on the
demo inputs and saves all I/O as the cross-platform reference for Orin
accuracy gates. Engine-only: needs torch + TRT + plugin, no training repo.

Usage: python tools/gen_x86_reference.py
Env: SD_PLUGIN_LIB, SD_X86_ENGINE_DIR, SD_DEMO_NPZ, SD_REF_NPZ
"""
import os, ctypes, json
import numpy as np
import torch
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PLUGIN = os.environ.get('SD_PLUGIN_LIB', os.path.join(REPO, 'plugins/build/libplugins.so'))
ENGINE_DIR = os.environ.get('SD_X86_ENGINE_DIR', os.path.join(REPO, 'x86/trt_compile/fp16/output'))
DEMO_NPZ = os.environ.get('SD_DEMO_NPZ', os.path.join(REPO, 'data/demo_data/sample0.npz'))
REF_NPZ = os.environ.get('SD_REF_NPZ', os.path.join(REPO, 'data/x86_reference.npz'))
FEAT_SHAPES = [(96, 176), (48, 88), (24, 44), (12, 22)]

import sys
sys.path.insert(0, os.path.join(REPO, 'x86/eval'))
from eval_pytorch_vs_trt import TrtEngine


def flatten_feats(feat_i_list):
    """Per-level [6,256,h,w] -> feat_flat [1,22440,256] + spat [4,2] i32 + lvl [5] i32."""
    flat = torch.cat([f.reshape(f.shape[0], f.shape[1], -1).permute(0, 2, 1).reshape(-1, f.shape[1])
                      for f in feat_i_list], dim=0)
    flat = flat.unsqueeze(0)
    sizes = [h * w for h, w in FEAT_SHAPES]
    spat = torch.tensor([[h, w] for h, w in FEAT_SHAPES], dtype=torch.int32)
    lvl = torch.tensor(np.cumsum([0] + sizes), dtype=torch.int32)
    return flat.contiguous(), spat.cuda(), lvl.cuda()




def main():
    demo = np.load(DEMO_NPZ)
    bb = TrtEngine(os.path.join(ENGINE_DIR, 'backbone_fp16.engine'))
    hd = TrtEngine(os.path.join(ENGINE_DIR, 'head_fp16.engine'), PLUGIN)
    mp = TrtEngine(os.path.join(ENGINE_DIR, 'map_fp16.engine'), PLUGIN)
    mo = TrtEngine(os.path.join(ENGINE_DIR, 'motion_plan_fp16.engine'), PLUGIN)
    print('[ref] 4 engines loaded')
    img = torch.from_numpy(np.ascontiguousarray(demo['img'])).cuda().float()
    bb_out = bb.infer({'img': img})
    feats = [bb_out['feat_%d' % i] for i in range(4)]
    flat, spat, lvl = flatten_feats(feats)
    pm = torch.from_numpy(np.ascontiguousarray(demo['projection_mat'])).cuda().float()
    iw = torch.from_numpy(np.ascontiguousarray(demo['image_wh'])).cuda().float()
    hd_out = hd.infer({
        'feat_0': feats[0], 'feat_1': feats[1],
        'feat_2': feats[2], 'feat_3': feats[3],
        'projection_mat': pm, 'image_wh': iw})
    print('[ref] head done:', {k: tuple(v.shape) for k, v in hd_out.items()})
    map_out = mp.infer({
        'feat_0': feats[0], 'feat_1': feats[1],
        'feat_2': feats[2], 'feat_3': feats[3],
        'projection_mat': pm, 'image_wh': iw})
    map_out['map_reg'] = torch.zeros(1, 100, 2, device='cuda')
    print('[ref] map done:', {k: tuple(v.shape) for k, v in map_out.items()})
    mo_out = mo.infer({
        'feat_flat': flat, 'spat_shape': spat, 'lvl_start': lvl,
        'det_cls': hd_out['det_cls'], 'det_reg': hd_out['det_reg'],
        'det_quality': hd_out['det_quality'],
        'det_instance_feat': hd_out['det_instance_feat'],
        'det_anchor_embed': demo_np(demo, 'anchor_embed').reshape(1, 900, 256),
        'map_cls': map_out['map_cls'],
        'map_reg': torch.zeros(1, 100, 2, device='cuda'),
        'map_instance_feat': map_out['map_instance_feature'],
        'map_anchor_embed': map_out['map_anchor_embed'],
        'projection_mat': pm, 'image_wh': iw,
        'timestamp': demo_np(demo, 'timestamp'),
        'ego_status': demo_np(demo, 'ego_status'),
        'tp_near': demo_np(demo, 'tp_near'),
        'gt_ego_fut_cmd': demo_np(demo, 'gt_ego_fut_cmd'),
        'mask': demo_np(demo, 'mask'),
        'queue_temp_feat': demo_np(demo, 'queue_temp_feat'),
        'queue_temp_anchor': demo_np(demo, 'queue_temp_anchor'),
        'period': demo_np(demo, 'period')})
    print('[ref] motion done:', {k: tuple(v.shape) for k, v in mo_out.items()})
    ref = {
        'img': demo['img'],
        'projection_mat': demo['projection_mat'],
        'image_wh': demo['image_wh'],
        'timestamp': demo['timestamp'],
        'ego_status': demo['ego_status'],
        'tp_near': demo['tp_near'],
        'gt_ego_fut_cmd': demo['gt_ego_fut_cmd'],
        'mask': demo['mask'],
    }
    for i in range(4):
        ref['bb_feat_%d' % i] = feats[i]
    ref['hd_det_cls'] = hd_out['det_cls']
    ref['hd_det_reg'] = hd_out['det_reg']
    ref['hd_det_quality'] = hd_out['det_quality']
    ref['hd_instance_feature'] = hd_out['det_instance_feat']
    ref['map_cls'] = map_out['map_cls']
    ref['map_reg'] = map_out['map_reg']
    ref['map_instance_feat'] = map_out['map_instance_feature']
    ref['map_anchor_embed'] = map_out['map_anchor_embed']
    ref['queue_temp_feat'] = demo['queue_temp_feat']
    ref['queue_temp_anchor'] = demo['queue_temp_anchor']
    ref['period'] = demo['period']
    ref['det_anchor_embed'] = demo['anchor_embed']
    for k, v in mo_out.items():
        ref['mp_' + k] = v
    os.makedirs(os.path.dirname(REF_NPZ), exist_ok=True)
    save_npz(REF_NPZ, ref)


def demo_np(demo, key):
    a = np.ascontiguousarray(demo[key])
    if a.dtype == np.float64 or a.dtype == np.float16:
        a = a.astype(np.float32)
    return torch.from_numpy(a).cuda()


def save_npz(path, arrays):
    d = {}
    for k, v in arrays.items():
        d[k] = v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)
    np.savez_compressed(path, **d)
    print('[ref] saved %s (%d arrays)' % (path, len(d)))


if __name__ == '__main__':
    main()
