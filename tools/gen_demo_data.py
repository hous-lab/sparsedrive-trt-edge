#!/usr/bin/env python
"""Generate demo inputs + motion-queue init from the first val sample.

Run inside the training-repo env (mmcv/mmdet). Produces
data/demo_data/sample0.npz with:
  img [6,3,384,704] f32, projection_mat [6,4,4], image_wh [6,2],
  timestamp [1], ego_status [10], tp_near [2], gt_ego_fut_cmd [6] i64,
  mask [900] f32 (-1), anchor_embed [1,900,256],
  queue_temp_feat [21,1,901,256], queue_temp_anchor [21,1,901,256],
  period [1,901] i64 = QL-1 (fresh-queue trick).

Usage: python tools/gen_demo_data.py
Env: SPARSEDRIVE_PROJECT_ROOT, SD_CKPT, SD_CONFIG
"""
import os, sys, types

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ROOT = os.environ.get('SPARSEDRIVE_PROJECT_ROOT',
                      os.path.abspath(os.path.join(REPO, '..', '..')))
CKPT = os.environ.get('SD_CKPT', os.path.join(ROOT, 'ckpt/sparsedrive_small_b2d_stage2.pth'))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import numpy as np
if not hasattr(np, '_core'):
    import numpy.core.multiarray as _ma
    import numpy.core.umath as _um
    _nc = types.ModuleType('numpy._core'); _nc.__path__ = []
    _nc.multiarray = _ma; _nc.umath = _um
    sys.modules['numpy._core'] = _nc
    sys.modules['numpy._core.multiarray'] = _ma
    sys.modules['numpy._core.umath'] = _um

import torch
from mmcv import Config
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet.models import build_detector
from mmdet.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader


def main():
    cfg = Config.fromfile(os.environ.get(
        'SD_CONFIG', os.path.join(REPO, 'configs/sparsedrive_stage2.py')))
    cfg.data.test.test_mode = True
    ds = build_dataset(cfg.data.test)
    ds.data_infos = ds.data_infos[:1]
    ds.work_dir = os.path.join(REPO, 'onnx_export/demo_data')
    dl = build_dataloader(ds, 1, 2, dist=False, shuffle=False)

    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, CKPT, map_location='cpu')
    if cfg.get('fp16'): wrap_fp16_model(model)
    model.cuda().eval()

    data = next(iter(dl))
    def tc(key, dtype=torch.float32):
        v = data[key]
        v = v.data[0] if hasattr(v, 'data') and not torch.is_tensor(v) else v
        return v.to('cuda:0').to(dtype).contiguous()

    img = tc('img')
    metas = {k: tc(k) for k in ['projection_mat', 'image_wh', 'timestamp',
                                'ego_status', 'tp_near', 'gt_ego_fut_cmd']}
    cmd = metas['gt_ego_fut_cmd'].long()

    with torch.no_grad():
        fm, _ = model.extract_feat(img)
        m = dict(metas)
        m['gt_ego_fut_cmd'] = cmd
        m['img_shape'] = (384, 704)
        det_out = model.head.det_head(fm, m, depth_prob=None)
        ae = det_out['anchor_embed']
        if_ = det_out['instance_feature']
        anchor_enc = model.head.det_head.anchor_encoder
        ego_q = np.load(os.path.join(REPO, 'configs/anchors/ego_query.npz'))
        ego_feat = torch.from_numpy(ego_q['ego_feature']).cuda()
        ego_anchor = torch.from_numpy(ego_q['ego_anchor']).cuda()
        ego_anchor_emb = anchor_enc(ego_anchor)
        QL = 21
        qf = torch.zeros(QL, 1, 901, 256, device='cuda')
        qa = torch.zeros(QL, 1, 901, 256, device='cuda')
        qf[QL - 1, 0, :900] = if_.squeeze(0)
        qf[QL - 1, 0, 900] = ego_feat[0]
        qa[QL - 1, 0, :900] = ae.squeeze(0)
        qa[QL - 1, 0, 900] = ego_anchor_emb[0]
        period = torch.full((1, 901), QL - 1, device='cuda', dtype=torch.int64)
        mask = torch.full((900,), -1.0, device='cuda')

        # PT reference outputs (engine-cascade mirror): map_reg is zeroed to
        # match the deployed chain (the map engine has no reg output).
        map_out = model.head.map_head(fm, m, depth_prob=None)
        map_out['prediction'][-1] = torch.zeros_like(map_out['prediction'][-1])
        mo_out, pl_out = model.head.motion_plan_head(
            det_out, map_out, fm, m, anchor_enc, mask, None)
        pr = pl_out['planning_results'][-1]
        pt = {
            'pt_motion_cls': mo_out['classification'][-1],
            'pt_motion_traj': mo_out['prediction'][-1],
            'pt_traj_cls': pr['traj_cls'],
            'pt_lat_anchor': pr['lat_anchor_filter'],
            'pt_lon_anchor': pr['lon_anchor_filter'],
        }

    out = os.environ.get('SD_DEMO_OUT', os.path.join(REPO, 'data/demo_data/sample0.npz'))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(
        out,
        img=img.squeeze(0).cpu().numpy(),
        projection_mat=metas['projection_mat'].squeeze(0).cpu().numpy(),
        image_wh=metas['image_wh'].squeeze(0).cpu().numpy(),
        timestamp=metas['timestamp'].cpu().numpy(),
        ego_status=metas['ego_status'].squeeze(0).cpu().numpy(),
        tp_near=metas['tp_near'].squeeze(0).cpu().numpy(),
        gt_ego_fut_cmd=metas['gt_ego_fut_cmd'].squeeze(0).cpu().numpy().astype(np.int64),
        mask=mask.cpu().numpy(),
        anchor_embed=ae.cpu().numpy(),
        queue_temp_feat=qf.cpu().numpy(),
        queue_temp_anchor=qa.cpu().numpy(),
        period=period.cpu().numpy(),
        pt_motion_cls=pt['pt_motion_cls'].cpu().numpy(),
        pt_motion_traj=pt['pt_motion_traj'].cpu().numpy(),
        pt_traj_cls=pt['pt_traj_cls'].cpu().numpy(),
        pt_lat_anchor=pt['pt_lat_anchor'].cpu().numpy(),
        pt_lon_anchor=pt['pt_lon_anchor'].cpu().numpy(),
    )
    print(f"Saved {out} ({os.path.getsize(out) / 1048576:.1f} MB)")


if __name__ == '__main__':
    main()
