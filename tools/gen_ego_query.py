#!/usr/bin/env python
"""Extract the ego query (learned instance-queue init) from the checkpoint.

The deployed motion-plan engine consumes the model's learned ego query as a
static deployment input: the instance queue holds a learned `ego_feature`
(nn.Embedding(1, embed_dims), checkpoint-trained) and a constant `ego_anchor`
(nn.Parameter, requires_grad=False). This tool dumps both into
configs/anchors/ego_query.npz - the same file consumed by
tools/gen_demo_data.py when building the motion-plan queue inputs.

Run inside the training-repo env (torch + mmcv), checkpoint on disk.
Usage: python tools/gen_ego_query.py
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
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector

import projects.mmdet3d_plugin  # noqa: F401  (registers the SparseDrive model)


def main():
    cfg = Config.fromfile(os.environ.get(
        'SD_CONFIG', os.path.join(REPO, 'configs/sparsedrive_stage2.py')))
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, CKPT, map_location='cpu')
    q = model.head.motion_plan_head.instance_queue
    ego_feature = q.ego_feature.weight.detach().cpu().numpy().astype(np.float32)
    ego_anchor = q.ego_anchor.detach().cpu().numpy().astype(np.float32)
    out = os.path.join(REPO, 'configs/anchors/ego_query.npz')
    np.savez(out, ego_feature=ego_feature, ego_anchor=ego_anchor)
    print('wrote', out)
    print('  ego_feature', ego_feature.shape, ego_feature.dtype)
    print('  ego_anchor', ego_anchor.shape, ego_anchor.dtype)


if __name__ == '__main__':
    main()
