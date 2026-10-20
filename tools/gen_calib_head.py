"""Generate head/motion INT8 calibration data from the val set.

Run inside the training-repo env. Saves BOTH engine input layouts:
  - head/map engines: feat_0..feat_3 [6,256,h,w] (unflattened, float16)
  - metas:            projection_mat [6,4,4], image_wh [6,2]

Usage: python tools/gen_calib_head.py
Env: SPARSEDRIVE_PROJECT_ROOT, SD_CKPT, SD_CONFIG (same as gen_demo_data).
"""
import os
import sys
import argparse
import torch
import numpy as np
if not hasattr(np, '_core'):
    # Info pickles written under numpy 2.x reference numpy._core (docs 7.8).
    import types
    import numpy.core.multiarray as _ma
    import numpy.core.umath as _um
    _nc = types.ModuleType('numpy._core'); _nc.__path__ = []
    _nc.multiarray = _ma; _nc.umath = _um
    sys.modules['numpy._core'] = _nc
    sys.modules['numpy._core.multiarray'] = _ma
    sys.modules['numpy._core.umath'] = _um
from mmcv import Config
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet.models import build_detector

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ROOT = os.environ.get('SPARSEDRIVE_PROJECT_ROOT',
                      os.path.abspath(os.path.join(REPO, '..', '..')))
CKPT = os.environ.get('SD_CKPT', os.path.join(ROOT, 'ckpt/sparsedrive_small_b2d_stage2.pth'))
CONFIG = os.environ.get('SD_CONFIG', os.path.join(ROOT, 'projects/configs/sparsedrive_stage2.py'))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import projects.mmdet3d_plugin  # noqa: E402
from projects.mmdet3d_plugin.datasets.builder import build_dataloader  # noqa: E402
from mmdet.datasets import build_dataset  # noqa: E402

FEAT_SHAPES = [(96, 176), (48, 88), (24, 44), (12, 22)]


FEAT_SHAPES = [(96, 176), (48, 88), (24, 44), (12, 22)]
SIZES = [h * w for h, w in FEAT_SHAPES]


def unflatten_flat(flat):
    """deformable_format output [1,6,N,256] -> per-level [6,256,h,w]."""
    x = flat[0]  # [6,N,256]
    feats = []
    off = 0
    for (h, w) in FEAT_SHAPES:
        n = h * w
        c = x[:, off:off + n]                                # [6,n,256]
        feats.append(c.reshape(6, h, w, 256).permute(0, 3, 1, 2).contiguous())
        off += n
    return feats


def flatten_feats(feat_i_list):
    """Per-level [6,256,h,w] -> motion-engine feat_flat (same as gen_x86_reference)."""
    flat = torch.cat([f.reshape(f.shape[0], f.shape[1], -1).permute(0, 2, 1).reshape(-1, f.shape[1])
                      for f in feat_i_list], dim=0)
    return flat.unsqueeze(0).contiguous()


def gen_calib_head(config_path, checkpoint_path, output_path, num_frames=64, device='cuda:0'):
    cfg = Config.fromfile(config_path)
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, samples_per_gpu=1,
                                   workers_per_gpu=1, dist=False, shuffle=False)
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, checkpoint_path, map_location='cpu')
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    model.to(device)
    model.eval()
    calib = []
    count = 0
    with torch.no_grad():
        for data in data_loader:
            if count >= num_frames:
                break
            img = data['img'].data[0] if hasattr(data['img'], 'data') else data['img']
            img = img.to(device).flatten(end_dim=1)  # [6,3,384,704]
            loop_dummy = None  # extracted feats are batch-first [6,256,h,w]
            fm, _ = model.extract_feat(img.unsqueeze(0))
            feats = unflatten_flat(fm[0])  # fm=(flat[1,6,N,256], spat, lvl)
            # float16 halves the npz (calibrators cast back to f32); no
            # feat_flat - the motion engine stays FP16 in this repo's INT8 tier
            frame = {'feat_%d' % i: f.cpu().numpy().astype(np.float16)
                     for i, f in enumerate(feats)}
            # head/map engines also take squeezed per-sample metas (docs 7.4)
            for k in ('projection_mat', 'image_wh'):
                v = data[k]
                v = v.data[0] if hasattr(v, 'data') and not torch.is_tensor(v) else v
                frame[k] = v.to('cpu').numpy().astype(np.float32)
            calib.append(frame)
            count += 1
            print(f"  calib frame {count}/{num_frames}", end='\r')
    print()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    keys = calib[0].keys()
    stacked = {}
    for k in keys:
        stacked[k] = np.stack([f[k] for f in calib])  # [64,*shape]
    np.savez(output_path, **stacked)
    print(f"[calib] saved {len(calib)} frames -> {output_path} "
          f"({os.path.getsize(output_path)/1e6:.1f} MB)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # REPO is absolute and was captured BEFORE chdir(ROOT); a relative default
    # here would resolve against the training repo (see docs/lessons.md 7.2).
    head_out = os.path.join(REPO, 'x86/trt_compile/int8/calib_data/head_calib.npz')
    parser.add_argument('--output', default=head_out)
    parser.add_argument('--config', default=CONFIG)
    parser.add_argument('--checkpoint', default=CKPT)
    parser.add_argument('--num-frames', type=int, default=64)
    args = parser.parse_args()
    gen_calib_head(args.config, args.checkpoint, args.output, args.num_frames)
