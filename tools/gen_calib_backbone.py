"""Generate backbone INT8 calibration data from the val set.

Run inside the training-repo env (mmcv/mmdet + torch). Feeds real val images
through the dataset pipeline (resize/normalize) and saves the raw
[6,3,384,704] float32 inputs — the backbone engine's exact input interface.

Writes x86/trt_compile/int8/calib_data/backbone_calib.npz with 'imgs'
[64,6,3,384,704] f32.

Usage: python tools/gen_calib_backbone.py
Env: SPARSEDRIVE_PROJECT_ROOT, SD_CKPT, SD_CONFIG (same as gen_demo_data).
"""
import os
import sys
import argparse
import numpy as np
import torch
if not hasattr(np, '_core'):
    # Info pickles written under numpy 2.x reference numpy._core (docs 7.8).
    import types
    import numpy.core.multiarray as _ma
    import numpy.core.umath as _um
    _nc = types.ModuleType('numpy._core')
    _nc.__path__ = []
    _nc.multiarray = _ma
    _nc.umath = _um
    sys.modules['numpy._core'] = _nc
    sys.modules['numpy._core.multiarray'] = _ma
    sys.modules['numpy._core.umath'] = _um
from mmcv import Config
from mmcv.runner import load_checkpoint
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


def gen_calib_backbone(config_path, checkpoint_path, output_path, num_frames=64, device='cuda:0'):
    cfg = Config.fromfile(config_path)
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, samples_per_gpu=1,
                                   workers_per_gpu=1, dist=False, shuffle=False)
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, checkpoint_path, map_location='cpu')
    model.to(device)
    model.eval()
    calib_imgs = []
    count = 0
    with torch.no_grad():
        for data in data_loader:
            if count >= num_frames:
                break
            img = data['img'].data[0] if hasattr(data['img'], 'data') else data['img']
            img = img.to(device).flatten(end_dim=1)  # [6,3,384,704]
            calib_imgs.append(img.cpu().numpy())
            count += 1
            print(f"  calib frame {count}/{num_frames}", end='\r')
    print()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, imgs=np.stack(calib_imgs))
    print(f"[calib] saved {len(calib_imgs)} frames -> {output_path} "
          f"({os.path.getsize(output_path)/1e6:.1f} MB)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=CONFIG)
    parser.add_argument('--checkpoint', default=CKPT)
    parser.add_argument('--output',
                        default=os.path.join(REPO, 'x86/trt_compile/int8/calib_data/backbone_calib.npz'))
    parser.add_argument('--num-frames', type=int, default=64)
    args = parser.parse_args()
    gen_calib_backbone(args.config, args.checkpoint, args.output, args.num_frames)
