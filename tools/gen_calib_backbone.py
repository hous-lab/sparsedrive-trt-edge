import os
import sys
import argparse
import numpy as np
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
import projects.mmdet3d_plugin

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet.datasets import build_dataset


def gen_calib_backbone(config_path, checkpoint_path, output_path, num_frames=64, device='cuda:0'):
    cfg = Config.fromfile(config_path)
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=1, dist=False, shuffle=False)

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
            img = img.to(device)
            img_flat = img.flatten(end_dim=1)  # [6, 3, 384, 704]
            calib_imgs.append(img_flat.cpu().numpy())
            count += 1
            print(f"  Calib frame {count}/{num_frames}", end='\r')

    print(f"\nSaving {len(calib_imgs)} frames to {output_path}")
    calib_data = {'imgs': np.stack(calib_imgs)}
    np.savez_compressed(output_path, **calib_data)
    print(f"Done: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='projects/configs/sparsedrive_stage2.py')
    parser.add_argument('--checkpoint', default='ckpt/sparsedrive_small_b2d_stage2.pth')
    parser.add_argument('--output', default='x86/trt_compile/int8/calib_data/backbone_calib.npz')
    parser.add_argument('--num-frames', type=int, default=64)
    args = parser.parse_args()
    gen_calib_backbone(args.config, args.checkpoint, args.output, args.num_frames)
