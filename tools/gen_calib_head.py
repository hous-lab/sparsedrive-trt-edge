"""
Generate head calibration data by running model and capturing inputs.
Similar to VAD's gen_calib_data.py
"""
import os
import sys
import argparse
import torch
import numpy as np
from mmcv import Config
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet.models import build_detector

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
import projects.mmdet3d_plugin
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet.datasets import build_dataset


def gen_calib_head(config_path, checkpoint_path, output_path,
                   num_frames=64, device='cuda:0'):
    cfg = Config.fromfile(config_path)
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=1,
        dist=False, shuffle=False)

    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, checkpoint_path, map_location='cpu')
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    model.to(device)
    model.eval()

    calib_frames = []
    prev_bev = None
    count = 0

    with torch.no_grad():
        for data in data_loader:
            if count >= num_frames:
                break

            img = data['img'].data[0] if hasattr(data['img'], 'data') else data['img']
            img = img.to(device)
            feat_maps, _ = model.extract_feat(img)

            calib_frames.append({
                'feat_0': feat_maps[0].cpu().numpy(),
                'feat_1': feat_maps[1].cpu().numpy(),
                'feat_2': feat_maps[2].cpu().numpy(),
                'feat_3': feat_maps[3].cpu().numpy(),
            })
            count += 1
            print(f"  Calib frame {count}/{num_frames}", end='\r')

    print(f"\nSaving {len(calib_frames)} frames to {output_path}")
    np.savez_compressed(output_path, frames=calib_frames)
    print(f"Done: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='projects/configs/sparsedrive_stage2.py')
    parser.add_argument('--checkpoint', default='ckpt/sparsedrive_small_b2d_stage2.pth')
    parser.add_argument('--output', default='x86/trt_compile/int8/calib_data/head_calib.npz')
    parser.add_argument('--num-frames', type=int, default=64)
    args = parser.parse_args()
    gen_calib_head(args.config, args.checkpoint, args.output, args.num_frames)
