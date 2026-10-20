#!/usr/bin/env python
"""Export SparseDriveV2 map head to ONNX.

The det head engine only outputs detection; the motion/planning engine needs
map_instance_feat / map_anchor_embed / map_cls as inputs. This script exports
the map head (Sparse4DHead for map, 100 anchors) so the runtime node can
produce real map features instead of zeros.

  inputs:
    feat_0..feat_3        [6, 256, ...]
    projection_mat        [6, 4, 4]
    image_wh              [6, 2]
  outputs:
    map_cls               [1, 100, 6]
    map_instance_feature  [1, 100, 256]
    map_anchor_embed      [1, 100, 256]
"""
import os, sys, types, torch, numpy as np
if not hasattr(np, '_core'):
    import numpy.core.multiarray as _ma
    import numpy.core.umath as _um
    _nc = types.ModuleType('numpy._core'); _nc.__path__ = []
    _nc.multiarray = _ma; _nc.umath = _um
    sys.modules['numpy._core'] = _nc
    sys.modules['numpy._core.multiarray'] = _ma
    sys.modules['numpy._core.umath'] = _um
# DEPLOY = the onnx_export dir (patches/ + scripts/ live under it);
# training repo root = DEPLOY/../../..
DEPLOY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(DEPLOY)
ROOT = os.path.abspath(os.path.join(DEPLOY, '..', '..', '..'))
PROJECT_ROOT = ROOT
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, DEPLOY); sys.path.insert(0, os.path.join(DEPLOY, 'patches'))
sys.path.insert(0, os.path.join(DEPLOY, 'scripts'))
from projects.mmdet3d_plugin.ops.deformable_aggregation import deformable_format
import export_head_model as ehm
import patch_deformable_agg


class MapHeadExportWrapper(torch.nn.Module):
    def __init__(self, head):
        super().__init__()
        self.head = head

    def forward(self, feat_0, feat_1, feat_2, feat_3,
                projection_mat, image_wh):
        bs = 1
        raw = [f.unsqueeze(0).expand(bs, -1, -1, -1, -1)
               for f in [feat_0, feat_1, feat_2, feat_3]]
        feat_flatten, spatial_shapes, level_start_index = deformable_format(raw)
        feat_tuple = (feat_flatten, spatial_shapes, level_start_index)

        metas = {
            'projection_mat': projection_mat.unsqueeze(0),
            'image_wh': image_wh.unsqueeze(0),
            'timestamp': torch.zeros(1, device=feat_0.device),
            'ego_status': torch.zeros(10, device=feat_0.device),
            'tp_near': torch.zeros(2, device=feat_0.device),
            'gt_ego_fut_cmd': torch.zeros(6, device=feat_0.device).long(),
            'img_shape': (384, 704),
        }

        map_out = self.head.map_head(feat_tuple, metas, depth_prob=None)

        return (map_out['classification'][-1],
                map_out['instance_feature'],
                map_out['anchor_embed'])


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--config', default=os.environ.get('SD_CONFIG', os.path.join(REPO, 'configs/sparsedrive_stage2.py')))
    p.add_argument('--checkpoint', default=os.environ.get('SD_CKPT', os.path.join(ROOT, 'ckpt/sparsedrive_small_b2d_stage2.pth')))
    p.add_argument('--output',
                   default=os.path.join(DEPLOY, 'output/sparsedrive_map.onnx'))
    args = p.parse_args()

    patch_deformable_agg.patch_DeformableFeatureAggregation()

    model_raw, cfg = ehm.load_model(args.config, args.checkpoint, 'cuda:0')
    data_loader, dataset = ehm.build_val_loader(cfg, 'cuda:0')
    for data in data_loader:
        img = data['img'].data[0] if hasattr(data['img'], 'data') else data['img']
        img = img.to('cuda:0')
        with torch.no_grad():
            img_f = img.flatten(end_dim=1)
            raw_feats = model_raw.img_backbone(img_f)
            raw_feats = list(model_raw.img_neck(raw_feats))
            for i, f in enumerate(raw_feats):
                raw_feats[i] = f.reshape(1, 6, *f.shape[1:])
        metas = {}
        for k in ['projection_mat', 'image_wh']:
            v = data[k]
            if hasattr(v, 'data') and not torch.is_tensor(v):
                v = v.data[0]
            if torch.is_tensor(v):
                v = v.to('cuda:0')
            metas[k] = v
        break

    wrapper = MapHeadExportWrapper(model_raw.head).to('cuda:0')
    wrapper.eval()
    dummy = (raw_feats[0].squeeze(0).float(), raw_feats[1].squeeze(0).float(),
             raw_feats[2].squeeze(0).float(), raw_feats[3].squeeze(0).float(),
             metas['projection_mat'].squeeze(0).float(),
             metas['image_wh'].squeeze(0).float())

    print("[export] checking wrapper output shapes...")
    with torch.no_grad():
        outs = wrapper(*dummy)
    for o in outs:
        print("  out:", tuple(o.shape))

    with torch.no_grad():
        torch.onnx.export(wrapper, dummy, args.output,
                          input_names=['feat_0', 'feat_1', 'feat_2', 'feat_3',
                                       'projection_mat', 'image_wh'],
                          output_names=['map_cls', 'map_instance_feature', 'map_anchor_embed'],
                          opset_version=15, do_constant_folding=False)
    print(f"Saved: {args.output} ({os.path.getsize(args.output) / 1024 / 1024:.1f} MB)")


if __name__ == '__main__':
    main()
