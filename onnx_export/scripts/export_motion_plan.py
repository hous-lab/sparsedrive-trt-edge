#!/usr/bin/env python
"""Export motion+planning head v2 — planning outputs fixed.

The v1 export returned `torch.zeros` for planning_traj (dead output), and the
planning computation was dead-code-eliminated in the TRT engine, making the
engine unable to produce a plan. This version exposes the real planning
tensors so the ROS2 node can decode the ego trajectory:

  outputs:
    motion_cls     [1, 900, 6]          object motion classification
    motion_traj    [1, 900, 6, 6, 2]    object motion trajectories
    traj_cls       [1, 64, 16]          planning trajectory logits (filtered)
    lat_anchor     [1, 64, 15, 2]       filtered lateral path anchors (ego frame)
    lon_anchor     [1, 16, 6]           filtered longitudinal velocity anchors

Node-side decode (matches the model's V13LatLonTrajDecoder.decode_v2):
    idx  = argmax(sigmoid(traj_cls).flatten(1,2), dim=-1)
    row  = idx // 16, col = idx % 16
    path = lat_anchor[0, row]   # [15, 2] ego-frame waypoints
    vel  = lon_anchor[0, col]   # [6]    m/s profile
"""
import os, sys, types, argparse, torch, numpy as np
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
sys.path.insert(0, DEPLOY); sys.path.insert(0, os.path.join(DEPLOY, 'patches'))
sys.path.insert(0, os.path.join(DEPLOY, 'scripts'))
import export_head_model as ehm
import patch_deformable_agg
from patch_motion_head import patch_InstanceQueue_external, patch_motion_tile

QL = 21  # queue length
NUM_LON_FILTERED = 16  # filter_num[1] of the last planning decoder


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.head = model.head
        self.anchor_encoder = model.head.det_head.anchor_encoder

    def forward(self, feat_flat, spat_shape, lvl_start,
                det_cls, det_reg, det_quality,
                det_instance_feat, det_anchor_embed,
                map_cls, map_reg, map_instance_feat, map_anchor_embed,
                projection_mat, image_wh, timestamp, ego_status,
                tp_near, gt_ego_fut_cmd, mask,
                queue_temp_feat, queue_temp_anchor, period):
        bs, dvc = 1, feat_flat.device
        n_inst = det_instance_feat.shape[1]

        det_out = {'classification': [det_cls] * 6, 'prediction': [det_reg] * 6,
                   'quality': [det_quality] * 6, 'instance_feature': det_instance_feat,
                   'anchor_embed': det_anchor_embed,
                   'instance_id': -torch.ones(1, n_inst, device=dvc).long()}
        map_out = {'classification': [map_cls] * 6, 'prediction': [map_reg] * 6,
                   'instance_feature': map_instance_feat, 'anchor_embed': map_anchor_embed}
        fm = (feat_flat, spat_shape, lvl_start)
        metas = {'projection_mat': projection_mat.unsqueeze(0),
                 'image_wh': image_wh.unsqueeze(0),
                 'timestamp': timestamp.unsqueeze(0),
                 'ego_status': ego_status.unsqueeze(0),
                 'tp_near': tp_near.unsqueeze(0),
                 'gt_ego_fut_cmd': gt_ego_fut_cmd.unsqueeze(0),
                 'img_shape': (384, 704)}

        queue = self.head.motion_plan_head.instance_queue
        ego_feat = queue.ego_feature.weight[None].repeat(bs, 1, 1)
        ego_anchor = queue.ego_anchor[None].repeat(bs, 1, 1)

        temp_feat_perm = queue_temp_feat.permute(1, 2, 0, 3)     # [bs, 901, ql, dim]
        temp_anchor_perm = queue_temp_anchor.permute(1, 2, 0, 3)

        queue.external_ego_feat = ego_feat
        queue.external_ego_anchor = ego_anchor
        queue.external_temp_feat = temp_feat_perm
        queue.external_temp_anchor = temp_anchor_perm

        ql = queue_temp_feat.shape[0]
        total_inst = temp_feat_perm.shape[1]
        ar = torch.arange(ql, 0, -1, device=dvc, dtype=torch.float)
        ar = ar[None, None, :].expand(bs, total_inst, -1)
        per = period.float().unsqueeze(-1)
        queue.external_temp_mask = (ar > per)

        motion_out, plan_out = self.head.motion_plan_head(
            det_out, map_out, fm, metas, self.anchor_encoder, mask, None)

        pr = plan_out['planning_results'][-1]
        motion_cls = motion_out['classification'][-1]
        motion_traj = motion_out['prediction'][-1]
        traj_cls = pr['traj_cls']
        lat_anchor = pr['lat_anchor_filter']
        lon_anchor = pr['lon_anchor_filter']
        return motion_cls, motion_traj, traj_cls, lat_anchor, lon_anchor


def main():
    p = argparse.ArgumentParser()
    _repo = REPO
    _root = os.environ.get('SPARSEDRIVE_PROJECT_ROOT', ROOT)
    p.add_argument('--config', default=os.environ.get('SD_CONFIG', os.path.join(_repo, 'configs/sparsedrive_stage2.py')))
    p.add_argument('--checkpoint', default=os.environ.get('SD_CKPT', os.path.join(_root, 'ckpt/sparsedrive_small_b2d_stage2.pth')))
    p.add_argument('--output', default=os.path.join(_repo, 'onnx_export/output/sparsedrive_motion_plan.onnx'))
    args = p.parse_args()
    if not os.path.isabs(args.output):
        args.output = os.path.join(REPO, args.output)

    patch_deformable_agg.patch_DeformableFeatureAggregation()
    patch_InstanceQueue_external()
    patch_motion_tile()

    model, cfg = ehm.load_model(args.config, args.checkpoint, 'cuda:0')
    dl, ds = ehm.build_val_loader(cfg, 'cuda:0')

    for data in dl:
        img = data['img'].data[0] if hasattr(data['img'], 'data') else data['img']
        img = img.to('cuda:0')
        with torch.no_grad():
            ft, _ = model.extract_feat(img)
            metas_min = {}
            for k in ['projection_mat', 'image_wh', 'timestamp', 'ego_status', 'tp_near', 'gt_ego_fut_cmd']:
                v = data[k]
                if hasattr(v, 'data') and not torch.is_tensor(v):
                    v = v.data
                if torch.is_tensor(v):
                    v = v.to('cuda:0')
                metas_min[k] = v
            metas_min['img_shape'] = (384, 704)
            det_out = model.head.det_head(ft, metas_min, depth_prob=None)
            map_out = model.head.map_head(ft, metas_min, depth_prob=None)
        break

    wrapper = Wrapper(model).to('cuda:0')
    wrapper.eval()
    feat_flat, spat_shape, lvl_start = ft

    bs, n_inst = 1, 900
    dim = 256
    inst_feat = det_out['instance_feature']
    anchor = det_out['anchor_embed']

    temp_feat = torch.zeros(QL, bs, n_inst + 1, dim, device='cuda:0')
    temp_anchor = torch.zeros(QL, bs, n_inst + 1, 256, device='cuda:0')
    temp_feat[QL - 1, :, :n_inst, :] = inst_feat
    temp_anchor[QL - 1, :, :n_inst, :] = anchor

    period = torch.full((bs, n_inst + 1), QL - 1, device='cuda:0', dtype=torch.long)
    mask_t = det_out.get('instance_id', torch.ones(1, 900, device='cuda:0')).float()

    dummy = (feat_flat.float(), spat_shape.int(), lvl_start.int(),
             det_out['classification'][-1].float(),
             det_out['prediction'][-1].float(),
             det_out['quality'][-1].float(),
             inst_feat.float(), anchor.float(),
             map_out['classification'][-1].float(),
             map_out['prediction'][-1].float(),
             map_out['instance_feature'].float(),
             map_out['anchor_embed'].float(),
             metas_min['projection_mat'].squeeze(0).float(),
             metas_min['image_wh'].squeeze(0).float(),
             metas_min['timestamp'].squeeze(0).float(),
             metas_min['ego_status'].squeeze(0).float(),
             metas_min['tp_near'].squeeze(0).float(),
             metas_min['gt_ego_fut_cmd'].squeeze(0).long(),
             mask_t.squeeze(0).float(),
             temp_feat.float(), temp_anchor.float(), period.long())

    print("[export] checking wrapper output shapes...")
    with torch.no_grad():
        outs = wrapper(*dummy)
    for o in outs:
        print("  out:", tuple(o.shape))

    with torch.no_grad():
        torch.onnx.export(wrapper, dummy, args.output,
                          input_names=['feat_flat', 'spat_shape', 'lvl_start',
                                       'det_cls', 'det_reg', 'det_quality',
                                       'det_instance_feat', 'det_anchor_embed',
                                       'map_cls', 'map_reg', 'map_instance_feat', 'map_anchor_embed',
                                       'projection_mat', 'image_wh', 'timestamp', 'ego_status',
                                       'tp_near', 'gt_ego_fut_cmd', 'mask',
                                       'queue_temp_feat', 'queue_temp_anchor', 'period'],
                          output_names=['motion_cls', 'motion_traj', 'traj_cls',
                                        'lat_anchor', 'lon_anchor'],
                          opset_version=17, do_constant_folding=False)
    print(f"Saved: {args.output} ({os.path.getsize(args.output) / 1024 / 1024:.1f} MB)")


if __name__ == '__main__':
    main()
