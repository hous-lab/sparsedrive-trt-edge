"""Export SparseDriveV2 detection head to ONNX (4 outputs incl. instance_feature)."""
import os, sys, argparse, logging
import torch, onnx, onnx_graphsurgeon as gs

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'onnx_export'))
sys.path.insert(0, os.path.join(REPO, 'onnx_export', 'patches'))
import export_head_model as ehm
import patch_deformable_agg, patch_instance_bank
from patch_head_forward import SparseDriveHeadExportWrapper
import numpy as np

CONFIG = os.environ.get(
    'SD_CONFIG', os.path.join(REPO, 'configs/sparsedrive_stage2.py'))
DEFAULT_CKPT = os.path.join(REPO, '..', '..', 'ckpt',
                            'sparsedrive_small_b2d_stage2.pth')
CKPT = os.environ.get('SD_CKPT', DEFAULT_CKPT)


def export_head(config_path, checkpoint_path, output_path, device='cuda:0'):
    patch_deformable_agg.patch_DeformableFeatureAggregation()
    patch_instance_bank.patch_InstanceBank()
    model_raw, cfg = ehm.load_model(config_path, checkpoint_path, device)
    data_loader, dataset = ehm.build_val_loader(cfg, device)
    for data in data_loader:
        img = data['img'].data[0] if hasattr(data['img'], 'data') else data['img']
        img = img.to(device)
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
                v = v.to(device)
            metas[k] = v
        break
    wrapper = SparseDriveHeadExportWrapper(model_raw.head).to(device)
    wrapper.eval()
    dummy = (raw_feats[0].squeeze(0).float(), raw_feats[1].squeeze(0).float(),
             raw_feats[2].squeeze(0).float(), raw_feats[3].squeeze(0).float(),
             metas['projection_mat'].squeeze(0).float(),
             metas['image_wh'].squeeze(0).float())
    with torch.no_grad():
        torch.onnx.export(wrapper, dummy, output_path,
            input_names=['feat_0', 'feat_1', 'feat_2', 'feat_3',
                         'projection_mat', 'image_wh'],
            output_names=['det_cls', 'det_reg', 'det_quality',
                          'det_instance_feat'],
            opset_version=15, do_constant_folding=False)
    graph = gs.import_onnx(onnx.load(output_path))
    n_plugin = sum(1 for n in graph.nodes if n.op == 'DeformableAggregationPlugin')
    n_ph = sum(1 for n in graph.nodes if n.op == 'Placeholder')
    logging.info('plugin nodes: %d, placeholder: %d', n_plugin, n_ph)
    print('Saved: %s (%.1f MB)' % (output_path,
                                   os.path.getsize(output_path) / 1048576))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=CONFIG)
    ap.add_argument('--checkpoint', default=CKPT)
    ap.add_argument('--output',
                    default=os.path.join(REPO, 'onnx_export/output/sparsedrive_head.onnx'))
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    export_head(a.config, a.checkpoint, a.output, a.device)
