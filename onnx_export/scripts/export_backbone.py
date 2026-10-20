import os
import sys
import argparse
import torch

# training repo root: scripts -> onnx_export -> repo -> pipeline -> ROOT
HERE = os.path.abspath(os.path.dirname(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
project_root = os.path.join(HERE, '..', '..', '..', '..')
project_root = os.environ.get('SPARSEDRIVE_PROJECT_ROOT', project_root)
sys.path.insert(0, project_root)
os.chdir(project_root)

from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
import projects.mmdet3d_plugin


class BackboneExportWrapper(torch.nn.Module):
    def __init__(self, backbone, neck):
        super().__init__()
        self.img_backbone = backbone
        self.img_neck = neck

    def forward(self, img):
        feature_maps = self.img_backbone(img)
        if self.img_neck is not None:
            feature_maps = list(self.img_neck(feature_maps))
        return tuple(feature_maps)


def export_backbone(config_path, checkpoint_path, output_path, device='cuda:0'):
    print(f"[export_backbone] Loading config: {config_path}")
    cfg = Config.fromfile(config_path)

    print("[export_backbone] Building full model...")
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))

    print(f"[export_backbone] Loading checkpoint: {checkpoint_path}")
    load_checkpoint(model, checkpoint_path, map_location='cpu')
    model.to(device)
    model.eval()

    backbone = model.module.img_backbone if hasattr(model, 'module') else model.img_backbone
    neck = model.module.img_neck if hasattr(model, 'module') else model.img_neck

    wrapper = BackboneExportWrapper(backbone, neck).to(device)
    wrapper.eval()

    dummy_input = torch.randn(6, 3, 384, 704, device=device)

    print("[export_backbone] Exporting to ONNX...")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_input,
            output_path,
            input_names=['img'],
            output_names=[f'feat_{i}' for i in range(4)],
            dynamic_axes=None,
            opset_version=15,
            do_constant_folding=False,
        )

    print(f"[export_backbone] Done: {output_path}")
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"[export_backbone] File size: {size_mb:.1f} MB")


if __name__ == '__main__':
    _repo, _root = REPO, project_root
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default=os.environ.get('SD_CONFIG',
                            os.path.join(_repo, 'configs/sparsedrive_stage2.py')))
    parser.add_argument('--checkpoint', type=str,
                        default=os.environ.get('SD_CKPT',
                            os.path.join(_root, 'ckpt/sparsedrive_small_b2d_stage2.pth')))
    parser.add_argument('--output', type=str,
                        default='onnx_export/output/sparsedrive_backbone.onnx')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    if not os.path.isabs(args.output):
        args.output = os.path.join(REPO, args.output)
    export_backbone(args.config, args.checkpoint, args.output, args.device)
