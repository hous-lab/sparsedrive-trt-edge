"""Model loading + data preparation for SparseDriveV2 head ONNX export."""
import os
import sys
import types
# info pickles may be written by numpy 2.x; register the numpy._core alias so
# older envs (numpy 1.x, e.g. py3.8 + torch 2.0.1) can unpickle them.
if not hasattr(sys.modules.get('numpy'), '_core'):
    import numpy.core.multiarray as _ma
    import numpy.core.umath as _um
    _nc = types.ModuleType('numpy._core'); _nc.__path__ = []
    _nc.multiarray = _ma; _nc.umath = _um
    sys.modules['numpy._core'] = _nc
    sys.modules['numpy._core.multiarray'] = _ma
    sys.modules['numpy._core.umath'] = _um
import torch
import numpy as np
from mmcv import Config
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet.models import build_detector
from mmdet.datasets import build_dataset
from mmcv.parallel import DataContainer, MMDataParallel

# training repo root: scripts -> onnx_export -> repo -> pipeline -> ROOT
project_root = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', '..', '..')
project_root = os.environ.get('SPARSEDRIVE_PROJECT_ROOT', project_root)
sys.path.insert(0, project_root)
os.chdir(project_root)
import projects.mmdet3d_plugin
from projects.mmdet3d_plugin.datasets.builder import build_dataloader


def load_model(config_path, checkpoint_path, device='cuda:0'):
    cfg = Config.fromfile(config_path)
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, checkpoint_path, map_location='cpu')
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    model.to(device)
    model.eval()
    return model, cfg


def build_val_loader(cfg, device='cuda:0'):
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=1,
        dist=False, shuffle=False)
    return data_loader, dataset


def extract_formatted_feats(module, img, device='cuda:0'):
    """Run extract_feat on the inner module to get formatted feature maps."""
    img = img.to(device)
    with torch.no_grad():
        feat_tuple, depth_prob = module.extract_feat(img)
    return feat_tuple  # (feat_flatten, spatial_shapes, level_start_index)
