"""Export wrapper with instance_feature output."""
import torch
from projects.mmdet3d_plugin.ops.deformable_aggregation import deformable_format

class SparseDriveHeadExportWrapper(torch.nn.Module):
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
            'ego_status': torch.zeros(1, 3, device=feat_0.device),
            'tp_near': torch.zeros(1, 2, device=feat_0.device),
            'gt_ego_fut_cmd': torch.zeros(1, device=feat_0.device).long(),
            'img_shape': (384, 704),
        }

        det_out = self.head.det_head(feat_tuple, metas, depth_prob=None)

        return (det_out['classification'][-1],
                det_out['prediction'][-1],
                det_out['quality'][-1],
                det_out['instance_feature'])
