"""Correct InstanceQueue externalization — proper tensor dims, no NaN."""
import torch
from torch.onnx import register_custom_op_symbolic
import torch.onnx.symbolic_helper as sym_help

# atan2 decomposition for ONNX: atan2(y,x) = 2*atan(y / (x + sqrt(x^2+y^2)))
def atan2_symbolic(g, y, x):
    two = g.op("Constant", value_t=torch.tensor(2.0, dtype=torch.float))
    eps = g.op("Constant", value_t=torch.tensor(1e-6, dtype=torch.float))
    x_sq = g.op("Mul", x, x)
    y_sq = g.op("Mul", y, y)
    x_sq_plus_y_sq = g.op("Add", x_sq, y_sq)
    sqrt_term = g.op("Sqrt", g.op("Add", x_sq_plus_y_sq, eps))
    denom = g.op("Add", x, sqrt_term)
    ratio = g.op("Div", y, denom)
    atan_val = g.op("Atan", ratio)
    return g.op("Mul", two, atan_val)
try:
    register_custom_op_symbolic("aten::atan2", atan2_symbolic, 17)
except:
    pass

def patch_InstanceQueue_external():
    from projects.mmdet3d_plugin.models.motion.instance_queue import InstanceQueue

    orig_get = InstanceQueue.get

    def patched_get(self, det_output, feature_maps, metas, batch_size, mask, anchor_handler):
        if hasattr(self, 'external_queue_state') and self.external_queue_state is not None:
            return self.external_queue_state
        if hasattr(self, 'external_temp_feat') and self.external_temp_feat is not None:
            # Use externalized queue tensors directly
            return (self.external_ego_feat, self.external_ego_anchor,
                    self.external_temp_feat, self.external_temp_anchor,
                    self.external_temp_mask)
        return orig_get(self, det_output, feature_maps, metas, batch_size, mask, anchor_handler)

    InstanceQueue.get = patched_get
    # cache methods are no-ops for ONNX export (state managed externally)
    InstanceQueue.cache_motion = lambda self, *a, **kw: None
    InstanceQueue.cache_planning = lambda self, *a, **kw: None
    print("[patch] InstanceQueue → externalized")


def patch_motion_tile():
    """Safe torch.tile -> expand replacement."""
    import projects.mmdet3d_plugin.models.motion.motion_planning_head_v13 as mp

    def safe_expand(tensor, repeats):
        # tile(tensor, repeats) = broadcast + repeat
        # expand(target_shape) = broadcast + repeat
        # When all repeats == 1, just add broadcast dims (no actual expansion)
        for _ in range(len(repeats) - tensor.dim()):
            tensor = tensor.unsqueeze(0)
        # If any repeat > 1, we need actual expansion
        if any(r > 1 for r in repeats):
            target = tensor.shape
            expand_shape = []
            for i in range(len(repeats)):
                expand_shape.append(target[i] * repeats[i])
            return tensor.expand(*expand_shape)
        return tensor

    orig_fwd = mp.MotionPlanningHeadV13.forward

    def patched_fwd(self, det_output, map_output, feature_maps, metas, anchor_encoder, mask, anchor_handler):
        old_tile = torch.tile
        torch.tile = safe_expand
        try:
            return orig_fwd(self, det_output, map_output, feature_maps, metas, anchor_encoder, mask, anchor_handler)
        finally:
            torch.tile = old_tile

    mp.MotionPlanningHeadV13.forward = patched_fwd
    print("[patch] motion_plan_head tile→expand")
