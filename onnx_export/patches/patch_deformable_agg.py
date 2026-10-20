"""
DeformableAggregation ONNX symbolic — VAD-style approach.
Directly adds symbolic() to the original autograd.Function classes.
This avoids creating wrapper classes that confuse PyTorch JIT tracer.
"""
import torch


def patch_DeformableFeatureAggregation():
    """Add symbolic methods to existing DeformableAggregationFunction classes."""
    from projects.mmdet3d_plugin.ops.deformable_aggregation import (
        DeformableAggregationFunction,
        DeformableAggregationWithDepthFunction,
    )

    # Add symbolic to standard version
    def _standard_symbolic(g, mc_ms_feat, spatial_shape, scale_start_index,
                           sampling_location, weights):
        return g.op("custom_op::DeformableAggregationPlugin",
                    mc_ms_feat, spatial_shape, scale_start_index,
                    sampling_location, weights,
                    num_depths_i=0)

    DeformableAggregationFunction.symbolic = staticmethod(_standard_symbolic)

    # Add symbolic to depth version
    def _depth_symbolic(g, mc_ms_feat, spatial_shape, scale_start_index,
                        sampling_location, weights, num_depths):
        return g.op("custom_op::DeformableAggregationPlugin",
                    mc_ms_feat, spatial_shape, scale_start_index,
                    sampling_location, weights,
                    num_depths_i=int(num_depths))

    DeformableAggregationWithDepthFunction.symbolic = staticmethod(_depth_symbolic)

    print("[patch] Added symbolic to DeformableAggregationFunction (VAD-style)")
