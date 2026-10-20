#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "common.h"
#include <stdint.h>

// ==================== FP32 bilinear ====================
__device__ float bilinear_sampling_f32(
    const float *bottom_data, int height, int width,
    int num_embeds, float h_im, float w_im, int base_ptr) {

    int h_low = floorf(h_im);
    int w_low = floorf(w_im);
    int h_high = h_low + 1;
    int w_high = w_low + 1;

    float lh = h_im - h_low;
    float lw = w_im - w_low;
    float hh = 1.0f - lh, hw = 1.0f - lw;

    int w_stride = num_embeds;
    int h_stride = width * w_stride;
    int h_low_off = h_low * h_stride;
    int h_high_off = h_low_off + h_stride;
    int w_low_off = w_low * w_stride;
    int w_high_off = w_low_off + w_stride;

    float v1 = 0, v2 = 0, v3 = 0, v4 = 0;
    if (h_low >= 0 && w_low >= 0)
        v1 = bottom_data[h_low_off + w_low_off + base_ptr];
    if (h_low >= 0 && w_high <= width - 1)
        v2 = bottom_data[h_low_off + w_high_off + base_ptr];
    if (h_high <= height - 1 && w_low >= 0)
        v3 = bottom_data[h_high_off + w_low_off + base_ptr];
    if (h_high <= height - 1 && w_high <= width - 1)
        v4 = bottom_data[h_high_off + w_high_off + base_ptr];

    return (hh * hw) * v1 + (hh * lw) * v2 + (lh * hw) * v3 + (lh * lw) * v4;
}

// ==================== FP16 bilinear ====================
__device__ __half bilinear_sampling_f16(
    const __half *bottom_data, int height, int width,
    int num_embeds, float h_im, float w_im, int base_ptr) {

    int h_low = floorf(h_im);
    int w_low = floorf(w_im);
    int h_high = h_low + 1;
    int w_high = w_low + 1;

    float lh = h_im - h_low;
    float lw = w_im - w_low;
    float hh = 1.0f - lh, hw = 1.0f - lw;

    int w_stride = num_embeds;
    int h_stride = width * w_stride;
    int h_low_off = h_low * h_stride;
    int h_high_off = h_low_off + h_stride;
    int w_low_off = w_low * w_stride;
    int w_high_off = w_low_off + w_stride;

    float v1 = 0, v2 = 0, v3 = 0, v4 = 0;
    if (h_low >= 0 && w_low >= 0)
        v1 = __half2float(bottom_data[h_low_off + w_low_off + base_ptr]);
    if (h_low >= 0 && w_high <= width - 1)
        v2 = __half2float(bottom_data[h_low_off + w_high_off + base_ptr]);
    if (h_high <= height - 1 && w_low >= 0)
        v3 = __half2float(bottom_data[h_high_off + w_low_off + base_ptr]);
    if (h_high <= height - 1 && w_high <= width - 1)
        v4 = __half2float(bottom_data[h_high_off + w_high_off + base_ptr]);

    float result = (hh * hw) * v1 + (hh * lw) * v2 + (lh * hw) * v3 + (lh * lw) * v4;
    return __float2half(result);
}

// ==================== FP32 bilinear with depth ====================
__device__ float bilinear_sampling_depth_f32(
    const float *bottom_data, int height, int width,
    int num_embeds_with_depth, int num_depths,
    float h_im, float w_im, float loc_d,
    int base_ptr, int depth_ptr) {

    int h_low = floorf(h_im);
    int w_low = floorf(w_im);
    int h_high = h_low + 1;
    int w_high = w_low + 1;
    int d_low = floorf(loc_d);
    int d_high = d_low + 1;

    float lh = h_im - h_low, lw = w_im - w_low;
    float hh = 1.0f - lh, hw = 1.0f - lw;
    float ld = loc_d - d_low, hd = 1.0f - ld;

    int w_stride = num_embeds_with_depth;
    int h_stride = width * w_stride;

    bool d_low_ok = d_low >= 0 && d_low < num_depths;
    bool d_high_ok = d_high >= 0 && d_high < num_depths;

    float v1 = 0, dp_l1 = 0, dp_h1 = 0;
    float v2 = 0, dp_l2 = 0, dp_h2 = 0;
    float v3 = 0, dp_l3 = 0, dp_h3 = 0;
    float v4 = 0, dp_l4 = 0, dp_h4 = 0;

    auto read_corner = [&](int h_idx, int w_idx, int hl_off, int wh_off,
                           float& v, float& dpl, float& dph) {
        int off = hl_off + wh_off;
        if (h_idx >= 0 && h_idx < height && w_idx >= 0 && w_idx < width) {
            v = bottom_data[off + base_ptr];
            if (d_low_ok) dpl = bottom_data[off + depth_ptr + d_low];
            if (d_high_ok) dph = bottom_data[off + depth_ptr + d_high];
        }
    };

    read_corner(h_low, w_low, h_low * h_stride, w_low * w_stride, v1, dp_l1, dp_h1);
    read_corner(h_low, w_high, h_low * h_stride, w_high * w_stride, v2, dp_l2, dp_h2);
    read_corner(h_high, w_low, h_high * h_stride, w_low * w_stride, v3, dp_l3, dp_h3);
    read_corner(h_high, w_high, h_high * h_stride, w_high * w_stride, v4, dp_l4, dp_h4);

    float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
    float val_low  = w1*v1*dp_l1 + w2*v2*dp_l2 + w3*v3*dp_l3 + w4*v4*dp_l4;
    float val_high = w1*v1*dp_h1 + w2*v2*dp_h2 + w3*v3*dp_h3 + w4*v4*dp_h4;
    return hd * val_low + ld * val_high;
}

// ==================== FP16 bilinear with depth ====================
__device__ __half bilinear_sampling_depth_f16(
    const __half *bottom_data, int height, int width,
    int num_embeds_with_depth, int num_depths,
    float h_im, float w_im, float loc_d,
    int base_ptr, int depth_ptr) {

    int h_low = floorf(h_im), w_low = floorf(w_im);
    int h_high = h_low + 1, w_high = w_low + 1;
    int d_low = floorf(loc_d), d_high = d_low + 1;

    float lh = h_im - h_low, lw = w_im - w_low;
    float hh = 1.0f - lh, hw = 1.0f - lw;
    float ld = loc_d - d_low, hd = 1.0f - ld;

    int w_stride = num_embeds_with_depth;
    int h_stride = width * w_stride;

    bool d_low_ok = d_low >= 0 && d_low < num_depths;
    bool d_high_ok = d_high >= 0 && d_high < num_depths;

    float v1 = 0, dp_l1 = 0, dp_h1 = 0;
    float v2 = 0, dp_l2 = 0, dp_h2 = 0;
    float v3 = 0, dp_l3 = 0, dp_h3 = 0;
    float v4 = 0, dp_l4 = 0, dp_h4 = 0;

    auto read_corner = [&](int h_idx, int w_idx, int hl_off, int wh_off,
                           float& v, float& dpl, float& dph) {
        int off = hl_off + wh_off;
        if (h_idx >= 0 && h_idx < height && w_idx >= 0 && w_idx < width) {
            v = __half2float(bottom_data[off + base_ptr]);
            if (d_low_ok) dpl = __half2float(bottom_data[off + depth_ptr + d_low]);
            if (d_high_ok) dph = __half2float(bottom_data[off + depth_ptr + d_high]);
        }
    };

    read_corner(h_low, w_low, h_low * h_stride, w_low * w_stride, v1, dp_l1, dp_h1);
    read_corner(h_low, w_high, h_low * h_stride, w_high * w_stride, v2, dp_l2, dp_h2);
    read_corner(h_high, w_low, h_high * h_stride, w_low * w_stride, v3, dp_l3, dp_h3);
    read_corner(h_high, w_high, h_high * h_stride, w_high * w_stride, v4, dp_l4, dp_h4);

    float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
    float val_low  = w1*v1*dp_l1 + w2*v2*dp_l2 + w3*v3*dp_l3 + w4*v4*dp_l4;
    float val_high = w1*v1*dp_h1 + w2*v2*dp_h2 + w3*v3*dp_h3 + w4*v4*dp_h4;
    return __float2half(hd * val_low + ld * val_high);
}

// ==================== FP32 forward kernel (no depth) ====================
__global__ void deformable_aggregation_kernel_f32(
    int num_kernels, float* output,
    const float* mc_ms_feat, const int* spatial_shape,
    const int* scale_start_index, const float* sample_location,
    const float* weights,
    int batch_size, int num_cams, int num_feat,
    int num_embeds, int num_scale, int num_pts, int num_groups) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_kernels) return;

    int channel_index = idx % num_embeds;
    int groups_index = channel_index / (num_embeds / num_groups);
    int tmp = idx / num_embeds;
    int pts_index = tmp % num_pts;
    int batch_index = tmp / num_pts;

    int value_cam_stride = num_feat * num_embeds;
    int weight_cam_stride = num_scale * num_groups;
    int loc_offset = (batch_index * num_pts + pts_index) * (num_cams << 1);
    int value_offset = batch_index * num_cams * value_cam_stride + channel_index;
    int weight_offset = (batch_index * num_pts + pts_index) * num_cams * weight_cam_stride + groups_index;

    float result = 0;
    for (int cam = 0; cam < num_cams; ++cam, loc_offset += 2) {
        float loc_w = sample_location[loc_offset];
        float loc_h = sample_location[loc_offset + 1];
        if (loc_w > 0 && loc_w < 1 && loc_h > 0 && loc_h < 1) {
            for (int s = 0; s < num_scale; ++s) {
                int scale_off = scale_start_index[s] * num_embeds;
                int h = spatial_shape[s * 2];
                int w = spatial_shape[s * 2 + 1];
                float h_im = loc_h * h - 0.5f;
                float w_im = loc_w * w - 0.5f;
                int vptr = value_offset + scale_off + value_cam_stride * cam;
                float wgt = weights[weight_offset + s * num_groups + weight_cam_stride * cam];
                result += bilinear_sampling_f32(mc_ms_feat, h, w, num_embeds, h_im, w_im, vptr) * wgt;
            }
        }
    }
    output[idx] = result;
}

// ==================== FP16 forward kernel (no depth) ====================
__global__ void deformable_aggregation_kernel_f16(
    int num_kernels, __half* output,
    const __half* mc_ms_feat, const int* spatial_shape,
    const int* scale_start_index, const __half* sample_location,
    const __half* weights,
    int batch_size, int num_cams, int num_feat,
    int num_embeds, int num_scale, int num_pts, int num_groups) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_kernels) return;

    int channel_index = idx % num_embeds;
    int groups_index = channel_index / (num_embeds / num_groups);
    int tmp = idx / num_embeds;
    int pts_index = tmp % num_pts;
    int batch_index = tmp / num_pts;

    int value_cam_stride = num_feat * num_embeds;
    int weight_cam_stride = num_scale * num_groups;
    int loc_offset = (batch_index * num_pts + pts_index) * (num_cams << 1);
    int value_offset = batch_index * num_cams * value_cam_stride + channel_index;
    int weight_offset = (batch_index * num_pts + pts_index) * num_cams * weight_cam_stride + groups_index;

    float result = 0;
    for (int cam = 0; cam < num_cams; ++cam, loc_offset += 2) {
        float loc_w = __half2float(sample_location[loc_offset]);
        float loc_h = __half2float(sample_location[loc_offset + 1]);
        if (loc_w > 0 && loc_w < 1 && loc_h > 0 && loc_h < 1) {
            for (int s = 0; s < num_scale; ++s) {
                int scale_off = scale_start_index[s] * num_embeds;
                int h = spatial_shape[s * 2];
                int w = spatial_shape[s * 2 + 1];
                float h_im = loc_h * h - 0.5f;
                float w_im = loc_w * w - 0.5f;
                int vptr = value_offset + scale_off + value_cam_stride * cam;
                float wgt = __half2float(weights[weight_offset + s * num_groups + weight_cam_stride * cam]);
                result += __half2float(bilinear_sampling_f16(mc_ms_feat, h, w, num_embeds, h_im, w_im, vptr)) * wgt;
            }
        }
    }
    output[idx] = __float2half(result);
}

// ==================== FP32 forward kernel (with depth) ====================
__global__ void deformable_aggregation_depth_kernel_f32(
    int num_kernels, float* output,
    const float* mc_ms_feat, const int* spatial_shape,
    const int* scale_start_index, const float* sample_location,
    const float* weights,
    int batch_size, int num_cams, int num_feat,
    int num_embeds, int num_scale, int num_pts, int num_groups,
    int num_depths) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_kernels) return;

    int channel_index = idx % num_embeds;
    int groups_index = channel_index / (num_embeds / num_groups);
    int tmp = idx / num_embeds;
    int pts_index = tmp % num_pts;
    int batch_index = tmp / num_pts;

    int num_embeds_with_depth = num_embeds + num_depths;
    int value_cam_stride = num_feat * num_embeds_with_depth;
    int weight_cam_stride = num_scale * num_groups;
    int loc_offset = (batch_index * num_pts + pts_index) * num_cams * 3;
    int base_offset = batch_index * num_cams * value_cam_stride;
    int depth_offset = base_offset + num_embeds;
    int value_offset = base_offset + channel_index;
    int weight_offset = (batch_index * num_pts + pts_index) * num_cams * weight_cam_stride + groups_index;

    float result = 0;
    for (int cam = 0; cam < num_cams; ++cam, loc_offset += 3) {
        float loc_w = sample_location[loc_offset];
        float loc_h = sample_location[loc_offset + 1];
        float loc_d = sample_location[loc_offset + 2];
        if (loc_w > 0 && loc_w < 1 && loc_h > 0 && loc_h < 1 &&
            loc_d > -1 && loc_d < num_depths) {
            for (int s = 0; s < num_scale; ++s) {
                int scale_off = scale_start_index[s] * num_embeds_with_depth;
                int h = spatial_shape[s * 2];
                int w = spatial_shape[s * 2 + 1];
                float h_im = loc_h * h - 0.5f;
                float w_im = loc_w * w - 0.5f;
                int vptr = value_offset + scale_off + value_cam_stride * cam;
                int dptr = depth_offset + scale_off + value_cam_stride * cam;
                float wgt = weights[weight_offset + s * num_groups + weight_cam_stride * cam];
                result += bilinear_sampling_depth_f32(
                    mc_ms_feat, h, w, num_embeds_with_depth, num_depths,
                    h_im, w_im, loc_d, vptr, dptr) * wgt;
            }
        }
    }
    output[idx] = result;
}

// ==================== FP16 forward kernel (with depth) ====================
__global__ void deformable_aggregation_depth_kernel_f16(
    int num_kernels, __half* output,
    const __half* mc_ms_feat, const int* spatial_shape,
    const int* scale_start_index, const __half* sample_location,
    const __half* weights,
    int batch_size, int num_cams, int num_feat,
    int num_embeds, int num_scale, int num_pts, int num_groups,
    int num_depths) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_kernels) return;

    int channel_index = idx % num_embeds;
    int groups_index = channel_index / (num_embeds / num_groups);
    int tmp = idx / num_embeds;
    int pts_index = tmp % num_pts;
    int batch_index = tmp / num_pts;

    int num_embeds_with_depth = num_embeds + num_depths;
    int value_cam_stride = num_feat * num_embeds_with_depth;
    int weight_cam_stride = num_scale * num_groups;
    int loc_offset = (batch_index * num_pts + pts_index) * num_cams * 3;
    int base_offset = batch_index * num_cams * value_cam_stride;
    int depth_offset = base_offset + num_embeds;
    int value_offset = base_offset + channel_index;
    int weight_offset = (batch_index * num_pts + pts_index) * num_cams * weight_cam_stride + groups_index;

    float result = 0;
    for (int cam = 0; cam < num_cams; ++cam, loc_offset += 3) {
        float loc_w = __half2float(sample_location[loc_offset]);
        float loc_h = __half2float(sample_location[loc_offset + 1]);
        float loc_d = __half2float(sample_location[loc_offset + 2]);
        if (loc_w > 0 && loc_w < 1 && loc_h > 0 && loc_h < 1 &&
            loc_d > -1 && loc_d < num_depths) {
            for (int s = 0; s < num_scale; ++s) {
                int scale_off = scale_start_index[s] * num_embeds_with_depth;
                int h = spatial_shape[s * 2];
                int w = spatial_shape[s * 2 + 1];
                float h_im = loc_h * h - 0.5f;
                float w_im = loc_w * w - 0.5f;
                int vptr = value_offset + scale_off + value_cam_stride * cam;
                int dptr = depth_offset + scale_off + value_cam_stride * cam;
                float wgt = __half2float(weights[weight_offset + s * num_groups + weight_cam_stride * cam]);
                result += __half2float(bilinear_sampling_depth_f16(
                    mc_ms_feat, h, w, num_embeds_with_depth, num_depths,
                    h_im, w_im, loc_d, vptr, dptr)) * wgt;
            }
        }
    }
    output[idx] = __float2half(result);
}

// ==================== FP32 entry points ====================
void deformable_aggregation_forward_cuda(
    float* output, const float* mc_ms_feat, const int* spatial_shape,
    const int* scale_start_index, const float* sample_location,
    const float* weights,
    int batch_size, int num_cams, int num_feat, int num_embeds,
    int num_scale, int num_pts, int num_groups, cudaStream_t stream) {

    int num_kernels = batch_size * num_pts * num_embeds;
    int blocks = DIVUP(num_kernels, THREADS_PER_BLOCK);
    deformable_aggregation_kernel_f32<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        num_kernels, output, mc_ms_feat, spatial_shape, scale_start_index,
        sample_location, weights, batch_size, num_cams, num_feat,
        num_embeds, num_scale, num_pts, num_groups);
    cudaCheckError();
}

void deformable_aggregation_with_depth_forward_cuda(
    float* output, const float* mc_ms_feat, const int* spatial_shape,
    const int* scale_start_index, const float* sample_location,
    const float* weights,
    int batch_size, int num_cams, int num_feat, int num_embeds,
    int num_scale, int num_pts, int num_groups, int num_depths,
    cudaStream_t stream) {

    int num_kernels = batch_size * num_pts * num_embeds;
    int blocks = DIVUP(num_kernels, THREADS_PER_BLOCK);
    deformable_aggregation_depth_kernel_f32<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        num_kernels, output, mc_ms_feat, spatial_shape, scale_start_index,
        sample_location, weights, batch_size, num_cams, num_feat,
        num_embeds, num_scale, num_pts, num_groups, num_depths);
    cudaCheckError();
}

// ==================== FP16 entry points ====================
void deformable_aggregation_forward_cuda_half(
    __half* output, const __half* mc_ms_feat, const int* spatial_shape,
    const int* scale_start_index, const __half* sample_location,
    const __half* weights,
    int batch_size, int num_cams, int num_feat, int num_embeds,
    int num_scale, int num_pts, int num_groups, cudaStream_t stream) {

    int num_kernels = batch_size * num_pts * num_embeds;
    int blocks = DIVUP(num_kernels, THREADS_PER_BLOCK);
    deformable_aggregation_kernel_f16<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        num_kernels, output, mc_ms_feat, spatial_shape, scale_start_index,
        sample_location, weights, batch_size, num_cams, num_feat,
        num_embeds, num_scale, num_pts, num_groups);
    cudaCheckError();
}

void deformable_aggregation_with_depth_forward_cuda_half(
    __half* output, const __half* mc_ms_feat, const int* spatial_shape,
    const int* scale_start_index, const __half* sample_location,
    const __half* weights,
    int batch_size, int num_cams, int num_feat, int num_embeds,
    int num_scale, int num_pts, int num_groups, int num_depths,
    cudaStream_t stream) {

    int num_kernels = batch_size * num_pts * num_embeds;
    int blocks = DIVUP(num_kernels, THREADS_PER_BLOCK);
    deformable_aggregation_depth_kernel_f16<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        num_kernels, output, mc_ms_feat, spatial_shape, scale_start_index,
        sample_location, weights, batch_size, num_cams, num_feat,
        num_embeds, num_scale, num_pts, num_groups, num_depths);
    cudaCheckError();
}
