#include "deformable_aggregation.h"
#include <vector>

using namespace nvinfer1;
using nvinfer1::plugin::DeformableAggregationPlugin;
using nvinfer1::plugin::DeformableAggregationPluginCreator;

namespace {
static const char* PLUGIN_VERSION{"1"};
static const char* PLUGIN_NAME{"DeformableAggregationPlugin"};
}

PluginFieldCollection DeformableAggregationPluginCreator::mFC{};
std::vector<PluginField> DeformableAggregationPluginCreator::mPluginAttributes{};

void deformable_aggregation_forward_cuda(
    float* output, const float* mc_ms_feat, const int* spatial_shape,
    const int* scale_start_index, const float* sample_location,
    const float* weights, int batch_size, int num_cams, int num_feat,
    int num_embeds, int num_scale, int num_pts, int num_groups,
    cudaStream_t stream);
void deformable_aggregation_forward_cuda_half(
    __half* output, const __half* mc_ms_feat, const int* spatial_shape,
    const int* scale_start_index, const __half* sample_location,
    const __half* weights, int batch_size, int num_cams, int num_feat,
    int num_embeds, int num_scale, int num_pts, int num_groups,
    cudaStream_t stream);

DeformableAggregationPlugin::DeformableAggregationPlugin(PluginFieldCollection const& fc) noexcept {
    for (int i = 0; i < fc.nbFields; i++) {
        if (strcmp(fc.fields[i].name, "num_depths") == 0)
            mNumDepths = *static_cast<const int*>(fc.fields[i].data);
    }
}
DeformableAggregationPlugin::DeformableAggregationPlugin(const std::string name, const void* data, size_t length) : mName(name) {
    const char* d = static_cast<const char*>(data);
    mNumDepths = read<int>(d);
}
int DeformableAggregationPlugin::getNbOutputs() const noexcept { return 1; }
DimsExprs DeformableAggregationPlugin::getOutputDimensions(int index, DimsExprs const* inputs, int nbInputDims, IExprBuilder& exprBuilder) noexcept {
    DimsExprs ret;
    ret.nbDims = 3;
    ret.d[0] = inputs[0].d[0];
    ret.d[1] = inputs[3].d[1];
    ret.d[2] = mNumDepths > 0
        ? exprBuilder.operation(DimensionOperation::kSUB, *inputs[0].d[3], *exprBuilder.constant(mNumDepths))
        : inputs[0].d[3];
    return ret;
}
int DeformableAggregationPlugin::initialize() noexcept { return 0; }
void DeformableAggregationPlugin::terminate() noexcept {}
size_t DeformableAggregationPlugin::getWorkspaceSize(PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept { return 0; }

int32_t DeformableAggregationPlugin::enqueue(
    PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept {

    auto batch_size = inputDesc[0].dims.d[0];
    auto num_cams   = inputDesc[0].dims.d[1];
    auto num_feat   = inputDesc[0].dims.d[2];
    auto num_embeds = inputDesc[0].dims.d[3];
    auto num_pts    = inputDesc[3].dims.d[1];
    auto num_groups = inputDesc[4].dims.d[4];

    auto spatial_shape_dims = inputDesc[1].dims;
    int num_scale = 1;
    for (int i = 0; i < spatial_shape_dims.nbDims; i++)
        num_scale *= spatial_shape_dims.d[i];
    num_scale /= 2;

    // CRITICAL FIX: Cast INT64 spatial_shape and scale_start_index to INT32
    // The CUDA kernel reads these as int32, but ONNX/TRT provides int64
    const int* d_spatial_shape = nullptr;
    const int* d_scale_start_index = nullptr;

    if (inputDesc[1].type == DataType::kINT32) {
        d_spatial_shape = (const int*)inputs[1];
    } else {
        // INT64 → INT32: copy device→host, convert, copy host→device
        int total_ints = num_scale * 2;
        std::vector<int64_t> h64(total_ints);
        cudaMemcpy(h64.data(), inputs[1], total_ints * sizeof(int64_t), cudaMemcpyDeviceToHost);
        std::vector<int> h32(total_ints);
        for (int i = 0; i < total_ints; i++) h32[i] = (int)h64[i];
        int* d_tmp = nullptr;
        cudaMalloc((void**)&d_tmp, total_ints * sizeof(int));
        cudaMemcpy(d_tmp, h32.data(), total_ints * sizeof(int), cudaMemcpyHostToDevice);
        d_spatial_shape = d_tmp;
    }

    if (inputDesc[2].type == DataType::kINT32) {
        d_scale_start_index = (const int*)inputs[2];
    } else {
        std::vector<int64_t> h64(num_scale);
        cudaMemcpy(h64.data(), inputs[2], num_scale * sizeof(int64_t), cudaMemcpyDeviceToHost);
        std::vector<int> h32(num_scale);
        for (int i = 0; i < num_scale; i++) h32[i] = (int)h64[i];
        int* d_tmp = nullptr;
        cudaMalloc((void**)&d_tmp, num_scale * sizeof(int));
        cudaMemcpy(d_tmp, h32.data(), num_scale * sizeof(int), cudaMemcpyHostToDevice);
        d_scale_start_index = d_tmp;
    }

    if (inputDesc[0].type == DataType::kFLOAT) {
        if (mNumDepths > 0) {
            // TODO: depth variant — skip for now
        }
        deformable_aggregation_forward_cuda(
            (float*)outputs[0], (const float*)inputs[0],
            d_spatial_shape, d_scale_start_index,
            (const float*)inputs[3], (const float*)inputs[4],
            batch_size, num_cams, num_feat, num_embeds, num_scale,
            num_pts, num_groups, stream);
    } else {
        deformable_aggregation_forward_cuda_half(
            (__half*)outputs[0], (const __half*)inputs[0],
            d_spatial_shape, d_scale_start_index,
            (const __half*)inputs[3], (const __half*)inputs[4],
            batch_size, num_cams, num_feat, num_embeds, num_scale,
            num_pts, num_groups, stream);
    }

    // Cleanup temporary INT32 buffers
    if (inputDesc[1].type != DataType::kINT32) { cudaFree((void*)d_spatial_shape); }
    if (inputDesc[2].type != DataType::kINT32) { cudaFree((void*)d_scale_start_index); }

    return 0;
}

size_t DeformableAggregationPlugin::getSerializationSize() const noexcept { return sizeof(int); }
void DeformableAggregationPlugin::serialize(void* buffer) const noexcept { char* d = static_cast<char*>(buffer); write(d, mNumDepths); }

bool DeformableAggregationPlugin::supportsFormatCombination(int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept {
    if (inOut[pos].format == TensorFormat::kLINEAR) {
        if (pos == 1 || pos == 2) {
            return inOut[pos].type == DataType::kINT32 || inOut[pos].type == DataType::kINT64;
        }
        bool fp_ok = (inOut[pos].type == DataType::kFLOAT) || (inOut[pos].type == DataType::kHALF);
        return fp_ok && (inOut[pos].type == inOut[0].type);
    }
    return false;
}
DataType DeformableAggregationPlugin::getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const noexcept { return inputTypes[0]; }
const char* DeformableAggregationPlugin::getPluginType() const noexcept { return PLUGIN_NAME; }
const char* DeformableAggregationPlugin::getPluginVersion() const noexcept { return PLUGIN_VERSION; }
void DeformableAggregationPlugin::destroy() noexcept { delete this; }
IPluginV2DynamicExt* DeformableAggregationPlugin::clone() const noexcept {
    auto* plugin = new DeformableAggregationPlugin(*this);
    plugin->setPluginNamespace(mPluginNamespace);
    return plugin;
}
void DeformableAggregationPlugin::setPluginNamespace(const char* pluginNamespace) noexcept { mPluginNamespace = pluginNamespace; }
const char* DeformableAggregationPlugin::getPluginNamespace() const noexcept { return mPluginNamespace; }
void DeformableAggregationPlugin::configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInput, DynamicPluginTensorDesc const* out, int32_t nbOutput) noexcept {}

DeformableAggregationPluginCreator::DeformableAggregationPluginCreator() {
    setupPluginAttributes(mPluginAttributes);
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}
void DeformableAggregationPluginCreator::setupPluginAttributes(std::vector<PluginField>& attributes) {
    attributes.clear();
    attributes.emplace_back(PluginField("num_depths", nullptr, PluginFieldType::kINT32, 1));
}
const char* DeformableAggregationPluginCreator::getPluginName() const noexcept { return PLUGIN_NAME; }
const char* DeformableAggregationPluginCreator::getPluginVersion() const noexcept { return PLUGIN_VERSION; }
void DeformableAggregationPluginCreator::setPluginNamespace(const char* pluginNamespace) noexcept { mNamespace = pluginNamespace; }
const char* DeformableAggregationPluginCreator::getPluginNamespace() const noexcept { return mNamespace.c_str(); }
const PluginFieldCollection* DeformableAggregationPluginCreator::getFieldNames() noexcept { return &mFC; }
IPluginV2* DeformableAggregationPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept {
    auto plugin = new DeformableAggregationPlugin(*fc);
    plugin->setPluginNamespace(mNamespace.c_str());
    mFC = *fc;
    return plugin;
}
IPluginV2* DeformableAggregationPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept {
    return new DeformableAggregationPlugin(name, serialData, serialLength);
}
