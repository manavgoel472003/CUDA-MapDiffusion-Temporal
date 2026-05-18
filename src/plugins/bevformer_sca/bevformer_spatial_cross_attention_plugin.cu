#include <NvInfer.h>
#include <cuda_runtime.h>

#include <math.h>
#include <stdint.h>
#include <string>

using namespace nvinfer1;

static const char* PLUGIN_NAME = "BEVFormerSpatialCrossAttentionPlugin";
static const char* PLUGIN_VERSION = "1";

static constexpr int B = 1;
static constexpr int NUM_CAMS = 6;
static constexpr int NQ = 5000;
static constexpr int C = 256;
static constexpr int NUM_HEADS = 8;
static constexpr int HEAD_DIM = 32;
static constexpr int NUM_LEVELS = 3;
static constexpr int NUM_POINTS = 8;
static constexpr int NUM_Z_ANCHORS = 4;
static constexpr int NUM_VALUE = 7875;

static inline size_t align256(size_t x) {
  return (x + 255) & ~size_t(255);
}

__global__ void linear_kernel(
    const float* x,
    const float* w,
    const float* b,
    float* y,
    int rows,
    int in_features,
    int out_features) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = rows * out_features;
  if (idx >= total) return;

  int row = idx / out_features;
  int out = idx % out_features;

  float acc = b ? b[out] : 0.0f;
  const float* xrow = x + row * in_features;
  const float* wrow = w + out * in_features;

  for (int k = 0; k < in_features; ++k) {
    acc += xrow[k] * wrow[k];
  }

  y[idx] = acc;
}

__global__ void make_query_pos_kernel(
    const float* query,
    const float* query_pos,
    float* query_with_pos) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = NQ * C;
  if (idx >= total) return;

  query_with_pos[idx] = query[idx] + query_pos[idx];
}

__global__ void softmax_sca_attention_kernel(float* attn) {
  // attn layout: [NQ, NUM_HEADS, NUM_LEVELS * NUM_POINTS]
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = NQ * NUM_HEADS;
  if (idx >= total) return;

  int q = idx / NUM_HEADS;
  int h = idx % NUM_HEADS;

  int base = q * (NUM_HEADS * NUM_LEVELS * NUM_POINTS)
           + h * (NUM_LEVELS * NUM_POINTS);

  float m = attn[base];
  for (int i = 1; i < NUM_LEVELS * NUM_POINTS; ++i) {
    m = fmaxf(m, attn[base + i]);
  }

  float s = 0.0f;
  float e[NUM_LEVELS * NUM_POINTS];

  for (int i = 0; i < NUM_LEVELS * NUM_POINTS; ++i) {
    e[i] = expf(attn[base + i] - m);
    s += e[i];
  }

  float inv = 1.0f / fmaxf(s, 1e-6f);
  for (int i = 0; i < NUM_LEVELS * NUM_POINTS; ++i) {
    attn[base + i] = e[i] * inv;
  }
}

__global__ void sca_msda_dense_kernel(
    const float* value_proj,
    const float* offsets,
    const float* attn,
    const float* reference_points_cam,
    const int* spatial_shapes,
    const int* level_start_index,
    float* cam_out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = NUM_CAMS * NQ * C;
  if (idx >= total) return;

  int ch = idx % C;
  int q = (idx / C) % NQ;
  int cam = idx / (NQ * C);

  int head = ch / HEAD_DIM;
  int dim = ch % HEAD_DIM;

  float acc = 0.0f;

  for (int lvl = 0; lvl < NUM_LEVELS; ++lvl) {
    int H = spatial_shapes[lvl * 2 + 0];
    int W = spatial_shapes[lvl * 2 + 1];
    int lvl_start = level_start_index[lvl];

    for (int p = 0; p < NUM_POINTS; ++p) {
      int z = p % NUM_Z_ANCHORS;

      int off_base =
          q * (NUM_HEADS * NUM_LEVELS * NUM_POINTS * 2)
        + ((head * NUM_LEVELS + lvl) * NUM_POINTS + p) * 2;

      float off_x = offsets[off_base + 0];
      float off_y = offsets[off_base + 1];

      int ref_base =
          cam * (NQ * NUM_Z_ANCHORS * 2)
        + q * (NUM_Z_ANCHORS * 2)
        + z * 2;

      float ref_x = reference_points_cam[ref_base + 0];
      float ref_y = reference_points_cam[ref_base + 1];

      float loc_x = ref_x + off_x / float(W);
      float loc_y = ref_y + off_y / float(H);

      // Same convention used in the TSA plugin parity path.
      float sample_x = loc_x * float(W) - 0.5f;
      float sample_y = loc_y * float(H) - 0.5f;

      int attn_idx =
          q * (NUM_HEADS * NUM_LEVELS * NUM_POINTS)
        + (head * NUM_LEVELS + lvl) * NUM_POINTS
        + p;

      float a = attn[attn_idx];

      const float* vbase =
          value_proj
        + cam * NUM_VALUE * C
        + head * HEAD_DIM
        + dim;

      float v = 0.0f;

      if (!(sample_y <= -1.0f || sample_y >= H || sample_x <= -1.0f || sample_x >= W)) {
        int y_low = floorf(sample_y);
        int x_low = floorf(sample_x);
        int y_high = y_low + 1;
        int x_high = x_low + 1;

        float ly = sample_y - y_low;
        float lx = sample_x - x_low;
        float hy = 1.0f - ly;
        float hx = 1.0f - lx;

        int base_hw = lvl_start;

        if (y_low >= 0 && x_low >= 0) {
          v += vbase[(base_hw + y_low * W + x_low) * C] * hy * hx;
        }
        if (y_low >= 0 && x_high <= W - 1) {
          v += vbase[(base_hw + y_low * W + x_high) * C] * hy * lx;
        }
        if (y_high <= H - 1 && x_low >= 0) {
          v += vbase[(base_hw + y_high * W + x_low) * C] * ly * hx;
        }
        if (y_high <= H - 1 && x_high <= W - 1) {
          v += vbase[(base_hw + y_high * W + x_high) * C] * ly * lx;
        }
      }

      acc += a * v;
    }
  }

  cam_out[idx] = acc;
}

__global__ void reduce_visible_cameras_kernel(
    const float* cam_out,
    const float* bev_mask,
    float* slots) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = NQ * C;
  if (idx >= total) return;

  int ch = idx % C;
  int q = idx / C;

  float acc = 0.0f;
  float count = 0.0f;

  for (int cam = 0; cam < NUM_CAMS; ++cam) {
    bool visible = false;

    for (int z = 0; z < NUM_Z_ANCHORS; ++z) {
      int mask_idx =
          cam * (NQ * NUM_Z_ANCHORS)
        + q * NUM_Z_ANCHORS
        + z;

      if (bev_mask[mask_idx] > 0.0f) {
        visible = true;
      }
    }

    if (visible) {
      acc += cam_out[cam * NQ * C + q * C + ch];
      count += 1.0f;
    }
  }

  if (count < 1.0f) count = 1.0f;

  slots[idx] = acc / count;
}

__global__ void output_proj_residual_kernel(
    const float* slots,
    const float* w,
    const float* b,
    const float* residual,
    float* out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = NQ * C;
  if (idx >= total) return;

  int q = idx / C;
  int co = idx % C;

  float acc = b ? b[co] : 0.0f;
  const float* xrow = slots + q * C;
  const float* wrow = w + co * C;

  for (int k = 0; k < C; ++k) {
    acc += xrow[k] * wrow[k];
  }

  out[idx] = acc + residual[idx];
}

class BEVFormerSpatialCrossAttentionPlugin : public IPluginV2DynamicExt {
public:
  BEVFormerSpatialCrossAttentionPlugin() {}
  BEVFormerSpatialCrossAttentionPlugin(const void* data, size_t length) {}

  const char* getPluginType() const noexcept override { return PLUGIN_NAME; }
  const char* getPluginVersion() const noexcept override { return PLUGIN_VERSION; }
  int getNbOutputs() const noexcept override { return 1; }

  DimsExprs getOutputDimensions(
      int outputIndex,
      const DimsExprs* inputs,
      int nbInputs,
      IExprBuilder& exprBuilder) noexcept override {
    return inputs[0];  // query [1, 5000, 256]
  }

  bool supportsFormatCombination(
      int pos,
      const PluginTensorDesc* inOut,
      int nbInputs,
      int nbOutputs) noexcept override {
    const auto& desc = inOut[pos];

    if (desc.format != TensorFormat::kLINEAR) return false;

    if (pos == 6 || pos == 7) {
      return desc.type == DataType::kINT32;
    }

    return desc.type == DataType::kFLOAT;
  }

  void configurePlugin(
      const DynamicPluginTensorDesc* in,
      int nbInputs,
      const DynamicPluginTensorDesc* out,
      int nbOutputs) noexcept override {}

  size_t getWorkspaceSize(
      const PluginTensorDesc* inputs,
      int nbInputs,
      const PluginTensorDesc* outputs,
      int nbOutputs) const noexcept override {
    size_t bytes = 0;

    bytes += align256(NUM_CAMS * NUM_VALUE * C * sizeof(float)); // value_proj
    bytes += align256(NQ * C * sizeof(float));                   // query_with_pos
    bytes += align256(NQ * 384 * sizeof(float));                 // offsets
    bytes += align256(NQ * 192 * sizeof(float));                 // attention
    bytes += align256(NUM_CAMS * NQ * C * sizeof(float));        // cam_out
    bytes += align256(NQ * C * sizeof(float));                   // slots

    return bytes;
  }

  int enqueue(
      const PluginTensorDesc* inputDesc,
      const PluginTensorDesc* outputDesc,
      const void* const* inputs,
      void* const* outputs,
      void* workspace,
      cudaStream_t stream) noexcept override {
    const float* query = static_cast<const float*>(inputs[0]);
    const float* value = static_cast<const float*>(inputs[2]);
    const float* query_pos = static_cast<const float*>(inputs[3]);
    const float* reference_points_cam = static_cast<const float*>(inputs[4]);
    const float* bev_mask = static_cast<const float*>(inputs[5]);
    const int* spatial_shapes = static_cast<const int*>(inputs[6]);
    const int* level_start_index = static_cast<const int*>(inputs[7]);

    const float* value_proj_w = static_cast<const float*>(inputs[8]);
    const float* value_proj_b = static_cast<const float*>(inputs[9]);
    const float* sampling_offsets_w = static_cast<const float*>(inputs[10]);
    const float* sampling_offsets_b = static_cast<const float*>(inputs[11]);
    const float* attention_weights_w = static_cast<const float*>(inputs[12]);
    const float* attention_weights_b = static_cast<const float*>(inputs[13]);
    const float* output_proj_w = static_cast<const float*>(inputs[14]);
    const float* output_proj_b = static_cast<const float*>(inputs[15]);

    float* out = static_cast<float*>(outputs[0]);

    char* ws = static_cast<char*>(workspace);

    float* value_proj_buf = reinterpret_cast<float*>(ws);
    ws += align256(NUM_CAMS * NUM_VALUE * C * sizeof(float));

    float* query_with_pos = reinterpret_cast<float*>(ws);
    ws += align256(NQ * C * sizeof(float));

    float* offsets = reinterpret_cast<float*>(ws);
    ws += align256(NQ * 384 * sizeof(float));

    float* attn = reinterpret_cast<float*>(ws);
    ws += align256(NQ * 192 * sizeof(float));

    float* cam_out = reinterpret_cast<float*>(ws);
    ws += align256(NUM_CAMS * NQ * C * sizeof(float));

    float* slots = reinterpret_cast<float*>(ws);

    int threads = 256;

    // value_proj(value): [6*7875,256] -> [6*7875,256]
    {
      int total = NUM_CAMS * NUM_VALUE * C;
      int blocks = (total + threads - 1) / threads;
      linear_kernel<<<blocks, threads, 0, stream>>>(
          value,
          value_proj_w,
          value_proj_b,
          value_proj_buf,
          NUM_CAMS * NUM_VALUE,
          C,
          C);
    }

    // query + query_pos
    {
      int total = NQ * C;
      int blocks = (total + threads - 1) / threads;
      make_query_pos_kernel<<<blocks, threads, 0, stream>>>(
          query,
          query_pos,
          query_with_pos);
    }

    // sampling_offsets(query): [5000,256] -> [5000,384]
    {
      int total = NQ * 384;
      int blocks = (total + threads - 1) / threads;
      linear_kernel<<<blocks, threads, 0, stream>>>(
          query_with_pos,
          sampling_offsets_w,
          sampling_offsets_b,
          offsets,
          NQ,
          C,
          384);
    }

    // attention_weights(query): [5000,256] -> [5000,192]
    {
      int total = NQ * 192;
      int blocks = (total + threads - 1) / threads;
      linear_kernel<<<blocks, threads, 0, stream>>>(
          query_with_pos,
          attention_weights_w,
          attention_weights_b,
          attn,
          NQ,
          C,
          192);
    }

    // softmax over levels * points
    {
      int total = NQ * NUM_HEADS;
      int blocks = (total + threads - 1) / threads;
      softmax_sca_attention_kernel<<<blocks, threads, 0, stream>>>(attn);
    }

    // dense SCA deformable sampling for all cameras and BEV queries
    {
      int total = NUM_CAMS * NQ * C;
      int blocks = (total + threads - 1) / threads;
      sca_msda_dense_kernel<<<blocks, threads, 0, stream>>>(
          value_proj_buf,
          offsets,
          attn,
          reference_points_cam,
          spatial_shapes,
          level_start_index,
          cam_out);
    }

    // reduce visible cameras
    {
      int total = NQ * C;
      int blocks = (total + threads - 1) / threads;
      reduce_visible_cameras_kernel<<<blocks, threads, 0, stream>>>(
          cam_out,
          bev_mask,
          slots);
    }

    // output projection + residual
    {
      int total = NQ * C;
      int blocks = (total + threads - 1) / threads;
      output_proj_residual_kernel<<<blocks, threads, 0, stream>>>(
          slots,
          output_proj_w,
          output_proj_b,
          query,
          out);
    }

    return 0;
  }

  size_t getSerializationSize() const noexcept override { return 0; }
  void serialize(void* buffer) const noexcept override {}

  IPluginV2DynamicExt* clone() const noexcept override {
    auto* p = new BEVFormerSpatialCrossAttentionPlugin();
    p->setPluginNamespace(namespace_.c_str());
    return p;
  }

  void destroy() noexcept override { delete this; }
  int initialize() noexcept override { return 0; }
  void terminate() noexcept override {}

  DataType getOutputDataType(
      int index,
      const DataType* inputTypes,
      int nbInputs) const noexcept override {
    return inputTypes[0];
  }

  const char* getPluginNamespace() const noexcept override {
    return namespace_.c_str();
  }

  void setPluginNamespace(const char* pluginNamespace) noexcept override {
    namespace_ = pluginNamespace ? pluginNamespace : "";
  }

private:
  std::string namespace_;
};

class BEVFormerSpatialCrossAttentionPluginCreator : public IPluginCreator {
public:
  BEVFormerSpatialCrossAttentionPluginCreator() {
    fc_.nbFields = 0;
    fc_.fields = nullptr;
  }

  const char* getPluginName() const noexcept override { return PLUGIN_NAME; }
  const char* getPluginVersion() const noexcept override { return PLUGIN_VERSION; }
  const PluginFieldCollection* getFieldNames() noexcept override { return &fc_; }

  IPluginV2* createPlugin(
      const char* name,
      const PluginFieldCollection* fc) noexcept override {
    auto* plugin = new BEVFormerSpatialCrossAttentionPlugin();
    plugin->setPluginNamespace(namespace_.c_str());
    return plugin;
  }

  IPluginV2* deserializePlugin(
      const char* name,
      const void* serialData,
      size_t serialLength) noexcept override {
    auto* plugin = new BEVFormerSpatialCrossAttentionPlugin(serialData, serialLength);
    plugin->setPluginNamespace(namespace_.c_str());
    return plugin;
  }

  void setPluginNamespace(const char* pluginNamespace) noexcept override {
    namespace_ = pluginNamespace ? pluginNamespace : "";
  }

  const char* getPluginNamespace() const noexcept override {
    return namespace_.c_str();
  }

private:
  std::string namespace_;
  PluginFieldCollection fc_;
};

REGISTER_TENSORRT_PLUGIN(BEVFormerSpatialCrossAttentionPluginCreator);
