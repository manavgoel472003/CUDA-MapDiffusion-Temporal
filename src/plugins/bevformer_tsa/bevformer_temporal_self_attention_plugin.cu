#include <NvInfer.h>
#include <cuda_runtime.h>

#include <math.h>
#include <stdint.h>
#include <string>

using namespace nvinfer1;

static const char* PLUGIN_NAME = "BEVFormerTemporalSelfAttentionPlugin";
static const char* PLUGIN_VERSION = "1";

static constexpr int B = 1;
static constexpr int NQ = 5000;
static constexpr int C = 256;
static constexpr int NUM_QUEUE = 2;
static constexpr int NUM_HEADS = 8;
static constexpr int HEAD_DIM = 32;
static constexpr int NUM_LEVELS = 1;
static constexpr int NUM_POINTS = 4;

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

__global__ void make_query_cat_kernel(
    const float* query,
    const float* value,
    const float* query_pos,
    float* query_cat) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = NQ * 512;
  if (idx >= total) return;

  int q = idx / 512;
  int c512 = idx % 512;

  if (c512 < 256) {
    // First half = value[:bs]
    query_cat[idx] = value[q * C + c512];
  } else {
    int c = c512 - 256;
    query_cat[idx] = query[q * C + c] + query_pos[q * C + c];
  }
}

__global__ void softmax_attention_kernel(float* attn) {
  // attn layout: [NQ, NUM_HEADS * NUM_QUEUE * NUM_POINTS]
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = NQ * NUM_HEADS * NUM_QUEUE;
  if (idx >= total) return;

  int q = idx / (NUM_HEADS * NUM_QUEUE);
  int rem = idx % (NUM_HEADS * NUM_QUEUE);
  int h = rem / NUM_QUEUE;
  int queue = rem % NUM_QUEUE;

  int base = q * (NUM_HEADS * NUM_QUEUE * NUM_POINTS)
           + ((h * NUM_QUEUE + queue) * NUM_POINTS);

  float m = attn[base];
  for (int p = 1; p < NUM_POINTS; ++p) {
    m = fmaxf(m, attn[base + p]);
  }

  float s = 0.0f;
  float e[NUM_POINTS];

  for (int p = 0; p < NUM_POINTS; ++p) {
    e[p] = expf(attn[base + p] - m);
    s += e[p];
  }

  float inv = 1.0f / fmaxf(s, 1e-6f);
  for (int p = 0; p < NUM_POINTS; ++p) {
    attn[base + p] = e[p] * inv;
  }
}

__device__ float bilinear_sample(
    const float* value,
    int H,
    int W,
    float y,
    float x) {
  if (y <= -1.0f || y >= H || x <= -1.0f || x >= W) {
    return 0.0f;
  }

  int y_low = floorf(y);
  int x_low = floorf(x);
  int y_high = y_low + 1;
  int x_high = x_low + 1;

  float ly = y - y_low;
  float lx = x - x_low;
  float hy = 1.0f - ly;
  float hx = 1.0f - lx;

  float v1 = 0.0f;
  float v2 = 0.0f;
  float v3 = 0.0f;
  float v4 = 0.0f;

  if (y_low >= 0 && x_low >= 0) {
    v1 = value[y_low * W + x_low];
  }
  if (y_low >= 0 && x_high <= W - 1) {
    v2 = value[y_low * W + x_high];
  }
  if (y_high <= H - 1 && x_low >= 0) {
    v3 = value[y_high * W + x_low];
  }
  if (y_high <= H - 1 && x_high <= W - 1) {
    v4 = value[y_high * W + x_high];
  }

  return v1 * hy * hx + v2 * hy * lx + v3 * ly * hx + v4 * ly * lx;
}

__global__ void tsa_msda_kernel(
    const float* value_proj,
    const float* offsets,
    const float* attn,
    const float* reference_points,
    const int* spatial_shapes,
    float* tsa_out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = NUM_QUEUE * NQ * C;
  if (idx >= total) return;

  int c = idx % C;
  int q = (idx / C) % NQ;
  int queue = idx / (NQ * C);

  int h = c / HEAD_DIM;
  int d = c % HEAD_DIM;

  int H = spatial_shapes[0];
  int W = spatial_shapes[1];

  float acc = 0.0f;

  for (int p = 0; p < NUM_POINTS; ++p) {
    int off_base =
        q * (NUM_HEADS * NUM_QUEUE * NUM_LEVELS * NUM_POINTS * 2)
      + (((h * NUM_QUEUE + queue) * NUM_LEVELS + 0) * NUM_POINTS + p) * 2;

    float off_x = offsets[off_base + 0];
    float off_y = offsets[off_base + 1];

    int ref_base = ((queue * NQ + q) * NUM_LEVELS + 0) * 2;
    float ref_x = reference_points[ref_base + 0];
    float ref_y = reference_points[ref_base + 1];

    float loc_x = ref_x + off_x / float(W);
    float loc_y = ref_y + off_y / float(H);

    // MMCV/MSDeformAttn convention: normalized [0,1] -> pixel with -0.5 shift.
    float sample_x = loc_x * float(W) - 0.5f;
    float sample_y = loc_y * float(H) - 0.5f;

    int attn_idx =
        q * (NUM_HEADS * NUM_QUEUE * NUM_POINTS)
      + (h * NUM_QUEUE + queue) * NUM_POINTS + p;

    float a = attn[attn_idx];

    const float* vbase = value_proj + queue * NQ * C + h * HEAD_DIM + d;

    // value_proj flattened as [queue, flat_hw, C]
    // bilinear over flat_hw, channel fixed.
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

      if (y_low >= 0 && x_low >= 0) {
        v += vbase[(y_low * W + x_low) * C] * hy * hx;
      }
      if (y_low >= 0 && x_high <= W - 1) {
        v += vbase[(y_low * W + x_high) * C] * hy * lx;
      }
      if (y_high <= H - 1 && x_low >= 0) {
        v += vbase[(y_high * W + x_low) * C] * ly * hx;
      }
      if (y_high <= H - 1 && x_high <= W - 1) {
        v += vbase[(y_high * W + x_high) * C] * ly * lx;
      }
    }

    acc += a * v;
  }

  tsa_out[idx] = acc;
}

__global__ void fuse_queue_kernel(
    const float* tsa_out,
    float* fused) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = NQ * C;
  if (idx >= total) return;

  fused[idx] = 0.5f * (tsa_out[idx] + tsa_out[NQ * C + idx]);
}

__global__ void output_proj_residual_kernel(
    const float* fused,
    const float* w,
    const float* b,
    const float* identity,
    float* out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = NQ * C;
  if (idx >= total) return;

  int q = idx / C;
  int co = idx % C;

  float acc = b ? b[co] : 0.0f;
  const float* xrow = fused + q * C;
  const float* wrow = w + co * C;

  for (int k = 0; k < C; ++k) {
    acc += xrow[k] * wrow[k];
  }

  out[idx] = acc + identity[idx];
}

class BEVFormerTemporalSelfAttentionPlugin : public IPluginV2DynamicExt {
public:
  BEVFormerTemporalSelfAttentionPlugin() {}
  BEVFormerTemporalSelfAttentionPlugin(const void* data, size_t length) {}

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

    // 6 spatial_shapes and 7 level_start_index are INT32.
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

    bytes += align256(NUM_QUEUE * NQ * C * sizeof(float));       // value_proj
    bytes += align256(NQ * 512 * sizeof(float));                 // query_cat
    bytes += align256(NQ * 128 * sizeof(float));                 // offsets
    bytes += align256(NQ * 64 * sizeof(float));                  // attention
    bytes += align256(NUM_QUEUE * NQ * C * sizeof(float));       // tsa_out
    bytes += align256(NQ * C * sizeof(float));                   // fused

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
    const float* identity = static_cast<const float*>(inputs[3]);
    const float* query_pos = static_cast<const float*>(inputs[4]);
    const float* reference_points = static_cast<const float*>(inputs[5]);
    const int* spatial_shapes = static_cast<const int*>(inputs[6]);

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
    ws += align256(NUM_QUEUE * NQ * C * sizeof(float));

    float* query_cat = reinterpret_cast<float*>(ws);
    ws += align256(NQ * 512 * sizeof(float));

    float* offsets = reinterpret_cast<float*>(ws);
    ws += align256(NQ * 128 * sizeof(float));

    float* attn = reinterpret_cast<float*>(ws);
    ws += align256(NQ * 64 * sizeof(float));

    float* tsa_out = reinterpret_cast<float*>(ws);
    ws += align256(NUM_QUEUE * NQ * C * sizeof(float));

    float* fused = reinterpret_cast<float*>(ws);

    int threads = 256;

    // value_proj(value): [2,5000,256] -> [2,5000,256]
    {
      int total = NUM_QUEUE * NQ * C;
      int blocks = (total + threads - 1) / threads;
      linear_kernel<<<blocks, threads, 0, stream>>>(
          value,
          value_proj_w,
          value_proj_b,
          value_proj_buf,
          NUM_QUEUE * NQ,
          C,
          C);
    }

    // query_cat = concat(value[:1], query + query_pos): [5000,512]
    {
      int total = NQ * 512;
      int blocks = (total + threads - 1) / threads;
      make_query_cat_kernel<<<blocks, threads, 0, stream>>>(
          query,
          value,
          query_pos,
          query_cat);
    }

    // sampling_offsets(query_cat): [5000,512] -> [5000,128]
    {
      int total = NQ * 128;
      int blocks = (total + threads - 1) / threads;
      linear_kernel<<<blocks, threads, 0, stream>>>(
          query_cat,
          sampling_offsets_w,
          sampling_offsets_b,
          offsets,
          NQ,
          512,
          128);
    }

    // attention_weights(query_cat): [5000,512] -> [5000,64]
    {
      int total = NQ * 64;
      int blocks = (total + threads - 1) / threads;
      linear_kernel<<<blocks, threads, 0, stream>>>(
          query_cat,
          attention_weights_w,
          attention_weights_b,
          attn,
          NQ,
          512,
          64);
    }

    // softmax over points dim
    {
      int total = NQ * NUM_HEADS * NUM_QUEUE;
      int blocks = (total + threads - 1) / threads;
      softmax_attention_kernel<<<blocks, threads, 0, stream>>>(attn);
    }

    // deformable sampling
    {
      int total = NUM_QUEUE * NQ * C;
      int blocks = (total + threads - 1) / threads;
      tsa_msda_kernel<<<blocks, threads, 0, stream>>>(
          value_proj_buf,
          offsets,
          attn,
          reference_points,
          spatial_shapes,
          tsa_out);
    }

    // mean over queue
    {
      int total = NQ * C;
      int blocks = (total + threads - 1) / threads;
      fuse_queue_kernel<<<blocks, threads, 0, stream>>>(
          tsa_out,
          fused);
    }

    // output_proj + residual
    {
      int total = NQ * C;
      int blocks = (total + threads - 1) / threads;
      output_proj_residual_kernel<<<blocks, threads, 0, stream>>>(
          fused,
          output_proj_w,
          output_proj_b,
          identity,
          out);
    }

    return 0;
  }

  size_t getSerializationSize() const noexcept override { return 0; }
  void serialize(void* buffer) const noexcept override {}

  IPluginV2DynamicExt* clone() const noexcept override {
    auto* p = new BEVFormerTemporalSelfAttentionPlugin();
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

class BEVFormerTemporalSelfAttentionPluginCreator : public IPluginCreator {
public:
  BEVFormerTemporalSelfAttentionPluginCreator() {
    fc_.nbFields = 0;
    fc_.fields = nullptr;
  }

  const char* getPluginName() const noexcept override { return PLUGIN_NAME; }
  const char* getPluginVersion() const noexcept override { return PLUGIN_VERSION; }
  const PluginFieldCollection* getFieldNames() noexcept override { return &fc_; }

  IPluginV2* createPlugin(
      const char* name,
      const PluginFieldCollection* fc) noexcept override {
    auto* plugin = new BEVFormerTemporalSelfAttentionPlugin();
    plugin->setPluginNamespace(namespace_.c_str());
    return plugin;
  }

  IPluginV2* deserializePlugin(
      const char* name,
      const void* serialData,
      size_t serialLength) noexcept override {
    auto* plugin = new BEVFormerTemporalSelfAttentionPlugin(serialData, serialLength);
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

REGISTER_TENSORRT_PLUGIN(BEVFormerTemporalSelfAttentionPluginCreator);
