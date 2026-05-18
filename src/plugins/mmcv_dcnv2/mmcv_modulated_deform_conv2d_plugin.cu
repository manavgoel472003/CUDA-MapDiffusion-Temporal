#include <NvInfer.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#include <cassert>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using namespace nvinfer1;

#define CHECK_CUDA(status) do {                                  \
  cudaError_t s = status;                                         \
  if (s != cudaSuccess) {                                         \
    printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__,          \
           cudaGetErrorString(s));                                \
  }                                                               \
} while (0)

static const char* PLUGIN_NAME = "MMCVModulatedDeformConv2d";
static const char* PLUGIN_VERSION = "1";

static inline int64_t volume(const Dims& d) {
  int64_t v = 1;
  for (int i = 0; i < d.nbDims; ++i) v *= d.d[i];
  return v;
}

template <typename scalar_t>
__device__ __forceinline__ float scalar_to_float(scalar_t v) {
  return static_cast<float>(v);
}

template <>
__device__ __forceinline__ float scalar_to_float<__half>(__half v) {
  return __half2float(v);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t float_to_scalar(float v) {
  return static_cast<scalar_t>(v);
}

template <>
__device__ __forceinline__ __half float_to_scalar<__half>(float v) {
  return __float2half(v);
}

template <typename scalar_t>
__device__ float bilinear_sample(
    const scalar_t* input,
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

  if (y_low >= 0 && x_low >= 0) v1 = scalar_to_float(input[y_low * W + x_low]);
  if (y_low >= 0 && x_high <= W - 1) v2 = scalar_to_float(input[y_low * W + x_high]);
  if (y_high <= H - 1 && x_low >= 0) v3 = scalar_to_float(input[y_high * W + x_low]);
  if (y_high <= H - 1 && x_high <= W - 1) v4 = scalar_to_float(input[y_high * W + x_high]);

  float w1 = hy * hx;
  float w2 = hy * lx;
  float w3 = ly * hx;
  float w4 = ly * lx;

  return v1 * w1 + v2 * w2 + v3 * w3 + v4 * w4;
}

template <typename scalar_t>
__global__ void mmcv_dcnv2_forward_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ offset,
    const scalar_t* __restrict__ mask,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    int N,
    int C_in,
    int H_in,
    int W_in,
    int C_out,
    int H_out,
    int W_out,
    int kH,
    int kW,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    int groups,
    int deform_groups) {

  int64_t total = (int64_t)N * C_out * H_out * W_out;
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= total) return;

  int ow = idx % W_out;
  int oh = (idx / W_out) % H_out;
  int oc = (idx / (W_out * H_out)) % C_out;
  int n = idx / ((int64_t)W_out * H_out * C_out);

  int c_per_group = C_in / groups;
  int out_per_group = C_out / groups;
  int group_id = oc / out_per_group;
  int ic_start = group_id * c_per_group;
  int ic_end = ic_start + c_per_group;

  float acc = 0.0f;

  for (int ic = ic_start; ic < ic_end; ++ic) {
    int deform_group_id = (ic - ic_start) * deform_groups / c_per_group;

    for (int kh = 0; kh < kH; ++kh) {
      for (int kw = 0; kw < kW; ++kw) {
        int kidx = kh * kW + kw;

        int offset_c_base = deform_group_id * 2 * kH * kW + 2 * kidx;
        int mask_c = deform_group_id * kH * kW + kidx;

        int offset_index_y = ((n * (2 * deform_groups * kH * kW) + offset_c_base) * H_out + oh) * W_out + ow;
        int offset_index_x = ((n * (2 * deform_groups * kH * kW) + offset_c_base + 1) * H_out + oh) * W_out + ow;
        int mask_index = ((n * (deform_groups * kH * kW) + mask_c) * H_out + oh) * W_out + ow;

        float off_y = scalar_to_float(offset[offset_index_y]);
        float off_x = scalar_to_float(offset[offset_index_x]);
        float m = scalar_to_float(mask[mask_index]);

        float y = oh * stride_h - pad_h + kh * dilation_h + off_y;
        float x = ow * stride_w - pad_w + kw * dilation_w + off_x;

        const scalar_t* input_ptr = input + ((n * C_in + ic) * H_in * W_in);
        float val = bilinear_sample(input_ptr, H_in, W_in, y, x);

        int weight_index = ((oc * c_per_group + (ic - ic_start)) * kH + kh) * kW + kw;
        float w = scalar_to_float(weight[weight_index]);

        acc += val * w * m;
      }
    }
  }

  output[idx] = float_to_scalar<scalar_t>(acc);
}



// -----------------------------------------------------------------------------
// Experimental FP16 im2col path for DCNv2.
// Layout:
//   columns is row-major [K, M], where:
//     K = C_in * kH * kW
//     M = H_out * W_out
//   For each batch b, cuBLAS computes:
//     output_b[C_out, M] = weight[C_out, K] x columns[K, M]
// using the row-major-to-column-major cuBLAS trick.
// -----------------------------------------------------------------------------
template <typename scalar_t>
__global__ void mmcv_dcnv2_im2col_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ offset,
    const scalar_t* __restrict__ mask,
    scalar_t* __restrict__ columns,
    int C_in,
    int H_in,
    int W_in,
    int H_out,
    int W_out,
    int kH,
    int kW,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    int deform_groups) {
  int M = H_out * W_out;
  int K = C_in * kH * kW;
  int total = K * M;

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;

  int m = idx % M;
  int k = idx / M;

  int out_y = m / W_out;
  int out_x = m % W_out;

  int kw = k % kW;
  int kh = (k / kW) % kH;
  int c = k / (kH * kW);

  int dg = c / max(1, C_in / deform_groups);

  int offset_base = ((2 * (dg * kH * kW + kh * kW + kw)) * H_out + out_y) * W_out + out_x;
  int mask_base = ((dg * kH * kW + kh * kW + kw) * H_out + out_y) * W_out + out_x;

  float off_y = scalar_to_float(offset[offset_base]);
  float off_x = scalar_to_float(offset[offset_base + H_out * W_out]);
  float mval = scalar_to_float(mask[mask_base]);

  float y = out_y * stride_h - pad_h + kh * dilation_h + off_y;
  float x = out_x * stride_w - pad_w + kw * dilation_w + off_x;

  const scalar_t* input_ptr = input + c * H_in * W_in;
  float sampled = bilinear_sample(input_ptr, H_in, W_in, y, x);

  columns[k * M + m] = float_to_scalar<scalar_t>(sampled * mval);
}

class MMCVModulatedDeformConv2dPlugin : public IPluginV2DynamicExt {
public:
  MMCVModulatedDeformConv2dPlugin(
      int deform_groups,
      int groups,
      int stride_h,
      int stride_w,
      int pad_h,
      int pad_w,
      int dilation_h,
      int dilation_w)
      : deform_groups_(deform_groups),
        groups_(groups),
        stride_h_(stride_h),
        stride_w_(stride_w),
        pad_h_(pad_h),
        pad_w_(pad_w),
        dilation_h_(dilation_h),
        dilation_w_(dilation_w) {}

  MMCVModulatedDeformConv2dPlugin(const void* data, size_t length) {
    const int* d = reinterpret_cast<const int*>(data);
    deform_groups_ = d[0];
    groups_ = d[1];
    stride_h_ = d[2];
    stride_w_ = d[3];
    pad_h_ = d[4];
    pad_w_ = d[5];
    dilation_h_ = d[6];
    dilation_w_ = d[7];
  }

  const char* getPluginType() const noexcept override { return PLUGIN_NAME; }
  const char* getPluginVersion() const noexcept override { return PLUGIN_VERSION; }
  int getNbOutputs() const noexcept override { return 1; }

  DimsExprs getOutputDimensions(
      int outputIndex,
      const DimsExprs* inputs,
      int nbInputs,
      IExprBuilder& exprBuilder) noexcept override {
    // inputs:
    // 0 input  [N,Cin,H,W]
    // 1 offset [N,2*kH*kW*dg,Hout,Wout]
    // 2 mask   [N,kH*kW*dg,Hout,Wout]
    // 3 weight [Cout,Cin/groups,kH,kW]
    DimsExprs out;
    out.nbDims = 4;
    out.d[0] = inputs[0].d[0];
    out.d[1] = inputs[3].d[0];
    out.d[2] = inputs[1].d[2];
    out.d[3] = inputs[1].d[3];
    return out;
  }

  bool supportsFormatCombination(
      int pos,
      const PluginTensorDesc* inOut,
      int nbInputs,
      int nbOutputs) noexcept override {
    const PluginTensorDesc& desc = inOut[pos];

    if (desc.format != TensorFormat::kLINEAR) return false;

    // Support FP32 and FP16. Require all plugin tensors to share input0 dtype.
    if (pos == 0) {
      return desc.type == DataType::kFLOAT || desc.type == DataType::kHALF;
    }
    return desc.type == inOut[0].type;
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
    const Dims& x = inputs[0].dims;
    const Dims& off = inputs[1].dims;
    const Dims& w = inputs[3].dims;

    int N = x.d[0];
    int C_in = x.d[1];
    int H_out = off.d[2];
    int W_out = off.d[3];
    int kH = w.d[2];
    int kW = w.d[3];

    // Workspace for one batch at a time: columns[K, M].
    // K = C_in * kH * kW, M = H_out * W_out.
    size_t K = static_cast<size_t>(C_in) * kH * kW;
    size_t M = static_cast<size_t>(H_out) * W_out;
    size_t elem_size = (inputs[0].type == DataType::kFLOAT) ? sizeof(float) : sizeof(__half);
    size_t bytes = K * M * elem_size;

    // 256-byte alignment padding.
    bytes = ((bytes + 255) / 256) * 256;
    return bytes;
  }


  int enqueue(
      const PluginTensorDesc* inputDesc,
      const PluginTensorDesc* outputDesc,
      const void* const* inputs,
      void* const* outputs,
      void* workspace,
      cudaStream_t stream) noexcept override {
    const Dims& x = inputDesc[0].dims;
    const Dims& off = inputDesc[1].dims;
    const Dims& w = inputDesc[3].dims;

    int N = x.d[0];
    int C_in = x.d[1];
    int H_in = x.d[2];
    int W_in = x.d[3];

    int H_out = off.d[2];
    int W_out = off.d[3];

    int C_out = w.d[0];
    int kH = w.d[2];
    int kW = w.d[3];

    int64_t total = (int64_t)N * C_out * H_out * W_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    // FP32 im2col+GEMM path for better accuracy experiments.
    // Fallback to direct kernel when groups != 1.
    if (inputDesc[0].type == DataType::kFLOAT) {
      if (groups_ != 1) {
        mmcv_dcnv2_forward_kernel<float><<<blocks, threads, 0, stream>>>(
            static_cast<const float*>(inputs[0]),
            static_cast<const float*>(inputs[1]),
            static_cast<const float*>(inputs[2]),
            static_cast<const float*>(inputs[3]),
            static_cast<float*>(outputs[0]),
            N,
            C_in,
            H_in,
            W_in,
            C_out,
            H_out,
            W_out,
            kH,
            kW,
            stride_h_,
            stride_w_,
            pad_h_,
            pad_w_,
            dilation_h_,
            dilation_w_,
            groups_,
            deform_groups_);
        return 0;
      }

      const float* input = static_cast<const float*>(inputs[0]);
      const float* offset = static_cast<const float*>(inputs[1]);
      const float* mask = static_cast<const float*>(inputs[2]);
      const float* weight = static_cast<const float*>(inputs[3]);
      float* output = static_cast<float*>(outputs[0]);
      float* columns = static_cast<float*>(workspace);

      int M = H_out * W_out;
      int K = C_in * kH * kW;

      int col_total = K * M;
      int col_blocks = (col_total + threads - 1) / threads;

      cublasHandle_t handle;
      cublasStatus_t stat = cublasCreate(&handle);
      if (stat != CUBLAS_STATUS_SUCCESS) return 1;
      stat = cublasSetStream(handle, stream);
      if (stat != CUBLAS_STATUS_SUCCESS) {
        cublasDestroy(handle);
        return 1;
      }

      float alpha = 1.0f;
      float beta = 0.0f;

      for (int b = 0; b < N; ++b) {
        const float* input_b = input + static_cast<size_t>(b) * C_in * H_in * W_in;
        const float* offset_b = offset + static_cast<size_t>(b) * (2 * deform_groups_ * kH * kW) * H_out * W_out;
        const float* mask_b = mask + static_cast<size_t>(b) * (deform_groups_ * kH * kW) * H_out * W_out;
        float* output_b = output + static_cast<size_t>(b) * C_out * H_out * W_out;

        mmcv_dcnv2_im2col_kernel<float><<<col_blocks, threads, 0, stream>>>(
            input_b,
            offset_b,
            mask_b,
            columns,
            C_in,
            H_in,
            W_in,
            H_out,
            W_out,
            kH,
            kW,
            stride_h_,
            stride_w_,
            pad_h_,
            pad_w_,
            dilation_h_,
            dilation_w_,
            deform_groups_);

        // Row-major output [C_out, M] = weight [C_out, K] x columns [K, M].
        // cuBLAS column-major trick:
        //   C_col[M, C_out] = columns_col[M, K] * weight_col[K, C_out]
        stat = cublasSgemm(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            M,
            C_out,
            K,
            &alpha,
            columns,
            M,
            weight,
            K,
            &beta,
            output_b,
            M);

        if (stat != CUBLAS_STATUS_SUCCESS) {
          cublasDestroy(handle);
          return 1;
        }
      }

      cublasDestroy(handle);
      return 0;
    }

    if (inputDesc[0].type != DataType::kHALF) {
      return 1;
    }

    // Experimental v1: im2col+GEMM only for groups=1.
    // If groups are not 1, use the known-correct direct half path.
    if (groups_ != 1) {
      mmcv_dcnv2_forward_kernel<__half><<<blocks, threads, 0, stream>>>(
          static_cast<const __half*>(inputs[0]),
          static_cast<const __half*>(inputs[1]),
          static_cast<const __half*>(inputs[2]),
          static_cast<const __half*>(inputs[3]),
          static_cast<__half*>(outputs[0]),
          N,
          C_in,
          H_in,
          W_in,
          C_out,
          H_out,
          W_out,
          kH,
          kW,
          stride_h_,
          stride_w_,
          pad_h_,
          pad_w_,
          dilation_h_,
          dilation_w_,
          groups_,
          deform_groups_);
      return 0;
    }

    const __half* input = static_cast<const __half*>(inputs[0]);
    const __half* offset = static_cast<const __half*>(inputs[1]);
    const __half* mask = static_cast<const __half*>(inputs[2]);
    const __half* weight = static_cast<const __half*>(inputs[3]);
    __half* output = static_cast<__half*>(outputs[0]);
    __half* columns = static_cast<__half*>(workspace);

    int M = H_out * W_out;
    int K = C_in * kH * kW;

    int col_total = K * M;
    int col_blocks = (col_total + threads - 1) / threads;

    cublasHandle_t handle;
    cublasStatus_t stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) return 1;
    stat = cublasSetStream(handle, stream);
    if (stat != CUBLAS_STATUS_SUCCESS) {
      cublasDestroy(handle);
      return 1;
    }

    float alpha = 1.0f;
    float beta = 0.0f;

    for (int b = 0; b < N; ++b) {
      const __half* input_b = input + static_cast<size_t>(b) * C_in * H_in * W_in;
      const __half* offset_b = offset + static_cast<size_t>(b) * (2 * deform_groups_ * kH * kW) * H_out * W_out;
      const __half* mask_b = mask + static_cast<size_t>(b) * (deform_groups_ * kH * kW) * H_out * W_out;
      __half* output_b = output + static_cast<size_t>(b) * C_out * H_out * W_out;

      mmcv_dcnv2_im2col_kernel<__half><<<col_blocks, threads, 0, stream>>>(
          input_b,
          offset_b,
          mask_b,
          columns,
          C_in,
          H_in,
          W_in,
          H_out,
          W_out,
          kH,
          kW,
          stride_h_,
          stride_w_,
          pad_h_,
          pad_w_,
          dilation_h_,
          dilation_w_,
          deform_groups_);

      // Row-major output [C_out, M] = weight [C_out, K] x columns [K, M].
      // cuBLAS is column-major, so compute C_col[M, C_out] = columns_col[M, K] * weight_col[K, C_out].
      stat = cublasGemmEx(
          handle,
          CUBLAS_OP_N,
          CUBLAS_OP_N,
          M,
          C_out,
          K,
          &alpha,
          columns,
          CUDA_R_16F,
          M,
          weight,
          CUDA_R_16F,
          K,
          &beta,
          output_b,
          CUDA_R_16F,
          M,
          CUDA_R_32F,
          CUBLAS_GEMM_DEFAULT_TENSOR_OP);

      if (stat != CUBLAS_STATUS_SUCCESS) {
        cublasDestroy(handle);
        return 1;
      }
    }

    cublasDestroy(handle);
    return 0;
  }


  size_t getSerializationSize() const noexcept override {
    return 8 * sizeof(int);
  }

  void serialize(void* buffer) const noexcept override {
    int* d = reinterpret_cast<int*>(buffer);
    d[0] = deform_groups_;
    d[1] = groups_;
    d[2] = stride_h_;
    d[3] = stride_w_;
    d[4] = pad_h_;
    d[5] = pad_w_;
    d[6] = dilation_h_;
    d[7] = dilation_w_;
  }

  IPluginV2DynamicExt* clone() const noexcept override {
    auto* p = new MMCVModulatedDeformConv2dPlugin(
        deform_groups_,
        groups_,
        stride_h_,
        stride_w_,
        pad_h_,
        pad_w_,
        dilation_h_,
        dilation_w_);
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

  const char* getPluginNamespace() const noexcept override { return namespace_.c_str(); }
  void setPluginNamespace(const char* pluginNamespace) noexcept override {
    namespace_ = pluginNamespace ? pluginNamespace : "";
  }

private:
  int deform_groups_{1};
  int groups_{1};
  int stride_h_{1};
  int stride_w_{1};
  int pad_h_{1};
  int pad_w_{1};
  int dilation_h_{1};
  int dilation_w_{1};
  std::string namespace_;
};

class MMCVModulatedDeformConv2dPluginCreator : public IPluginCreator {
public:
  MMCVModulatedDeformConv2dPluginCreator() {
    fields_.clear();
    fields_.emplace_back(PluginField{"deform_groups", nullptr, PluginFieldType::kINT32, 1});
    fields_.emplace_back(PluginField{"groups", nullptr, PluginFieldType::kINT32, 1});
    fields_.emplace_back(PluginField{"stride", nullptr, PluginFieldType::kINT32, 2});
    fields_.emplace_back(PluginField{"padding", nullptr, PluginFieldType::kINT32, 2});
    fields_.emplace_back(PluginField{"dilation", nullptr, PluginFieldType::kINT32, 2});

    fc_.nbFields = fields_.size();
    fc_.fields = fields_.data();
  }

  const char* getPluginName() const noexcept override { return PLUGIN_NAME; }
  const char* getPluginVersion() const noexcept override { return PLUGIN_VERSION; }
  const PluginFieldCollection* getFieldNames() noexcept override { return &fc_; }

  IPluginV2* createPlugin(
      const char* name,
      const PluginFieldCollection* fc) noexcept override {
    int deform_groups = 1;
    int groups = 1;
    int stride_h = 1;
    int stride_w = 1;
    int pad_h = 1;
    int pad_w = 1;
    int dilation_h = 1;
    int dilation_w = 1;

    for (int i = 0; i < fc->nbFields; ++i) {
      std::string fname(fc->fields[i].name);
      const void* data = fc->fields[i].data;

      if (fname == "deform_groups") {
        deform_groups = *static_cast<const int*>(data);
      } else if (fname == "groups") {
        groups = *static_cast<const int*>(data);
      } else if (fname == "stride") {
        const int* v = static_cast<const int*>(data);
        stride_h = v[0];
        stride_w = v[1];
      } else if (fname == "padding") {
        const int* v = static_cast<const int*>(data);
        pad_h = v[0];
        pad_w = v[1];
      } else if (fname == "dilation") {
        const int* v = static_cast<const int*>(data);
        dilation_h = v[0];
        dilation_w = v[1];
      }
    }

    auto* plugin = new MMCVModulatedDeformConv2dPlugin(
        deform_groups,
        groups,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w);
    plugin->setPluginNamespace(namespace_.c_str());
    return plugin;
  }

  IPluginV2* deserializePlugin(
      const char* name,
      const void* serialData,
      size_t serialLength) noexcept override {
    auto* plugin = new MMCVModulatedDeformConv2dPlugin(serialData, serialLength);
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
  std::vector<PluginField> fields_;
  PluginFieldCollection fc_;
};

REGISTER_TENSORRT_PLUGIN(MMCVModulatedDeformConv2dPluginCreator);
