#include "bevfusion/mapdiffusion-routeA-adapter.hpp"

#include <cuda_fp16.h>

#include "common/dtype.hpp"
#include "common/launch.cuh"

namespace bevfusion {
namespace routea {

static constexpr int kSrcH = 180;
static constexpr int kSrcW = 180;

static constexpr int kDstC = 256;
static constexpr int kDstH = 50;
static constexpr int kDstW = 100;

static constexpr int kYOffset = (kSrcH - kDstH) / 2;  // 65
static constexpr int kXOffset = (kSrcW - kDstW) / 2;  // 40

static __global__ void adapt_bevfusion_to_mapdiffusion_kernel(
    int numel,
    const nvtype::half* fusion_feature,
    nvtype::half* mapdiffusion_bev) {
  int idx = cuda_linear_index;
  if (idx >= numel) return;

  int x = idx % kDstW;
  int t = idx / kDstW;
  int y = t % kDstH;
  int c = t / kDstH;

  int src_c = c;             // first 256 channels from 512
  int src_y = y + kYOffset;  // center crop 180 -> 50
  int src_x = x + kXOffset;  // center crop 180 -> 100

  int src_idx = (src_c * kSrcH + src_y) * kSrcW + src_x;

  mapdiffusion_bev[idx] = fusion_feature[src_idx];
}

void adapt_bevfusion_feature_to_mapdiffusion(
    const nvtype::half* fusion_feature,
    nvtype::half* mapdiffusion_bev,
    void* stream) {
  constexpr int kNumel = kDstC * kDstH * kDstW;
  cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

  cuda_linear_launch(adapt_bevfusion_to_mapdiffusion_kernel,
                     cuda_stream,
                     kNumel,
                     fusion_feature,
                     mapdiffusion_bev);
}

}  // namespace routea
}  // namespace bevfusion
