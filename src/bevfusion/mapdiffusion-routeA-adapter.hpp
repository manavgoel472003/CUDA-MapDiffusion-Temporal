#pragma once

#include "common/dtype.hpp"

namespace bevfusion {
namespace routea {

// Stage-1 fixed adapter:
//   input:  BEVFusion fusion_feature [1, 512, 180, 180] fp16
//   output: MapDiffusion BEV       [1, 256,  50, 100] fp16
void adapt_bevfusion_feature_to_mapdiffusion(
    const nvtype::half* fusion_feature,
    nvtype::half* mapdiffusion_bev,
    void* stream);

}  // namespace routea
}  // namespace bevfusion
