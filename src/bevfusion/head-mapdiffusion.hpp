#ifndef __HEAD_MAPDIFFUSION_HPP__
#define __HEAD_MAPDIFFUSION_HPP__

#include <memory>
#include <string>
#include <vector>

#include "common/dtype.hpp"

namespace bevfusion {
namespace head {
namespace mapdiffusion {

struct MapDiffusionParameter {
  std::string model;

  int batch_size = 1;
  int bev_channels = 256;
  int bev_height = 50;
  int bev_width = 100;

  int num_queries = 100;
  int num_points = 20;
  int num_classes = 3;

  float dummy_query_value = 0.5f;
  float dummy_timestep_value = 1.0f;
};

struct MapDiffusionOutput {
  std::vector<float> cls_logits;  // [1, 100, 3]
  std::vector<float> line_preds;  // [1, 100, 40]
};

class MapDiffusionHead {
 public:
  virtual ~MapDiffusionHead() = default;

  virtual MapDiffusionOutput forward(
      const float* bev_features,
      const float* query_coords,
      const float* timestep,
      void* stream) = 0;

  // Route A integration helper:
  // BEVFusion produces half BEV features, but current MapDiffusion plan is FP32.
  virtual MapDiffusionOutput forward_from_half_bev(
      const nvtype::half* bev_features_half,
      void* stream) = 0;

  // Route A debug diffusion API.
  // query_coords_host must be CPU float array [1, 100, 20, 2] = 4000 floats.
  // timestep is copied to the TensorRT timestep input.
  virtual MapDiffusionOutput forward_from_half_bev_with_query(
      const nvtype::half* bev_features_half,
      const float* query_coords_host,
      float timestep,
      void* stream) = 0;

  virtual void print() = 0;
};

std::shared_ptr<MapDiffusionHead> create_mapdiffusion_head(const MapDiffusionParameter& param);

}  // namespace mapdiffusion
}  // namespace head
}  // namespace bevfusion

#endif
