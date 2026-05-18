/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "bevfusion.hpp"
#include "bev-feature-dump.hpp"
#include "head-mapdiffusion.hpp"
#include "mapdiffusion-routeA-adapter.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <vector>
#include <random>

#include <numeric>

#include "common/check.hpp"
#include "common/timer.hpp"

namespace bevfusion {


static float routeA_sigmoid(float x) {
  return 1.0f / (1.0f + std::exp(-x));
}

struct RouteAPredSummary {
  int query_id;
  int class_id;
  float score;
};

static const char* routeA_class_name(int class_id) {
  if (class_id == 0) return "ped_crossing";
  if (class_id == 1) return "divider";
  if (class_id == 2) return "boundary";
  return "unknown";
}

static void save_routeA_decoded_vectors_json(
    const head::mapdiffusion::MapDiffusionOutput& output,
    const char* output_path,
    float score_threshold,
    int top_k) {
  constexpr int kNumQueries = 100;
  constexpr int kNumClasses = 3;
  constexpr int kNumPoints = 20;
  constexpr int kLineDim = 40;

  if (output.cls_logits.size() != kNumQueries * kNumClasses ||
      output.line_preds.size() != kNumQueries * kLineDim) {
    printf("[RouteA Decode] Unexpected output sizes: cls_logits=%zu line_preds=%zu\n",
           output.cls_logits.size(), output.line_preds.size());
    return;
  }

  std::vector<RouteAPredSummary> summaries;
  summaries.reserve(kNumQueries);

  for (int q = 0; q < kNumQueries; ++q) {
    int best_cls = 0;
    float best_score = -1.0f;

    for (int c = 0; c < kNumClasses; ++c) {
      float score = routeA_sigmoid(output.cls_logits[q * kNumClasses + c]);
      if (score > best_score) {
        best_score = score;
        best_cls = c;
      }
    }

    summaries.push_back({q, best_cls, best_score});
  }

  std::sort(summaries.begin(), summaries.end(),
            [](const RouteAPredSummary& a, const RouteAPredSummary& b) {
              return a.score > b.score;
            });

  std::ofstream ofs(output_path);
  if (!ofs.is_open()) {
    printf("[RouteA Decode] Failed to open output file: %s\n", output_path);
    return;
  }

  ofs << std::fixed << std::setprecision(6);
  ofs << "{\n";
  ofs << "  \"format\": \"routeA_debug_vector_decode\",\n";
  ofs << "  \"note\": \"This is debug decoding of raw MapDiffusion head tensors, not official submission_vector.json.\",\n";
  ofs << "  \"source\": \"BEVFusion fusion_feature -> fixed RouteA adapter -> MapDiffusion TensorRT head\",\n";
  ofs << "  \"tensor_shapes\": {\n";
  ofs << "    \"cls_logits\": [1, 100, 3],\n";
  ofs << "    \"line_preds\": [1, 100, 40],\n";
  ofs << "    \"points_per_query\": 20\n";
  ofs << "  },\n";
  ofs << "  \"coordinate_note\": \"normalized_points are direct line_preds values. metric_points assume 60m x 30m ROI centered at ego: x=(nx-0.5)*60, y=(ny-0.5)*30.\",\n";
  ofs << "  \"score_threshold\": " << score_threshold << ",\n";
  ofs << "  \"top_k_printed\": " << top_k << ",\n";

  ofs << "  \"top_predictions\": [\n";
  int printed = 0;
  for (size_t rank = 0; rank < summaries.size() && printed < top_k; ++rank) {
    const auto& pred = summaries[rank];
    if (pred.score < score_threshold) continue;

    if (printed > 0) ofs << ",\n";

    ofs << "    {\n";
    ofs << "      \"rank\": " << printed << ",\n";
    ofs << "      \"query_id\": " << pred.query_id << ",\n";
    ofs << "      \"class_id\": " << pred.class_id << ",\n";
    ofs << "      \"class_name\": \"" << routeA_class_name(pred.class_id) << "\",\n";
    ofs << "      \"score\": " << pred.score << ",\n";

    ofs << "      \"normalized_points\": [";
    for (int p = 0; p < kNumPoints; ++p) {
      int base = pred.query_id * kLineDim + p * 2;
      float nx = output.line_preds[base + 0];
      float ny = output.line_preds[base + 1];

      if (p > 0) ofs << ", ";
      ofs << "[" << nx << ", " << ny << "]";
    }
    ofs << "],\n";

    ofs << "      \"metric_points\": [";
    for (int p = 0; p < kNumPoints; ++p) {
      int base = pred.query_id * kLineDim + p * 2;
      float nx = output.line_preds[base + 0];
      float ny = output.line_preds[base + 1];

      float mx = (nx - 0.5f) * 60.0f;
      float my = (ny - 0.5f) * 30.0f;

      if (p > 0) ofs << ", ";
      ofs << "[" << mx << ", " << my << "]";
    }
    ofs << "]\n";

    ofs << "    }";
    printed++;
  }
  ofs << "\n  ],\n";

  ofs << "  \"all_query_scores\": [";
  for (int q = 0; q < kNumQueries; ++q) {
    if (q > 0) ofs << ", ";
    int best_cls = 0;
    float best_score = -1.0f;
    for (int c = 0; c < kNumClasses; ++c) {
      float score = routeA_sigmoid(output.cls_logits[q * kNumClasses + c]);
      if (score > best_score) {
        best_score = score;
        best_cls = c;
      }
    }
    ofs << "{\"query_id\":" << q
        << ",\"class_id\":" << best_cls
        << ",\"class_name\":\"" << routeA_class_name(best_cls)
        << "\",\"score\":" << best_score << "}";
  }
  ofs << "]\n";

  ofs << "}\n";
  ofs.close();

  printf("[RouteA Decode] Saved decoded debug vectors: %s\n", output_path);

  printf("[RouteA Decode] Top predictions:\n");
  int printed_console = 0;
  for (size_t i = 0; i < summaries.size() && printed_console < top_k; ++i) {
    const auto& pred = summaries[i];
    if (pred.score < score_threshold) continue;
    printf("  rank=%d query=%d class=%s score=%.6f\n",
           printed_console,
           pred.query_id,
           routeA_class_name(pred.class_id),
           pred.score);
    printed_console++;
  }
}

static void routeA_mapdiffusion_from_fusion_feature_smoke(
    const nvtype::half* fusion_feature,
    void* stream) {
  static bool already_ran = false;
  static std::shared_ptr<head::mapdiffusion::MapDiffusionHead> md_head = nullptr;
  static nvtype::half* adapted_bev = nullptr;

  if (already_ran) return;

  printf("==================MapDiffusion RouteA Adapted Fusion Smoke===================\n");

  if (md_head == nullptr) {
    head::mapdiffusion::MapDiffusionParameter md_param;
    md_param.model = "model/mapdiffusion_routeA/build/mapdiffusion.head.fp32.plan";
    md_param.dummy_query_value = 0.5f;
    md_param.dummy_timestep_value = 999.0f;

    md_head = head::mapdiffusion::create_mapdiffusion_head(md_param);
    if (md_head == nullptr) {
      printf("[RouteA Adapted] Failed to create MapDiffusion head.\n");
      printf("============================================================================\n");
      already_ran = true;
      return;
    }
  }

  constexpr int kDstNumel = 1 * 256 * 50 * 100;
  if (adapted_bev == nullptr) {
    cudaError_t err = cudaMalloc(&adapted_bev, kDstNumel * sizeof(nvtype::half));
    if (err != cudaSuccess) {
      printf("[RouteA Adapted] cudaMalloc failed: %s\n", cudaGetErrorString(err));
      printf("============================================================================\n");
      already_ran = true;
      return;
    }
  }

  routea::adapt_bevfusion_feature_to_mapdiffusion(fusion_feature, adapted_bev, stream);

  // -----------------------------------------------------------------------
  // Route A debug diffusion loop.
  //
  // This verifies the missing infrastructure:
  //   1) external query_coords
  //   2) external timestep
  //   3) repeated TensorRT MapDiffusion head calls
  //   4) final vector JSON decoding
  //
  // This is not exact DDIM yet. Current update rule:
  //   query_coords_{next} = clamp(line_preds, 0, 1)
  // -----------------------------------------------------------------------
  constexpr int kNumQueries = 100;
  constexpr int kNumPoints = 20;
  constexpr int kCoordDim = 2;
  constexpr int kQueryNumel = kNumQueries * kNumPoints * kCoordDim;

  std::vector<float> query_coords(kQueryNumel);
  std::mt19937 rng(12345);
  std::normal_distribution<float> normal_dist(0.5f, 0.25f);

  for (int i = 0; i < kQueryNumel; ++i) {
    float v = normal_dist(rng);
    if (v < 0.0f) v = 0.0f;
    if (v > 1.0f) v = 1.0f;
    query_coords[i] = v;
  }

  const int sampling_steps = 5;
  const int times[sampling_steps] = {1000, 750, 500, 250, 0};

  head::mapdiffusion::MapDiffusionOutput output;

  printf("[RouteA Diffusion] Begin debug sampling loop: steps=%d\n", sampling_steps);

  for (int step = 0; step < sampling_steps; ++step) {
    int t = times[step];

    output = md_head->forward_from_half_bev_with_query(
        adapted_bev,
        query_coords.data(),
        static_cast<float>(t),
        stream);

    printf("[RouteA Diffusion] step=%d timestep=%d cls_logits=%zu line_preds=%zu\n",
           step, t, output.cls_logits.size(), output.line_preds.size());

    if (output.line_preds.size() == kQueryNumel) {
      query_coords.assign(output.line_preds.begin(), output.line_preds.end());

      for (float& v : query_coords) {
        if (v < 0.0f) v = 0.0f;
        if (v > 1.0f) v = 1.0f;
      }
    } else {
      printf("[RouteA Diffusion] unexpected line_preds size, stopping loop.\n");
      break;
    }
  }

  cudaStreamSynchronize(static_cast<cudaStream_t>(stream));

  printf("[RouteA Adapted] fusion_feature [1,512,180,180] -> md_bev [1,256,50,100]\n");
  printf("[RouteA Adapted] final diffusion cls_logits=%zu line_preds=%zu\n",
         output.cls_logits.size(), output.line_preds.size());

  save_routeA_decoded_vectors_json(
      output,
      "build/mapdiffusion_routeA_diffusion_vectors.json",
      0.001f,
      10);

  if (!output.cls_logits.empty()) {
    printf("[RouteA Adapted] first cls logits: ");
    for (int i = 0; i < 6 && i < static_cast<int>(output.cls_logits.size()); ++i) {
      printf("%.6f ", output.cls_logits[i]);
    }
    printf("\n");
  }

  if (!output.line_preds.empty()) {
    printf("[RouteA Adapted] first line preds: ");
    for (int i = 0; i < 8 && i < static_cast<int>(output.line_preds.size()); ++i) {
      printf("%.6f ", output.line_preds[i]);
    }
    printf("\n");
  }

  printf("============================================================================\n");
  already_ran = true;
}



class CoreImplement : public Core {
 public:
  virtual ~CoreImplement() {
    if (lidar_points_device_) checkRuntime(cudaFree(lidar_points_device_));
    if (lidar_points_host_) checkRuntime(cudaFreeHost(lidar_points_host_));
      if (mapdiffusion_bev_float_) checkRuntime(cudaFree(mapdiffusion_bev_float_));
      if (mapdiffusion_query_) checkRuntime(cudaFree(mapdiffusion_query_));
      if (mapdiffusion_timestep_) checkRuntime(cudaFree(mapdiffusion_timestep_));
  }

  bool init(const CoreParameter& param) {
    camera_backbone_ = camera::create_backbone(param.camera_model);
    if (camera_backbone_ == nullptr) {
      printf("Failed to create camera backbone.\n");
      return false;
    }

    camera_bevpool_ =
        camera::create_bevpool(camera_backbone_->camera_shape(), param.geometry.geometry_dim.x, param.geometry.geometry_dim.y);
    if (camera_bevpool_ == nullptr) {
      printf("Failed to create camera bevpool.\n");
      return false;
    }

    camera_vtransform_ = camera::create_vtransform(param.camera_vtransform);
    if (camera_vtransform_ == nullptr) {
      printf("Failed to create camera vtransform.\n");
      return false;
    }

    transfusion_ = fuser::create_transfusion(param.transfusion);
    if (transfusion_ == nullptr) {
      printf("Failed to create transfusion.\n");
      return false;
    }

    transbbox_ = head::transbbox::create_transbbox(param.transbbox);
    if (transbbox_ == nullptr) {
      printf("Failed to create head transbbox.\n");
      return false;
    }

    lidar_scn_ = lidar::create_scn(param.lidar_scn);
    if (lidar_scn_ == nullptr) {
      printf("Failed to create lidar scn.\n");
      return false;
    }

    normalizer_ = camera::create_normalization(param.normalize);
    if (normalizer_ == nullptr) {
      printf("Failed to create normalizer.\n");
      return false;
    }

    camera_depth_ = camera::create_depth(param.normalize.output_width, param.normalize.output_height, param.normalize.num_camera);
    if (camera_depth_ == nullptr) {
      printf("Failed to create depth.\n");
      return false;
    }

    camera_geometry_ = camera::create_geometry(param.geometry);
    if (camera_geometry_ == nullptr) {
      printf("Failed to create geometry.\n");
      return false;
    }

    capacity_points_ = 300000;
    bytes_capacity_points_ = capacity_points_ * param.lidar_scn.voxelization.num_feature * sizeof(nvtype::half);
    checkRuntime(cudaMalloc(&lidar_points_device_, bytes_capacity_points_));
    checkRuntime(cudaMallocHost(&lidar_points_host_, bytes_capacity_points_));
    param_ = param;
    return true;
  }

  std::vector<head::transbbox::BoundingBox> forward_only(const void* camera_images, const nvtype::half* lidar_points,
                                                         int num_points, void* stream, bool do_normalization) {
    int cappoints = static_cast<int>(capacity_points_);
    if (num_points > cappoints) {
      printf("If it exceeds %d points, the default processing will simply crop it out.\n", cappoints);
    }

    num_points = std::min(cappoints, num_points);

    cudaStream_t _stream = static_cast<cudaStream_t>(stream);
    size_t bytes_points = num_points * param_.lidar_scn.voxelization.num_feature * sizeof(nvtype::half);
    checkRuntime(cudaMemcpyAsync(lidar_points_host_, lidar_points, bytes_points, cudaMemcpyHostToHost, _stream));
    checkRuntime(cudaMemcpyAsync(lidar_points_device_, lidar_points_host_, bytes_points, cudaMemcpyHostToDevice, _stream));

    const nvtype::half* lidar_feature = this->lidar_scn_->forward(lidar_points_device_, num_points, stream);
    nvtype::half* normed_images = (nvtype::half*)camera_images;
    if (do_normalization) {
      normed_images = (nvtype::half*)this->normalizer_->forward((const unsigned char**)(camera_images), stream);
    }
    const nvtype::half* depth = this->camera_depth_->forward(lidar_points_device_, num_points, 5, stream);

    this->camera_backbone_->forward(normed_images, depth, stream);
    const nvtype::half* camera_bev = this->camera_bevpool_->forward(
        this->camera_backbone_->feature(), this->camera_backbone_->depth(), this->camera_geometry_->indices(),
        this->camera_geometry_->intervals(), this->camera_geometry_->num_intervals(), stream);

    const nvtype::half* camera_bevfeat = camera_vtransform_->forward(camera_bev, stream);
    const nvtype::half* fusion_feature = this->transfusion_->forward(camera_bevfeat, lidar_feature, stream);
    static bool dumped_fusion_feature = false;
    if (!dumped_fusion_feature) {
      debug::dump_half_tensor_to_file(
          "build/bevfusion_fusion_feature_1x512x180x180.fp16.bin",
          fusion_feature,
          1 * 512 * 180 * 180,
          stream);
      dumped_fusion_feature = true;
    }
    routeA_mapdiffusion_from_fusion_feature_smoke(fusion_feature, stream);
    return this->transbbox_->forward(fusion_feature, param_.transbbox.confidence_threshold, stream,
                                     param_.transbbox.sorted_bboxes);
  }

  std::vector<head::transbbox::BoundingBox> forward_timer(const void* camera_images, const nvtype::half* lidar_points,
                                                          int num_points, void* stream, bool do_normalization) {
    int cappoints = static_cast<int>(capacity_points_);
    if (num_points > cappoints) {
      printf("If it exceeds %d points, the default processing will simply crop it out.\n", cappoints);
    }

    num_points = std::min(cappoints, num_points);

    printf("==================BEVFusion===================\n");
    std::vector<float> times;
    cudaStream_t _stream = static_cast<cudaStream_t>(stream);
    timer_.start(_stream);

    size_t bytes_points = num_points * param_.lidar_scn.voxelization.num_feature * sizeof(nvtype::half);
    checkRuntime(cudaMemcpyAsync(lidar_points_host_, lidar_points, bytes_points, cudaMemcpyHostToHost, _stream));
    checkRuntime(cudaMemcpyAsync(lidar_points_device_, lidar_points_host_, bytes_points, cudaMemcpyHostToDevice, _stream));
    timer_.stop("[NoSt] CopyLidar");

    nvtype::half* normed_images = (nvtype::half*)camera_images;
    if (do_normalization) {
      timer_.start(_stream);
      normed_images = (nvtype::half*)this->normalizer_->forward((const unsigned char**)(camera_images), stream);
      timer_.stop("[NoSt] ImageNrom");
    }

    timer_.start(_stream);
    const nvtype::half* lidar_feature = this->lidar_scn_->forward(lidar_points_device_, num_points, stream);
    times.emplace_back(timer_.stop("Lidar Backbone"));

    timer_.start(_stream);
    const nvtype::half* depth = this->camera_depth_->forward(lidar_points_device_, num_points, 5, stream);
    times.emplace_back(timer_.stop("Camera Depth"));

    timer_.start(_stream);
    this->camera_backbone_->forward(normed_images, depth, stream);
    times.emplace_back(timer_.stop("Camera Backbone"));

    timer_.start(_stream);
    const nvtype::half* camera_bev = this->camera_bevpool_->forward(
        this->camera_backbone_->feature(), this->camera_backbone_->depth(), this->camera_geometry_->indices(),
        this->camera_geometry_->intervals(), this->camera_geometry_->num_intervals(), stream);
    times.emplace_back(timer_.stop("Camera Bevpool"));

    timer_.start(_stream);
    const nvtype::half* camera_bevfeat = camera_vtransform_->forward(camera_bev, stream);
    times.emplace_back(timer_.stop("VTransform"));

    timer_.start(_stream);
    const nvtype::half* fusion_feature = this->transfusion_->forward(camera_bevfeat, lidar_feature, stream);
    static bool dumped_fusion_feature = false;
    if (!dumped_fusion_feature) {
      debug::dump_half_tensor_to_file(
          "build/bevfusion_fusion_feature_1x512x180x180.fp16.bin",
          fusion_feature,
          1 * 512 * 180 * 180,
          stream);
      dumped_fusion_feature = true;
    }
    routeA_mapdiffusion_from_fusion_feature_smoke(fusion_feature, stream);
    times.emplace_back(timer_.stop("Transfusion"));

    timer_.start(_stream);
    auto output =
        this->transbbox_->forward(fusion_feature, param_.transbbox.confidence_threshold, stream, param_.transbbox.sorted_bboxes);
    times.emplace_back(timer_.stop("Head BoundingBox"));

    float total_time = std::accumulate(times.begin(), times.end(), 0.0f, std::plus<float>{});
    printf("Total: %.3f ms\n", total_time);
    printf("=============================================\n");
    return output;
  }

  virtual std::vector<head::transbbox::BoundingBox> forward(const unsigned char** camera_images, const nvtype::half* lidar_points,
                                                            int num_points, void* stream) override {
    if (enable_timer_) {
      return this->forward_timer(camera_images, lidar_points, num_points, stream, true);
    } else {
      return this->forward_only(camera_images, lidar_points, num_points, stream, true);
    }
  }

  virtual std::vector<head::transbbox::BoundingBox> forward_no_normalize(const nvtype::half* camera_normed_images_device,
                                                                         const nvtype::half* lidar_points, int num_points,
                                                                         void* stream) override {
    if (enable_timer_) {
      return this->forward_timer(camera_normed_images_device, lidar_points, num_points, stream, false);
    } else {
      return this->forward_only(camera_normed_images_device, lidar_points, num_points, stream, false);
    }
  }

  virtual void set_timer(bool enable) override { enable_timer_ = enable; }

  virtual void print() override {
    camera_backbone_->print();
    camera_vtransform_->print();
    transfusion_->print();
    transbbox_->print();
  }

  virtual void update(const float* camera2lidar, const float* camera_intrinsics, const float* lidar2image,
                      const float* img_aug_matrix, void* stream) override {
    camera_depth_->update(img_aug_matrix, lidar2image, stream);
    camera_geometry_->update(camera2lidar, camera_intrinsics, img_aug_matrix, stream);
  }

  virtual void free_excess_memory() override { camera_geometry_->free_excess_memory(); }

 private:
  CoreParameter param_;
  nv::EventTimer timer_;
  nvtype::half* lidar_points_device_ = nullptr;
  nvtype::half* lidar_points_host_ = nullptr;
  size_t capacity_points_ = 0;
  size_t bytes_capacity_points_ = 0;

  std::shared_ptr<camera::Normalization> normalizer_;
  std::shared_ptr<camera::Backbone> camera_backbone_;
  std::shared_ptr<camera::BEVPool> camera_bevpool_;
  std::shared_ptr<camera::VTransform> camera_vtransform_;
  std::shared_ptr<camera::Depth> camera_depth_;
  std::shared_ptr<camera::Geometry> camera_geometry_;
  std::shared_ptr<lidar::SCN> lidar_scn_;
  std::shared_ptr<fuser::Transfusion> transfusion_;
  std::shared_ptr<head::transbbox::TransBBox> transbbox_;
    std::shared_ptr<head::mapdiffusion::MapDiffusionHead> mapdiffusion_head_;
    float* mapdiffusion_bev_float_ = nullptr;
    float* mapdiffusion_query_ = nullptr;
    float* mapdiffusion_timestep_ = nullptr;
  float confidence_threshold_ = 0;
  bool enable_timer_ = false;
};

std::shared_ptr<Core> create_core(const CoreParameter& param) {
  std::shared_ptr<CoreImplement> instance(new CoreImplement());
  if (!instance->init(param)) {
    instance.reset();
  }
  return instance;
}

};  // namespace bevfusion