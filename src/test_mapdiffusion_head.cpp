#include <cuda_runtime.h>

#include <cstdio>
#include <string>
#include <vector>

#include "bevfusion/head-mapdiffusion.hpp"
#include "common/check.hpp"

int main(int argc, char** argv) {
  std::string plan = "model/mapdiffusion_routeA/build/mapdiffusion.head.fp32.plan";
  if (argc > 1) {
    plan = argv[1];
  }

  printf("[Test] Plan: %s\n", plan.c_str());

  cudaStream_t stream;
  checkRuntime(cudaStreamCreate(&stream));

  const int bev_numel = 1 * 256 * 50 * 100;
  const int query_numel = 1 * 100 * 20 * 2;
  const int timestep_numel = 1;

  std::vector<float> bev_host(bev_numel, 0.01f);
  std::vector<float> query_host(query_numel, 0.5f);
  std::vector<float> timestep_host(timestep_numel, 1.0f);

  float* bev_device = nullptr;
  float* query_device = nullptr;
  float* timestep_device = nullptr;

  checkRuntime(cudaMalloc(&bev_device, bev_numel * sizeof(float)));
  checkRuntime(cudaMalloc(&query_device, query_numel * sizeof(float)));
  checkRuntime(cudaMalloc(&timestep_device, timestep_numel * sizeof(float)));

  checkRuntime(cudaMemcpyAsync(bev_device, bev_host.data(), bev_numel * sizeof(float), cudaMemcpyHostToDevice, stream));
  checkRuntime(cudaMemcpyAsync(query_device, query_host.data(), query_numel * sizeof(float), cudaMemcpyHostToDevice, stream));
  checkRuntime(cudaMemcpyAsync(timestep_device, timestep_host.data(), timestep_numel * sizeof(float), cudaMemcpyHostToDevice, stream));

  bevfusion::head::mapdiffusion::MapDiffusionParameter param;
  param.model = plan;

  auto head = bevfusion::head::mapdiffusion::create_mapdiffusion_head(param);
  if (head == nullptr) {
    printf("[Test] Failed to create MapDiffusion head.\n");
    return 1;
  }

  head->print();

  auto output = head->forward(bev_device, query_device, timestep_device, stream);

  printf("[Test] cls_logits size: %zu\n", output.cls_logits.size());
  printf("[Test] line_preds size: %zu\n", output.line_preds.size());

  printf("[Test] first cls logits: ");
  for (int i = 0; i < 6 && i < (int)output.cls_logits.size(); ++i) {
    printf("%.6f ", output.cls_logits[i]);
  }
  printf("\n");

  printf("[Test] first line preds: ");
  for (int i = 0; i < 8 && i < (int)output.line_preds.size(); ++i) {
    printf("%.6f ", output.line_preds[i]);
  }
  printf("\n");

  checkRuntime(cudaFree(bev_device));
  checkRuntime(cudaFree(query_device));
  checkRuntime(cudaFree(timestep_device));
  checkRuntime(cudaStreamDestroy(stream));

  printf("[Test] Done.\n");
  return 0;
}
