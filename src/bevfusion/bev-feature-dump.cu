#include "bevfusion/bev-feature-dump.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace bevfusion {
namespace debug {

void dump_half_tensor_to_file(
    const char* path,
    const nvtype::half* device_ptr,
    int numel,
    void* stream) {
  cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

  std::vector<nvtype::half> host(numel);

  cudaMemcpyAsync(
      host.data(),
      device_ptr,
      numel * sizeof(nvtype::half),
      cudaMemcpyDeviceToHost,
      cuda_stream);

  cudaStreamSynchronize(cuda_stream);

  FILE* f = fopen(path, "wb");
  if (f == nullptr) {
    printf("[BEVFeatureDump] Failed to open %s\n", path);
    return;
  }

  fwrite(host.data(), sizeof(nvtype::half), numel, f);
  fclose(f);

  printf("[BEVFeatureDump] Saved %s, numel=%d, bytes=%zu\n",
         path,
         numel,
         numel * sizeof(nvtype::half));
}

}  // namespace debug
}  // namespace bevfusion
