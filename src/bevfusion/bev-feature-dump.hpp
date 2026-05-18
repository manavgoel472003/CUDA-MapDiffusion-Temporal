#pragma once

#include "common/dtype.hpp"

namespace bevfusion {
namespace debug {

void dump_half_tensor_to_file(
    const char* path,
    const nvtype::half* device_ptr,
    int numel,
    void* stream);

}  // namespace debug
}  // namespace bevfusion
