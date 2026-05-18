#!/usr/bin/env bash
set -euo pipefail

cd /home/018198687/Mapping/Lidar_AI_Solution/CUDA-BEVFusion

# tool/environment.sh expects these to exist; set -u would otherwise crash.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export PATH="${PATH:-/usr/local/bin:/usr/bin:/bin}"

source tool/environment.sh

mkdir -p build/plugins

# cuBLAS headers/libs are sometimes under NVHPC math_libs, not CUDA_Inc.
CUBLAS_INC="${CUBLAS_INC:-}"
CUBLAS_LIB="${CUBLAS_LIB:-}"

if [ -z "$CUBLAS_INC" ]; then
  for d in \
    "$CUDA_Inc" \
    "${CUDA_HOME:-}/include" \
    "/opt/fs/atipa/local/opt/ohpc/pub/apps/nvidia/nvhpc/24.11/Linux_x86_64/24.11/math_libs/11.8/include" \
    "/opt/fs/atipa/local/opt/ohpc/pub/apps/nvidia/nvhpc/24.11/Linux_x86_64/24.11/math_libs/include"
  do
    if [ -f "$d/cublas_v2.h" ]; then
      CUBLAS_INC="$d"
      break
    fi
  done
fi

if [ -z "$CUBLAS_LIB" ]; then
  for d in \
    "$CUDA_Lib" \
    "${CUDA_HOME:-}/targets/x86_64-linux/lib" \
    "/opt/fs/atipa/local/opt/ohpc/pub/apps/nvidia/nvhpc/24.11/Linux_x86_64/24.11/math_libs/11.8/lib64" \
    "/opt/fs/atipa/local/opt/ohpc/pub/apps/nvidia/nvhpc/24.11/Linux_x86_64/24.11/math_libs/lib64"
  do
    if ls "$d"/libcublas.so* >/dev/null 2>&1; then
      CUBLAS_LIB="$d"
      break
    fi
  done
fi

echo "TensorRT_Inc=${TensorRT_Inc}"
echo "TensorRT_Lib=${TensorRT_Lib}"
echo "CUDA_Inc=${CUDA_Inc}"
echo "CUDA_Lib=${CUDA_Lib}"
echo "CUBLAS_INC=${CUBLAS_INC}"
echo "CUBLAS_LIB=${CUBLAS_LIB}"

if [ -z "$CUBLAS_INC" ] || [ -z "$CUBLAS_LIB" ]; then
  echo "ERROR: could not locate cuBLAS include/lib paths"
  exit 1
fi

nvcc -std=c++14 -shared -Xcompiler -fPIC \
  -O3 --use_fast_math -Xptxas -O3 -Xptxas -v \
  -gencode arch=compute_80,code=sm_80 \
  -I${TensorRT_Inc} \
  -I${CUDA_Inc} \
  -I${CUBLAS_INC} \
  src/plugins/mmcv_dcnv2/mmcv_modulated_deform_conv2d_plugin.cu \
  -L${TensorRT_Lib} \
  -L${CUDA_Lib} \
  -L${CUBLAS_LIB} \
  -lnvinfer \
  -lcublas \
  -lcudart \
  -o build/plugins/libmmcv_dcnv2_trt.so

ls -lh build/plugins/libmmcv_dcnv2_trt.so

echo "Plugin strings:"
strings build/plugins/libmmcv_dcnv2_trt.so | grep -E "MMCVModulatedDeformConv2d|Plugin" | head -40 || true
