#!/bin/bash

# TensorRT
export TensorRT_Root=/home/018198687/Mapping/local/TensorRT-8.5.3.1
export TensorRT_Bin=$TensorRT_Root/bin
export TensorRT_Lib=$TensorRT_Root/lib
export TensorRT_Inc=$TensorRT_Root/include

# CUDA 11.8 from NVHPC
export CUDA11_ROOT=/opt/fs/atipa/local/opt/ohpc/pub/apps/nvidia/nvhpc/24.11/Linux_x86_64/24.11/cuda/11.8
export CUDA_HOME=$CUDA11_ROOT
export CUDA_Bin=$CUDA_HOME/bin
export CUDA_Inc=$CUDA_HOME/include
export CUDA_Lib=$CUDA_HOME/targets/x86_64-linux/lib

# CUDA runtime / cuBLAS / cuDNN locations
export CUDA11_REDIST=/opt/fs/atipa/local/opt/ohpc/pub/apps/nvidia/nvhpc/24.11/Linux_x86_64/24.11/REDIST/cuda/11.8/targets/x86_64-linux/lib
export CUBLAS_LIB=/opt/fs/atipa/local/opt/ohpc/pub/apps/nvidia/nvhpc/24.11/Linux_x86_64/24.11/math_libs/11.8/targets/x86_64-linux/lib
export CUBLAS_REDIST=/opt/fs/atipa/local/opt/ohpc/pub/apps/nvidia/nvhpc/24.11/Linux_x86_64/24.11/REDIST/math_libs/11.8/targets/x86_64-linux/lib
export CUDNN_Lib=/opt/fs/atipa/local/opt/ohpc/pub/coe/apps/matlab/R2023b/bin/glnxa64

# CUDA-BEVFusion model selection
export DEBUG_MODEL=resnet50int8
export DEBUG_PRECISION=int8
export DEBUG_DATA=example-data
export USE_Python=OFF
export SPCONV_CUDA_VERSION=11.8

# Required so build_trt_engine.sh can call "trtexec" directly.
export PATH=$TensorRT_Bin:$CUDA_Bin:$PATH

# Runtime library path
export LD_LIBRARY_PATH=$TensorRT_Lib:$CUDA_Lib:$CUDA11_REDIST:$CUBLAS_LIB:$CUBLAS_REDIST:$CUDNN_Lib:$LD_LIBRARY_PATH

# Configuration status consumed by tool/build_trt_engine.sh
export ConfigurationStatus=Success

if [ ! -f "$TensorRT_Bin/trtexec" ]; then
    echo "Can not find $TensorRT_Bin/trtexec, there may be a mistake in the directory you configured."
    export ConfigurationStatus=Failed
fi
