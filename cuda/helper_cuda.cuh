#ifndef HELPER_CUDA_H_
#define HELPER_CUDA_H_

#include <stdio.h>
#include <stdlib.h>

#pragma once

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUFFT(func)                                                      \
{                                                                              \
    cufftResult_t status = (func);                                             \
    if (status != CUFFT_SUCCESS) {                                             \
        printf("CUFFT API failed at line %d with error code: %d\n",            \
               __LINE__, status);                                              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#endif

__global__ void scaling_kernel(cufftComplex* data, int element_count, float scale) {
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int stride = blockDim.x * gridDim.x;
    for (auto i = tid; i<element_count; i+= stride) {
        data[i].x *= scale;
        data[i].y *= scale;
    }
}
