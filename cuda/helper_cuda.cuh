#ifndef HELPER_CUDA_H_
#define HELPER_CUDA_H_

#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <vector>

#pragma once

#define CHECK_CUDA_KERNEL()                                                    \
{                                                                              \
    cudaError_t err = cudaGetLastError();                                      \
    if (err != cudaSuccess)                                                    \
    printf("Launch error: %s\n", cudaGetErrorString(err));                     \
}                                                                              \

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

__global__ void scaling_kernel(float* data, int element_count, float scale) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int stride = blockDim.x * gridDim.x;
    for (auto i = tid; i < element_count; i += stride) {
        data[i] *= scale;
    }
}

__global__ void reordering_kernel(const cufftReal* __restrict__ x,
                                 cufftReal* __restrict__ v,
                                 int N, int stride, int elementCount) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= elementCount) return;
    
    // tid = x*12 + y*4 + z*1;
    // n = y;
    // base = tid - y*4 = x*12 + z;
    int n    = (tid / stride) % N;
    int base = tid - (n * stride);
    int half = (N + 1) / 2;

    int src = (n < half) ? 2*n : 2*N - 2*n - 1;

    v[tid] = x[base + src*stride];
}

__global__ void unreordering_kernel(const cufftReal* __restrict__ v,
                                    cufftReal* __restrict__ x,
                                    int N, int stride, int elementCount) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= elementCount) return;

    int n    = (tid / stride) % N;
    int base = tid - (n * stride);

    int src = (n % 2 == 0) ? n/2 : N-1 - ((n-1) / 2);

    x[tid] = v[base + src*stride];
}

__global__ void dct_phase_correction(const cufftComplex* __restrict__ X,
                                     cufftReal* __restrict__ C,
                                     int N, int stride, int elementCount) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= elementCount) return;

    int k = (tid / stride) % N;

    float s, c;
    sincospif(k / (2.0f*N), &s, &c);

    cufftComplex x = X[k];
    C[tid] = 2.0f * (x.x * c + x.y * s);
}

__global__ void idct_calc_vk(const cufftReal* __restrict__ C,
                             cufftComplex* __restrict__ V,
                             int N, int stride, int elementCount) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= elementCount) return;

    int k = (tid / stride) % N;
    int base = tid - (k * stride);

    float a = C[tid]; // C[k]
    float b = (k == 0) ? 0.0 : C[base + (N - k)*stride]; // C[k]_mirrored
    
    float s, c;
    sincospif(k / (2.0*N), &s, &c);

    // theta = kpi / 2N
    // 0.5 * e^{1j * theta} * (a - 1j * b)
    V[tid] = make_cuFloatComplex(0.5 * (a*c + b*s),
                                 0.5 * (a*s - b*c));
}

bool allclose(const std::vector<float>& a, const std::vector<float>& b, 
              float rtol = 1e-05, float atol = 1e-08) {
    if (a.size() != b.size()) return false;

    return std::equal(a.begin(), a.end(), b.begin(),
        [rtol, atol](float x, float y) {
            return std::abs(x - y) <= atol + rtol * std::abs(y);
        }
    );
}
