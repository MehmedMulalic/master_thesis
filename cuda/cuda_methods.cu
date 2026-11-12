/*
 * Taken inspiration from the following GitHub repository:
 * https://github.com/NVIDIA/CUDALibrarySamples/
 * 
 * Macro definitions are commonly used across many authors
 * which is why they are copied as-is from the repository.
 */

#include <cuda_runtime.h>
#include <cusparse.h>
#include <stdio.h>
#include <stdlib.h>

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

// __global__ void cuda_dcs(float *x) {
//     return;
// }

// __global__ void cuda_stencil(float *x) {
//     return;
// }

int cuda_spmv() {
    const int64_t A_num_rows = 4;
    const int64_t A_num_cols = 4;
    const int64_t A_nnz = 9;
    int   hA_indptr[]  = { 0, 3, 6, 8, 9 };
    int   hA_indices[] = { 0, 2, 3, 1, 2, 3, 0, 1, 0 };
    float hA_values[]  = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f };
    float hX[] = { 1.0f, 2.0f, 3.0f, 4.0f };
    float hY[] = { 0.0f, 0.0f, 0.0f, 0.0f };
    float alpha = 1.0f;
    float beta  = 0.0f;

    int *dA_indices, *dA_indptr;
    float *dA_values, *dX, *dY;
    CHECK_CUDA(cudaMalloc((void**) &dA_indptr, sizeof(int) * (A_num_rows+1)));
    CHECK_CUDA(cudaMalloc((void**) &dA_indices, sizeof(int) * A_nnz));
    CHECK_CUDA(cudaMalloc((void**) &dA_values, sizeof(float) * A_nnz));
    CHECK_CUDA(cudaMalloc((void**) &dX, sizeof(float) * A_num_cols));
    CHECK_CUDA(cudaMalloc((void**) &dY, sizeof(float) * A_num_rows));

    CHECK_CUDA(cudaMemcpy(dA_indptr, hA_indptr, sizeof(int) * (A_num_rows+1), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dA_indices, hA_indices, sizeof(int) * A_nnz, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dA_values, hA_values, sizeof(float) * A_nnz, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dX, hX, sizeof(float) * A_num_cols, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dY, hY, sizeof(float) * A_num_rows, cudaMemcpyHostToDevice));

    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    void*  dBuffer    = NULL;
    size_t bufferSize = 0;
    
    CHECK_CUSPARSE(cusparseCreate(&handle));
    CHECK_CUSPARSE(cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
                                     dA_indptr, dA_indices, dA_values,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, A_num_cols, dX, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, A_num_rows, dY, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                                           CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    
                                           CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));
    CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));

    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecX));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecY));
    CHECK_CUSPARSE(cusparseDestroy(handle));

    CHECK_CUDA(cudaMemcpy(hY, dY, sizeof(float) * A_num_rows, cudaMemcpyDeviceToHost));
    int correct = 1;
    float sum = 0.0f;
    for (int i = 0; i < A_num_rows; ++i) {
        sum += static_cast<float>(hY[i]);
    }
    if (sum != 98.0) correct = 0;
    if (correct) {
        printf("SpMv test PASSED\n");
    } else {
        printf("SpMv test FAILED: sum = %f\n", sum);
    }

    CHECK_CUDA(cudaFree(dBuffer));
    CHECK_CUDA(cudaFree(dY));
    CHECK_CUDA(cudaFree(dX));
    CHECK_CUDA(cudaFree(dA_values));
    CHECK_CUDA(cudaFree(dA_indptr));
    CHECK_CUDA(cudaFree(dA_indices));
    return 0;
}

int main() {
    cuda_spmv();
    return EXIT_SUCCESS;
}
