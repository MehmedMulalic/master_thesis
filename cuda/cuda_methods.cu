/*
 * Taken inspiration from the following GitHub repository:
 * https://github.com/NVIDIA/CUDALibrarySamples/
 * 
 * Macro definitions are commonly used across many authors
 * which is why they are copied as-is from the repository.
 */

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cufft.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

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

#define CHECK_CUFFT(func)                                                   \
{                                                                              \
    cufftResult_t status = (func);                                          \
    if (status != CUFFT_SUCCESS) {                                   \
        printf("CUFFT API failed at line %d with error code: %d\n",         \
               __LINE__, status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

__global__ void scaling_kernel(cufftComplex* data, int element_count, float scale) {
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int stride = blockDim.x * gridDim.x;
    for (auto i = tid; i<element_count; i+= stride) {
        data[i].x *= scale;
        data[i].y *= scale;
    }
}

int cuda_monte_carlo() {
    return 0;
}

int cuda_conjugate_gradient() {
    return 0;
}

int cuda_dcs() {
    return 0;
}

/* NOTES
* potrebno je znati malo vise teorije zbog mallo
* generalno treba error checking jer su async funkcije,
* nesto sto dodje vec uz cupy
* oboje su generalno sitnice
*/
int cuda_spectral() {
    // const float A = 100.0f;               // Peak temperature (°C)
    // const float sigma_initial = 0.2f;     // Initial width (m)
    // const float alpha = 1.6563e-4;        // Thermal diffusivity of pure silver (99.9%) (m^2/s)

    // x0 = y0 = z0 = 0.5f              // Center
    // Lx = Ly = Lz = 1.0f              // Domain limit (metres)
    // nx = ny = nz = args.grid_size

    // dx = Lx / (nx - 1)              // x spacing
    // dy = Ly / (ny - 1)              // y spacing
    // dz = Lz / (nz - 1)              // z spacing
    // dt = 0.0003f                     // Time step (s)
    // t_final = 60                    // Final time step (s)
    // num_steps = round(t_final / dt)

    cufftHandle planR2C, planC2R;
    
    int fft_size = 10;
    int batch_size = 1;
    int element_count = batch_size * fft_size;
    
    std::vector<cufftReal> input = {
            0.37454012f, 0.95071431f, 0.73199394f, 0.59865848f, 0.15601864f,
            0.15599452f, 0.05808361f, 0.86617615f, 0.60111501f, 0.70807258f
        };
    std::vector<cufftReal> output(element_count, 0);

    // Sanity check
    for (auto v: input) {
        printf("%f, ", v);
    }
    printf("\n");

    cufftReal *d_input;
    cufftComplex *d_output;
    
    CHECK_CUFFT(cufftPlan1d(&planR2C, fft_size, CUFFT_R2C, batch_size));
    CHECK_CUFFT(cufftPlan1d(&planC2R, fft_size, CUFFT_C2R, batch_size));

    CHECK_CUDA(cudaMalloc((void**) &d_input, sizeof(cufftReal) * input.size()));
    CHECK_CUDA(cudaMalloc((void**) &d_output, sizeof(cufftComplex) * output.size()));

    // Forward FFT
    CHECK_CUDA(cudaMemcpy(d_input, input.data(), sizeof(cufftReal) * input.size(), cudaMemcpyHostToDevice));
    CHECK_CUFFT(cufftExecR2C(planR2C, d_input, d_output));

    // Inverse FFT
    CHECK_CUFFT(cufftExecC2R(planC2R, d_output, d_input));
    CHECK_CUDA(cudaMemcpy(output.data(), d_input, sizeof(cufftReal) * output.size(), cudaMemcpyDeviceToHost));

    // Display results
    for (auto v: output) {
        printf("%f, ", v);
    }
    printf("\n");

    CHECK_CUFFT(cufftDestroy(planR2C));
    CHECK_CUFFT(cufftDestroy(planC2R));
    return 0;
}

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

int cuda_stencil() {
    return 0;
}

int main() {
    cuda_spectral();
    return EXIT_SUCCESS;
}
