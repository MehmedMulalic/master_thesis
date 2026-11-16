/*
 * Taken inspiration from the following GitHub repository:
 * https://github.com/NVIDIA/CUDALibrarySamples/
 * 
 * Macro definitions are commonly used across many authors
 * which is why they are copied as-is from the repository.
 */

#include "helper_cuda.cuh"
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cufft.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

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
    
    int Nx = 2;
    int Ny = 3;
    int Nz = 4;
    int fftSize = Ny;
    int batchSize = Nx*Nz;
    int elementCount = Nx * Ny * Nz;

    //! Custom matrix for testing purposes
    std::vector<cufftReal> input = {
        0.37454012f, 0.95071431f, 0.73199394f, 0.59865848f, 0.15601864f,
        0.15599452f, 0.05808361f, 0.86617615f, 0.60111501f, 0.70807258f,
        0.02058449f, 0.96990985f, 0.83244264f, 0.21233911f, 0.18182497f,
        0.18340451f, 0.30424224f, 0.52475643f, 0.43194502f, 0.29122914f,
        0.61185289f, 0.13949386f, 0.29214465f, 0.36636184f
    };
    std::vector<cufftReal> output(elementCount, 0);

    int threads = 256;
    int blocks = (elementCount + threads - 1) / threads; 

    cufftReal *d_originalInput, *d_input;
    cufftComplex *d_output;
    
    CHECK_CUFFT(cufftPlan1d(&planR2C, fftSize, CUFFT_R2C, batchSize));
    CHECK_CUFFT(cufftPlan1d(&planC2R, fftSize, CUFFT_C2R, batchSize));

    CHECK_CUDA(cudaMalloc((void**) &d_originalInput, sizeof(cufftReal) * input.size()));
    CHECK_CUDA(cudaMalloc((void**) &d_output, sizeof(cufftComplex) * output.size()));

    CHECK_CUDA(cudaMemcpy(d_originalInput, input.data(), sizeof(cufftReal) * input.size(), cudaMemcpyHostToDevice));

    // test
    int minGridSize, blockSize;
    CHECK_CUDA(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, reordering_kernel, 0, 0));
    printf("%d, %d\n", minGridSize, blockSize);

    // DCT attempt
    // 1. Reorder data
    reordering_kernel<<<blocks, threads>>>(d_originalInput, d_input, Ny, Nz, elementCount);
    CHECK_CUDA_KERNEL();

    // 2. Forward FFT
    CHECK_CUFFT(cufftExecR2C(planR2C, d_input, d_output));

    // 3. Phase correction
    cufftReal *d_dct;
    dct_phase_correction<<<blocks, threads>>>(d_output, d_dct, Ny, Nz, elementCount);
    CHECK_CUDA_KERNEL();

    // iDCT attempt
    // 1. Calculate V(k)
    cufftComplex *d_vk;
    idct_calc_vk<<<blocks, threads>>>(d_dct, d_vk, Ny, Nz, elementCount);
    CHECK_CUDA_KERNEL();
    
    // 2. Inverse FFT (backwards normalization)
    CHECK_CUFFT(cufftExecC2R(planC2R, d_output, d_input));
    
    scaling_kernel<<<blocks, threads>>>(d_input, elementCount, 1/elementCount);
    CHECK_CUDA_KERNEL();

    // 3. Reorder data
    unreordering_kernel<<<blocks, threads>>>(d_input, d_originalInput, Ny, Nz, elementCount);
    CHECK_CUDA_KERNEL();

    CHECK_CUDA(cudaMemcpy(output.data(), d_input, sizeof(cufftReal) * output.size(), cudaMemcpyDeviceToHost));

    // Check results
    bool is_close = allclose(input, output);
    printf("%d\n", is_close);

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
