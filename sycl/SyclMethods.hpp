#pragma once

#define RANDOM_SEED 42

#include <chrono>
#include <sycl/sycl.hpp>
#include "ParsedArgs.hpp"

struct CSR {
    int n;
    int *row_ptr, *col;
    float *val;
};

struct Timing {
    double cpu_miliseconds;
    double gpu_miliseconds;
};

inline int gpu_selector(sycl::queue q) {
    try {
        if (!q.get_device().has(sycl::aspect::gpu))
            throw std::runtime_error("No GPU device available");

        std::cout << "Running on " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

        return 0;
    } catch (std::exception &e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
}

void spmv(sycl::queue& q, std::size_t N, const CSR& A, const float* x, float* y) {
    q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> id) {
        int i = id[0];
        float sum = 0.0f;
        for (int k = A.row_ptr[i]; k < A.row_ptr[i+1]; ++k)
            sum += A.val[k] * x[A.col[k]];
        
        y[i] = sum;
    });
}

float dot(sycl::queue& q, std::size_t N, const float* A, const float* B) {
    float* device_output = sycl::malloc_device<float>(1, q);

    q.submit([&](sycl::handler& h) {
        auto reduction = sycl::reduction(
            device_output, sycl::plus<float>(),
            sycl::property::reduction::initialize_to_identity{}
        );

        h.parallel_for(
            sycl::range<1>(N), reduction,
            [=](sycl::id<1> id, auto& reducer) {
                reducer += A[id] * B[id];
            }
        );
    });

    float output;
    q.memcpy(&output, device_output, sizeof(float)).wait();

    sycl::free(device_output, q);
    return output;
}

void axpy(sycl::queue& q, std::size_t N, float alpha, const float* X, float* y) {
    q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> id) {
        y[id] += alpha * X[id];
    });
}

int cg(sycl::queue& q, int maxiter) {
    int it = 0;

    for (; it < maxiter; ++it) {
            spmv(q, A, p, Ap);
            axpy(q, n, alpha, p, x); // x += alpha * p
            axpy(q, n, -alpha, Ap, r); // r -= alpha * Ap
    }

    /*
    r = b - A@x
    if prec is not None:
        u = prec@r
    else:
        u = r.copy()
    p = u.copy()
    ru = (r.T) @ u

    for k in range(maxit):
        res = rel_residual_norm_iter(A, x, b)
        rel_res_vec.append(res)
        if res < tol:
            break

        Ap = A@p
        alpha = ru / (p.T @ Ap)
        x = x + alpha*p
        r = r - alpha*Ap

        u = prec@r if prec is not None else r.copy()
        ru_new = (r.T) @ u
        beta = ru_new / ru
        p = u + beta*p
        
        ru = ru_new
        iterations += 1
    */
}

template <typename Kernel>
Timing timed_kernel(sycl::range<3> range, Kernel kernel) {
    sycl::queue q{
        sycl::gpu_selector_v,
        sycl::property::queue::enable_profiling{}
    };

    int err = gpu_selector(q);
    if (err) {
        std::exit(EXIT_FAILURE);
    }

    auto cpu_start = std::chrono::steady_clock::now();

    auto event = q.parallel_for(range, kernel);
    event.wait();

    auto cpu_end = std::chrono::steady_clock::now();

    auto gpu_start = event.template get_profiling_info<sycl::info::event_profiling::command_start>();
    auto gpu_end = event.template get_profiling_info<sycl::info::event_profiling::command_end>();

    return {
        std::chrono::duration<double>(cpu_end - cpu_start).count() * 1000,
        static_cast<double>(gpu_end - gpu_start) / 1e6
    };
}

int sycl_monte_carlo(const ParsedArgs& args);
int sycl_conjugate_gradient(const ParsedArgs& args);
int sycl_dcs(const ParsedArgs& args);
int sycl_spectral(const ParsedArgs& args);
int sycl_spmv(const ParsedArgs& args);
int sycl_stencil(const ParsedArgs& args);
