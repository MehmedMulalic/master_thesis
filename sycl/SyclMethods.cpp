#include <iostream>
#include <random>
#include <vector>
#include <chrono>
#include <cmath>
#include <string>
#include <sycl/sycl.hpp>
#include "SyclMethods.hpp"

#define MAX_NNZ = 4'000'000'000     // 32GB VRAM
#define K_E = 8.987551786214e9      // https://en.wikipedia.org/wiki/Coulomb%27s_law

// Source:
// https://rosettacode.org/wiki/Pseudo-random_numbers/Splitmix64
inline std::uint64_t splitmix64(std::uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

// Source:
// https://github.com/numpy/numpy/blob/main/numpy/random/src/distributions/distributions.c#L19-L21
inline float next_float(std::uint64_t x) {
    return static_cast<float>(x >> 40) * (1.0f / 16777216.0f);
}

float mc_kernel(sycl::queue& q, std::uint64_t seed, std::size_t N,
                std::size_t d, float bound, double* kernel_ms) {
    float* sum = sycl::malloc_device<float>(1, q);

    sycl::event ev = q.submit([&](sycl::handler& h) {
        auto reduction = sycl::reduction(
            sum, sycl::plus<float>(),
            sycl::property::reduction::initialize_to_identity{}
        );
        
        h.parallel_for(
            sycl::range<1>(N), reduction,
            [=](sycl::id<1> id, auto& reducer) {
                std::uint64_t state = splitmix64(seed + id[0] * 0x9E3779B97F4A7C15ULL);

                float sq_norm = 0.0f;
                for (std::size_t i = 0; i < d; ++i) {
                    state = splitmix64(state);
                    float u = next_float(state);
                    float v = -bound + 2.0f * bound * u;
                    sq_norm += v * v;
                }

                acc += sycl::exp(-sq_norm);
            }
        );
    });
    ev.wait();

    auto t_start = ev.get_profiling_info<sycl::info::event_profiling::command_start>();
    auto t_end = ev.get_profiling_info<sycl::info::event_profiling::command_end>();
    *kernel_ms = static_cast<double>(t_end - t_start) * 1e-6;

    float host_sum = 0.0f;
    q.memcpy(&host_sum, sum, sizeof(float)).wait();
    sycl::free(sum, q);

    float mean = host_sum / static_cast<float>(N);
    float volume = std::pow(2.0f * bound, static_cast<float>(d));
    return volume * mean;
}

/*
* Monte Carlo integration of the Gaussian integral.
* The theoretical value should converge to pi ** (d / 2) as N approaches infinity.
* 1. Draw samples
* 2. Normalization
* 3. Evaluate f(x_i)
* 4. MC estimator
*/
int sycl_monte_carlo(const ParsedArgs& args) {
    std::cout << "\nProblem size N:     " << args.N;
    std::cout << "Dimension d:        " << args.d;

    sycl::queue q{
        sycl::gpu_selector_v,
        sycl::property_list{sycl::property::queue::enable_profiling{}}
    };

    int err = gpu_selector(q);
    if (err) {
        std::exit(EXIT_FAILURE);
    }

    double warmup_ms = 0.0;
    (void)mc_kernel(q, 42, args.N, args.d, args.bound, &warmup_ms);
    q.wait();

    double kernel_ms = 0.0;
    auto start = std::chrono::steady_clock::now();
    float I_estimate = mc_kernel(q, 42, args.N, args.d, args.bound, &kernel_ms);
    auto end = std::chrono::steady_clock::now();

    double framework_ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::printf("Estimate:          %.6f\n", I_estimate);
    std::printf("Kernel time:       %.6f ms\n", kernel_ms);
    std::printf("Framework time:    %.6f ms\n", framework_ms);
    return 0;
}

float cg_kernel() {

}

/*
* Using Conjugate Gradient to solve -d^2u/dx^2 = sin(pi * x) with zero boundary conditions
* whose exact solution is known to be u = sin(pi*x)/pi^2. Initial x is set to 0.
* 1. Build the sparse Laplacian matrix A
* 2. Calculate the solution b
* 3. Iterate until convergence
*/
int sycl_conjugate_gradient(const ParsedArgs& args) {
    std::cout << "\nProblem size N:           " << args.N;
    std::cout << "Maximum iteration count:  " << args.maxiter;

    sycl::queue q{
        sycl::gpu_selector_v,
        sycl::property_list{sycl::property::queue::enable_profiling{}}
    };

    int err = gpu_selector(q);
    if (err) {
        std::exit(EXIT_FAILURE);
    }

    double warmup_ms = 0.0;
    (void)cg_kernel();
    q.wait();

    double kernel_ms = 0.0;
    auto start = std::chrono::steady_clock::now();
    float x = cg_kernel();
    auto end = std::chrono::steady_clock::now();

    double framework_ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::printf("Estimate:          %.6f\n", x);
    std::printf("Kernel time:       %.6f ms\n", kernel_ms);
    std::printf("Framework time:    %.6f ms\n", framework_ms);
    return 0;


    // const size_t X = 4;
    // std::vector<int> result(X*X*X, -1);

    // auto timing = timed_kernel(sycl::range<3>(X,X,X), [&](sycl::id<3> idx) {
    //     size_t x = idx[0];
    //     size_t y = idx[1];
    //     size_t z = idx[2];

    //     size_t linear = x * (X*X) + y * X + z;
    //     result[linear] = static_cast<int>(x * 100 + y * 10 + z);
    // });

    // Check results
    // bool passed = true;

    // for (size_t x = 0; x < X; ++x) {
    //     for (size_t y = 0; y < X; ++y) {
    //         for (size_t z = 0; z < X; ++z) {

    //             size_t linear =
    //                 x * (X * X) +
    //                 y * X +
    //                 z;

    //             int expected =
    //                 static_cast<int>(x * 100 + y * 10 + z);

    //             if (result[linear] != expected) {
    //                 std::cout
    //                     << "FAIL: ("
    //                     << x << ", " << y << ", " << z
    //                     << ") = " << result[linear]
    //                     << ", expected " << expected << '\n';

    //                 passed = false;
    //             }
    //         }
    //     }
    // }

    // std::cout << (passed
    //     ? "SANITY TEST PASSED\n"
    //     : "SANITY TEST FAILED\n");

    // std::cout << "CPU time: "
    //           << timing.cpu_miliseconds
    //           << " ms\n";

    // std::cout << "GPU time: "
    //           << timing.gpu_miliseconds
    //           << " ms\n";

    // return passed ? 0 : 1;
}

float dcs_kernel() {}

/*
* Implementation of the Direct Coulomb Summation.
* Defines atoms as an array of [x, y, z, q] where x, y, z are the position and q is the charge.
* 1. Sample random atom values
* 2. Calculate U = k_e * (Ei qi/ri)
*/
int sycl_dcs(const ParsedArgs& args) {
    std::printf("\nGrid size:              [%d, %d, %d]\n", args.grid_size, args.grid_size, args.grid_size);
    std::printf("Amount of particles:    %d\n", args.particle_count);

    sycl::queue q{
        sycl::gpu_selector_v,
        sycl::property_list{sycl::property::queue::enable_profiling{}}
    };

    int err = gpu_selector(q);
    if (err) {
        std::exit(EXIT_FAILURE);
    }

    double warmup_ms = 0.0;
    (void)dcs_kernel();
    q.wait();

    double kernel_ms = 0.0;
    auto start = std::chrono::steady_clock::now();
    float q = dcs_kernel();
    auto end = std::chrono::steady_clock::now();

    double framework_ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::printf("Estimate:          %.6f\n", q);
    std::printf("Kernel time:       %.6f ms\n", kernel_ms);
    std::printf("Framework time:    %.6f ms\n", framework_ms);
    return 0;
}

float spectral_kernel() {}

/*
*/
int sycl_spectral(const ParsedArgs& args) {
    std::printf("\nGrid size:          [%d, %d, %d]", args.grid_size, args.grid_size, args.grid_size);

    sycl::queue q{
        sycl::gpu_selector_v,
        sycl::property_list{sycl::property::queue::enable_profiling{}}
    };

    int err = gpu_selector(q);
    if (err) {
        std::exit(EXIT_FAILURE);
    }

    double warmup_ms = 0.0;
    (void)spectral_kernel();
    q.wait();

    double kernel_ms = 0.0;
    auto start = std::chrono::steady_clock::now();
    float u = spectral_kernel();
    auto end = std::chrono::steady_clock::now();

    double framework_ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::printf("Estimate:          %.6f\n", u);
    std::printf("Kernel time:       %.6f ms\n", kernel_ms);
    std::printf("Framework time:    %.6f ms\n", framework_ms);
    
    return 0;
}

float spmv_kernel(sycl::queue& q, std::size_t N, double* kernel_ms) {
    float* sum = sycl::malloc_device<float>(1, q);
    
    sycl::event ev1 = q.submit([&](sycl::handler& h) {
        auto reduction = sycl::reduction(
            sum, sycl::plus<float>(),
            sycl::property::reduction::initialize_to_identity{}
        );

        h.parallel_for(
            sycl::range<1>(N), reduction,
            [=](sycl::id<1> id, auto& reducer) {
                //
        });
    });


    sycl::event ev2 = q.submit([&](sycl::handler h) {
        h.depends_on(ev1);

        h.parallel_for(sycl::range<1>(N), [=]() {

        });
    });
    ev2.wait();

    auto t_start = ev2.get_profiling_info<sycl::info::event_profiling::command_start>();
    auto t_end = ev2.get_profiling_info<sycl::info::event_profiling::command_end>();
    *kernel_ms = static_cast<double>(t_end - t_start) * 1e-6;

    return ;
}

/*
* Calculates the sparse matrix vector multiplication y = Ax.
* The kernel utilizes power iteration to increase its duration.
*/
int sycl_spmv(const ParsedArgs& args) {
    sycl::queue q{
        sycl::gpu_selector_v,
        sycl::property_list{sycl::property::queue::enable_profiling{}}
    };

    int err = gpu_selector(q);
    if (err) {
        std::exit(EXIT_FAILURE);
    }

    // double warmup_ms = 0.0;
    // (void)spmv_kernel();
    // q.wait();

    // double kernel_ms = 0.0;
    // auto start = std::chrono::steady_clock::now();
    // float y = spmv_kernel();
    // auto end = std::chrono::steady_clock::now();

    // double framework_ms = std::chrono::duration<double, std::milli>(end - start).count();

    // std::printf("Estimate:          %.6f\n", y);
    // std::printf("Kernel time:       %.6f ms\n", kernel_ms);
    // std::printf("Framework time:    %.6f ms\n", framework_ms);

    return 0;
}

/*
*/
int sycl_stencil(const ParsedArgs& args) {
    return 0;
}
