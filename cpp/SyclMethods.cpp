#include <iostream>
#include <random>
#include <vector>
#include <chrono>
#include <cmath>
#include <string>
#include <sycl/sycl.hpp>
#include "SyclMethods.hpp"

int sycl_monte_carlo(const ParsedArgs& args) {
    // sycl::queue q{
    //     sycl::property::queue::enable_profiling{}
    // };

    // int err = gpu_selector(q);
    // if (err) {
    //     std::exit(EXIT_FAILURE);
    // }

    // auto timing = timed_kernel(q, [=](sycl::id<> idx) {});

    // std::cout << "CPU time: " << timing.cpu_miliseconds << " ms\n";
    // std::cout << "GPU time: " << timing.gpu_miliseconds << " ms\n";

    return 0;
}

int sycl_conjugate_gradient(const ParsedArgs& args) {
    const size_t X = 4;
    std::vector<int> result(X*X*X, -1);

    auto timing = timed_kernel(sycl::range<3>(X,X,X), [&](sycl::id<3> idx) {
        size_t x = idx[0];
        size_t y = idx[1];
        size_t z = idx[2];

        size_t linear = x * (X*X) + y * X + z;
        result[linear] = static_cast<int>(x * 100 + y * 10 + z);
    });

    // Check results
    bool passed = true;

    for (size_t x = 0; x < X; ++x) {
        for (size_t y = 0; y < X; ++y) {
            for (size_t z = 0; z < X; ++z) {

                size_t linear =
                    x * (X * X) +
                    y * X +
                    z;

                int expected =
                    static_cast<int>(x * 100 + y * 10 + z);

                if (result[linear] != expected) {
                    std::cout
                        << "FAIL: ("
                        << x << ", " << y << ", " << z
                        << ") = " << result[linear]
                        << ", expected " << expected << '\n';

                    passed = false;
                }
            }
        }
    }

    std::cout << (passed
        ? "SANITY TEST PASSED\n"
        : "SANITY TEST FAILED\n");

    std::cout << "CPU time: "
              << timing.cpu_miliseconds
              << " ms\n";

    std::cout << "GPU time: "
              << timing.gpu_miliseconds
              << " ms\n";

    return passed ? 0 : 1;
}

int sycl_dcs(const ParsedArgs& args) {
    return 0;
}

int sycl_spectral(const ParsedArgs& args) {
    return 0;
}

int sycl_spmv(const ParsedArgs& args) {
    // auto kernel = [](sycl::queue q) {
    //     auto start = std::chrono::high_resolution_clock::now();

    //     q.parallel_for(sycl::range<>(), [=](sycl::id<> idx) {
            
    //     });

    //     auto end = std::chrono::high_resolution_clock::now();

    //     std::chrono::duration<double> elapsed = end - start;
    //     return elapsed;
    // };

    // std::mt19937 gen(RANDOM_SEED);
    // int N           = 100'000;
    // double decimal  = 0.001;

    // A_host =
    // A_gpu =
    // x_gpu =
    // y_gpu =
    // y_cpu =

    return 0;
}

int sycl_stencil(const ParsedArgs& args) {
    // auto kernel = [](sycl::queue q) {
    //     auto start = std::chrono::high_resolution_clock::now();

    //     q.parallel_for(sycl::range<>(), [=](sycl::id<> idx) {
            
    //     });

    //     auto end = std::chrono::high_resolution_clock::now();

    //     std::chrono::duration<double> elapsed = end - start;
    //     return elapsed;
    // };

    return 0;
}
