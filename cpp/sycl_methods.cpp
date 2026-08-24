#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <string>
#include <sycl/sycl.hpp>

int gpu_selector(sycl::queue q) {
    // GPU device selection
    try {
        q = sycl::queue{sycl::gpu_selector_v, sycl::property::queue::in_order()};
        if (!q.get_device().has(sycl::aspect::gpu))
            throw std::runtime_error("No GPU device available");

        std::cout << "Running on " << q.get_device().get_info<sycl::info::device::name>() << std::endl;
    } catch (std::exception &e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
}

void kernel() {

}

int sycl_monte_carlo() {
    sycl::queue q;
    int err = gpu_selector(q);
    if (err) {
        std::exit(EXIT_FAILURE);
    }

    auto start = std::chrono::high_resolution_clock::now();

    kernel();

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Simulation completed in " << elapsed.count() << " seconds." << std::endl;

    return 0;
}

int sycl_conjugate_gradient() {}
int sycl_dcs() {}
int sycl_spectral() {}
int sycl_spmv() {}
int sycl_stencil() {}
