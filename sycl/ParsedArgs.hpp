#pragma once

#include <string>

//? Treba li dodati USM/buffer opciju? kao dodatan validation?
struct ParsedArgs {
    std::string workload;
    int N = 100;
    int d = 10;
    float bound = 6.0;
    int maxiter = 1000;
    float tol = 1e-10;
    int particle_count = 100;
    int grid_size;
    int iterations = 200;
};
