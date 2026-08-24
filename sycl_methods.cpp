#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <string>
#include <CLI11.hpp>
#include <sycl/sycl.hpp>

struct ParsedArgs {
    int grid_size = 0;
    std::string framework;
    std::string workload;
};

ParsedArgs parse_args(int argc, char** argv) {
    CLI::App app{""};
    ParsedArgs args;

    app.add_option("--grid-size", args.grid_size, "Number of grid points along each spatial dimension.")
        ->required();

    app.add_option("--framework", args.framework, "Computational framework to use.")
        ->required()
        ->check(CLI::IsMember({"cupy", "numba"}));

    app.add_option("--workload", args.workload,
                    "Numerical method to calculate: "
                    "Monte Carlo, Conjugate Gradient, Direct Coulomb Simulation, "
                    "Spectral, Sparse Matrix Vector, Stencil.")
        ->required()
        ->check(CLI::IsMember({"mc", "cg", "dcs", "spectral", "spmv", "stencil"}));

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError &e) {
        std::exit(app.exit(e));
    }

    return args;
}

int main(int argc, char** argv) {
    ParsedArgs args = parse_args(argc, argv);

    std::cout << args.grid_size << "\n";
    std::cout << args.framework << "\n";
    std::cout << args.workload << "\n";

    return 0;
}