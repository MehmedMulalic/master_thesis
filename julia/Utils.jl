module Utils

using LinearAlgebra, SparseArrays
using CUDA, CUDA.CUSPARSE
using Krylov
using Random
using ArgParse

export parse_cli, julia_monte_carlo

function parse_cli()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--workload"
            range_tester = x -> x in ["mc", "cg", "dcs", "spectral", "spmv", "stencil"]
            arg_type = String
            required = true
            help = "Numerical method to calculate: " *
                "Monte Carlo, Conjugate Gradient, Direct Coulomb Simulation, " *
                "Spectral, Sparse Matrix Vector, Stencil."

        # "--validate"
        #     action = "store_true"
        #     help = "Run a CPU reference calculation and compare the result. BROKEN"

        ### Workload specific arguments

        # Monte Carlo
        "--N"
            arg_type = Int
            default = 100
            help = "Problem size N for the Monte Carlo integration. Default is 100."

        "--d"
            arg_type = Int
            default = 10
            help = "Dimension d for the Monte Carlo integration. Default is 10."

        "--bound"
            arg_type = Float64
            default = 6.0
            help = "Domain bound used for volume for Monte Carlo. Default value is 6.0, " *
                "accurate enough for double-point floating values."

        # Conjugate Gradient
        "--maxiter"
            arg_type = Int
            default = 1000
            help = "Maximum number of iterations for the Conjugate Gradient. " *
                "Default is 1000."

        "--tol"
            arg_type = Float64
            default = 1e-10
            help = "Convergence tolerance for the Conjugate Gradient. Default is 1e-10."

        # DCS
        "--particle-count"
            arg_type = Int
            default = 100
            help = "Amount of particles for the DCS. Default is 100."

        # Heat Equation
        "--grid-size"
            arg_type = Int
            help = "Number of grid points along each spatial dimension needed to solve the Heat Equation."

        "--iterations"
            arg_type = Int
            default = 200
            help = "Amount of iterations for Sparse Matrix-Vector Multiplication. " *
                "Default is 200."
    end

    return parse_args(s)
end

function julia_monte_carlo(args, seed::Int64=0)
    if seed != 0
        Random.seed!(seed)
    end

    X = CUDA.rand(Float32, (args["N"], args["d"]))

    return X
end

end