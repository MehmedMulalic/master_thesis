module Utils

using LinearAlgebra, SparseArrays
using CUDA, CUDA.CUSPARSE
using Random, Statistics
using Printf
using Krylov
using ArgParse

export parse_cli, julia_monte_carlo, julia_conjugate_gradient
export julia_dcs, julia_spectral, julia_spmv, julia_stencil

const NUM_METHOD = Dict(
    "mc"        => "Monte Carlo",
    "cg"        => "Conjugate Gradient",
    "dcs"       => "Direct Coulomb Summation",
    "spectral"  => "Spectral",
    "spmv"      => "Sparse Matrix-Vector Multiplication",
    "stencil"   => "Stencil",
)

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

    args = parse_args(s)

    if args["workload"] in ["spectral", "spmv", "stencil"]
        if isnothing(args["grid-size"]) || args["grid-size"] <= 0
            error("Grid size must be a positive integer.")
        end
    end

    return args
end

function julia_monte_carlo(args, seed::Int64=0)
    @printf("Problem size N:    %d\n", args["N"])
    @printf("Dimension d:       %d\n", args["d"])
    
    if seed != 0
        Random.seed!(seed)
    end

    # Random array with bounds [-bound, bound),
    # defined as lb + (up - lb) * A
    X = -args["bound"] .+ (2*args["bound"]) .* CUDA.rand(Float32, (args["N"], args["d"]))
    sq_norms = sum(X.^2, dims=2)
    f_vals = exp.(-sq_norms)
    
    volume = (2*args["bound"])^args["d"]
    I_estimate = volume * mean(f_vals)

    return
end

function julia_conjugate_gradient()
    println("pass")
end

function julia_dcs()
    println("pass")
end

function julia_spectral()
    println("pass")
end

function julia_spmv()
    println("pass")
end

function julia_stencil()
    println("pass")
end

end
