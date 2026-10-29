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

"""
Monte Carlo integration of the Gaussian integral.
The theoretical value should converge to pi ** (d / 2) as N approaches infinity.

1. Draw samples
2. Normalization
3. Evaluate f(x_i)
4. MC estimator
"""
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

"""
Using Conjugate Gradient to solve -d^2u/dx^2 = sin(pi * x) with zero boundary conditions
whose exact solution is known to be u = sin(pi*x)/pi^2.

Initial x is set to 0.
1. Build the sparse Laplacian matrix A
2. Calculate the solution b
3. Iterate until convergence
"""
function julia_conjugate_gradient()
    println("pass")
end

"""
Implementation of the Direct Coulomb Summation.

Defines atoms as an array of [x, y, z, q] where x, y, z are the position and q is the charge.
1. Sample random atom values
2. Calculate U = k_e * (Ei qi/ri)
"""
function julia_dcs()
    println("pass")
end

"""
Solve the 3D heat equation using a spectral method.
1. Transform the initial temperature field to frequency space
2. Evolve the frequency modes over time and transform back to the spatial domain
"""
function julia_spectral()
    println("pass")
end

"""
Calculates the sparse matrix vector multiplication y = Ax.

The kernel utilizes power iteration to increase its duration.
"""
function julia_spmv()
    println("pass")
end

"""
Explicit 7-point stencil for the 3D heat equation.
Assumes equal spacing in all three dimensions (dx=dy=dz). Will not work for unequal spacing.

Stability, for equal spacing, is defined as r = alpha*dt / spacing^2 for which, in 3D, must be r <= 1/6.

The stencil is calculated by convolving the laplacian kernel.
Uses the 'reflect' mode as per defined Neumann boundary condition.
"""
function julia_stencil()
    println("pass")
end

end
