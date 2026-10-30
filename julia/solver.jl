# env -u LD_LIBRARY_PATH julia --project=. main.jl --workload X
include("Utils.jl")

using .Utils
using BenchmarkTools
using CUDA
using Random

const SAMPLES = 10

const NUM_METHOD = Dict(
    "mc"        => "Monte Carlo",
    "cg"        => "Conjugate Gradient",
    "dcs"       => "Direct Coulomb Summation",
    "spectral"  => "Spectral",
    "spmv"      => "Sparse Matrix-Vector Multiplication",
    "stencil"   => "Stencil",
)

const workload_map = Dict(
    "mc"        => julia_monte_carlo,
    "cg"        => julia_conjugate_gradient,
    "dcs"       => julia_dcs,
    "spectral"  => julia_spectral,
    "spmv"      => julia_spmv,
    "stencil"   => julia_stencil,
)

Random.seed!()
CUDA.allowscalar(false)
args = parse_cli()

implementation = workload_map[args["workload"]]

# warm up runs
CUDA.@sync implementation(args)
_ = @benchmark CUDA.@sync(implementation($args)) samples=SAMPLES evals=1

results = @benchmark CUDA.@sync(implementation($args)) samples=SAMPLES evals=1
CUDA.synchronize()
profile = CUDA.@profile implementation(args)

display(results)
display(profile)
