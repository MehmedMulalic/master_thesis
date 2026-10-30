# env -u LD_LIBRARY_PATH julia --project=. main.jl --workload X
include("Utils.jl")

using .Utils
using BenchmarkTools
using CUDA
using Random

Random.seed!()
CUDA.allowscalar(false)
args = parse_cli()

# warm up runs
CUDA.@sync julia_monte_carlo(args)
CUDA.@sync julia_conjugate_gradient(args)

results = @benchmark CUDA.@sync(julia_monte_carlo($args)) samples=10 evals=1
CUDA.synchronize()
profile = CUDA.@profile julia_monte_carlo(args)

display(results)
display(profile)
