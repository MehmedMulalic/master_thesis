include("Utils.jl")

using .Utils
using LinearAlgebra, SparseArrays
using CUDA, CUDA.CUSPARSE
using Krylov
using Random
using ArgParse

Random.seed!()
CUDA.allowscalar(false)
args = parse_cli()

X = julia_monte_carlo(args)
println(X)
