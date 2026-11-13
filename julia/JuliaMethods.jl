module JuliaMethods

using LinearAlgebra, SparseArrays
using CUDA, CUDA.CUSPARSE
using Statistics
using Printf
using Krylov
using FFTW

export julia_monte_carlo, julia_conjugate_gradient, julia_dcs
export julia_spectral, julia_spmv, julia_stencil

const TypeAlias = Float32
const MAX_NNZ = 4_000_000_000   # 32GB VRAM

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
        CUDA.seed!(seed)
    end

    # Random array with bounds [-bound, bound),
    # defined as lb + (up - lb) * A
    X = -args["bound"] .+ (2*args["bound"]) .* CUDA.rand(TypeAlias, (args["N"], args["d"]))
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
function julia_conjugate_gradient(args)
    @printf("Problem size N:            %d\n", args["N"])
    @printf("Maximum iteration count:   %d\n", args["maxiter"])

    h = 1f0 / (args["N"] + 1)
    main_diag = fill(2f0, args["N"]) #? treba li ovo biti random a ne samo 2? jel 2 samo ono user-defined?
    off_diag = fill(-1f0, args["N"]-1)
    A = spdiagm(-1 => off_diag, 0 => main_diag, 1 => off_diag)
    A = A ./ (h^2)

    grid = range(0, 1, length=args["N"]+2)[2:end-1]
    u = sin.(pi .* grid)
    b = A * u

    devA = CUDA.CuSparseMatrixCSR(A)
    devb = CuArray(b)
    _ = Krylov.cg(devA, devb; rtol=1e-10, itmax=args["maxiter"])

    x, stats = Krylov.cg(devA, devb; rtol=1e-10, itmax=args["maxiter"])

    return
end

"""
Implementation of the Direct Coulomb Summation.

Defines atoms as an array of [x, y, z, q] where x, y, z are the position and q is the charge.
1. Sample random atom values
2. Calculate U = k_e * (Ei qi/ri)
"""
function julia_dcs(args)
    @printf("Grid size:              [%d, %d, %d]\n", args[grid-size], args[grid-size], args[grid-size])
    @printf("Amount of particles:    %d\n", args["particle-count"])
end

"""
Solve the 3D heat equation using a spectral method.
1. Transform the initial temperature field to frequency space
2. Evolve the frequency modes over time and transform back to the spatial domain
"""
function julia_spectral(args)
    @printf("Grid size:              [%d, %d, %d]\n", args["grid-size"], args["grid-size"], args["grid-size"])

    function kernel(u0, alpha, k_squared, dt, num_steps)
        u_hat = dct(u0, (1, 2, 3))
        evolution_operator = exp.(alpha .* k_squared .* dt)

        for _ in 1:num_steps
            u_hat .*= evolution_operator
        end
        
        u_final = idct(u_hat, (1, 2, 3))
        return real.(u_final)
    end

    A = 100f0                       # Peak temperature (°C)
    sigma_initial = 0f2             # Initial width (m)
    alpha = 1.6563f-4               # Thermal diffusivity of pure silver (99.9%) (m^2/s)

    x0 = y0 = z0 = 0f5              # Center
    Lx = Ly = Lz = 1f0              # Domain limit (metres)
    nx = ny = nz = args["grid-size"]

    dx = Lx / (nx - 1)              # x spacing
    dy = Ly / (ny - 1)              # y spacing
    dz = Lz / (nz - 1)              # z spacing
    dt = 3f-4                       # Time step (s)
    t_final = 60                    # Final time step (s)
    num_steps = round(Int32, t_final / dt)

    devx = CuArray{TypeAlias}(undef, nx)
    devy = CuArray{TypeAlias}(undef, ny)
    devz = CuArray{TypeAlias}(undef, nz)
    devx .= range(0f0, Lx, nx)
    devy .= range(0f0, Ly, ny)
    devz .= range(0f0, Lz, nz)
    
    # NumPy's meshgrid equates to Julia's reshaping which is
    # automatically broadcasted during u0 and frequency grid calculations.
    X = reshape(devx, :, 1, 1)
    Y = reshape(devy, 1, :, 1)
    Z = reshape(devz, 1, 1, :)

    # Initial condition (Gaussian bump)
    u0 = A .* exp.(
        -(
            (X .- x0).^2 .+
            (Y .- y0).^2 .+
            (Z .- z0).^2
        ) ./ (2 * sigma_initial^2)
    )
    N = size(u0, 1)

    # Frequency grid
    kx = pi .* (0:N-1) ./ N
    ky = pi .* (0:N-1) ./ N
    kz = pi .* (0:N-1) ./ N

    KX = reshape(kx, :, 1, 1)
    KY = reshape(ky, 1, :, 1)
    KZ = reshape(kz, 1, 1, :)
    
    # Eigenvalues for Neumann BC
    k_squared = (
        2 .* (cos.(KX) .- 1) ./ dx^2 .+
        2 .* (cos.(KY) .- 1) ./ dy^2 .+
        2 .* (cos.(KZ) .- 1) ./ dz^2
    )
    
    return
end

"""
Calculates the sparse matrix vector multiplication y = Ax.

The kernel utilizes power iteration to increase its duration.
"""
function julia_spmv(args)
    @printf("Amount of iterations:       %d\n", args["iterations"])

    N = 300_000
    density = 0.001f0
    nnz = round(Int, N * N * density)
    if nnz > MAX_NNZ
        throw(error(
            "Requested matrix would contain $nnz nonzeros." *
            "Maximum allowed is $MAX_NNZ."
        ))
    end

    CUDA.seed!(42)

    x = CUDA.rand(Float32, N)
    A = sprand(Float32, N, N, density)
    dA = CuSparseMatrixCSR(A)

    _ = dA * x
    y = dA * x
end

"""
Explicit 7-point stencil for the 3D heat equation.
Assumes equal spacing in all three dimensions (dx=dy=dz). Will not work for unequal spacing.

Stability, for equal spacing, is defined as r = alpha*dt / spacing^2 for which, in 3D, must be r <= 1/6.

The stencil is calculated by convolving the laplacian kernel.
Uses the 'reflect' mode as per defined Neumann boundary condition.
"""
function julia_stencil(args)
    println("pass")
end

end
