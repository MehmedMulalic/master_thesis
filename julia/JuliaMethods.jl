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
const MAX_NNZ = 4_000_000_000     # 32GB VRAM
const K_E = 8.987551786214e9      # https://en.wikipedia.org/wiki/Coulomb%27s_law

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
Implementation which uses Conjugate Gradient to solve -d^2u/dx^2 = sin(pi * x) with
Dirichlet boundary conditions, u(0) = u(1) = 1, whose exact solution is known to be u = sin(pi*x) / pi**2.

Initial x is set to 0.
1. Build the sparse Laplacian matrix A
2. Calculate the solution b
3. Iterate until convergence
"""
function julia_conjugate_gradient(args)
    @printf("Problem size N:            %d\n", args["N"])
    @printf("Maximum iteration count:   %d\n", args["maxiter"])

    # Laplacian representation with spacing h
    # -d^2u/dx^2 = -u[i-1] + 2u[i] - u[i+1], thus off-diagonals are -1 and diagonals are 2
    h = 1.0f0 / (args["N"] + 1)
    main_diag = fill(2.0f0, args["N"])
    off_diag = fill(-1.0f0, args["N"]-1)

    # A = -d^2/dx^2; Au = -d^2u/dx^2
    A = spdiagm(-1 => off_diag, 0 => main_diag, 1 => off_diag)
    A ./= (h^2)

    # Grid skips the 0 boundary values
    # Au = sin(pi * x); b = sin(pi * x); Au = b
    grid = CuArray{TypeAlias}(undef, args["N"+2])
    grid = range(0.0f0, 1.0f0, length=args["N"]+2)[2:end-1]
    b = sin.(pi .* grid)

    devA = CUDA.CuSparseMatrixCSR(A)
    devb = CuArray(b)
    _ = Krylov.cg(devA, devb; rtol=1e-10, itmax=args["maxiter"])

    x, stats = Krylov.cg(devA, devb; rtol=1e-10, itmax=args["maxiter"])

    return
end

"""
Implementation of the Direct Coulomb Summation.
Due to a large 3D grid, batch processing is required with a default value of 65,536.

Defines atoms as an array of [x, y, z, q] where x, y, z are the position and q is the charge.
1. Sample random atom values
2. Calculate U = k_e * (Ei qi/ri)
"""
function julia_dcs(args; seed::Int64=0)
    @printf("Grid size:              [%d, %d, %d]\n", args["grid-size"], args["grid-size"], args["grid-size"])
    @printf("Amount of particles:    %d\n", args["particle-count"])

    function kernel(atoms, result; chunk::Int64=1 << 16, k_e = K_E)
        nx = ny = nz = args["grid-size"]
        total = nx * ny * nz

        pos = @view atoms[:, 1:3]
        q = @view atoms[:, 4]

        for start in 1:chunk:total
            stop = min(start + chunk, total) #? minus jedan?
            idx = CuArray(start:stop)

            gx = Float32.(idx .% nx)
            gy = Float32.((idx .// nx) % ny)
            gz = Float32.(idx .// (nx * ny))

            # Broadcasting: shape (chunk_size, 1) - shape (1, particle_count)
            # shape (chunk_size, particle_count)
            dx = reshape(gx, :, 1) .- reshape(pos[:, 1], 1, :)
            dy = reshape(gy, :, 1) .- reshape(pos[:, 2], 1, :)
            dz = reshape(gz, :, 1) .- reshape(pos[:, 3], 1, :)

            r = sqrt.(dx.^2 .+ dy.^2 .+ dz.^2)
            r = ifelse.(r .> 0, r, 1f0) # Avoid division by zero

            contrib = ifelse.(
                r .> 0,
                reshape(q, 1, :) ./ r,
                0f0
            )
            result[start:start+length(idx)] .= k_e .* vec(sum(contrib, dims=2))
        end
        
        return reshape(result, nz, ny, nx)
    end
    
    if seed != 0
        CUDA.seed!(seed)
    end

    atoms = empty(TypeAlias, args["particle-count"], 4)
    atoms[:, 1:3] .= CUDA.rand(TypeAlias, 0, args["grid-size"], args["particle-count"], 3)
    atoms[:, 4] .= 1f-5 .* CUDA.rand(args["particle-count"]) .- 5f-6
    
    total = args["grid-size"]^3
    d_atoms = cp.asarray(atoms)
    result = CuArray{TypeAlias}(undef, total)

    d_q = kernel(d_atoms, result)
    q = Array(d_q)

    return sum(q)
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

    A = 100.0f0                     # Peak temperature (°C)
    sigma_initial = 0.2f            # Initial width (m)
    alpha = 1.6563f-4               # Thermal diffusivity of pure silver (99.9%) (m^2/s)

    x0 = y0 = z0 = 0.5f             # Center
    Lx = Ly = Lz = 1.0f0            # Domain limit (metres)
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
    devx .= range(0.0f0, Lx, nx)
    devy .= range(0.0f0, Ly, ny)
    devz .= range(0.0f0, Lz, nz)
    
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
    A = 100.0f0                     # Peak temperature (°C)
    sigma_initial = 0.2f            # Initial width (m)
    alpha = 1.6563f-4               # Thermal diffusivity of pure silver (99.9%) (m^2/s)

    x0 = y0 = z0 = 0.5f             # Center
    Lx = Ly = Lz = 1.0f0            # Domain limit (metres)
    nx = ny = nz = args["grid-size"]

    dx = Lx / (nx - 1)              # x spacing
    dy = Ly / (ny - 1)              # y spacing
    dz = Lz / (nz - 1)              # z spacing
    dt = 3f-4                       # Time step (s)
    t_final = 60                    # Final time step (s)
    num_steps = round(Int32, t_final / dt)

    # Stability
    r = alpha * dt / (dx^2)
    if r > 1/6
        throw(StabilityError("Unstable simulation; must be below 0.167. Current r: $(r)"))
    end

    devx = CuArray{TypeAlias}(undef, nx)
    devy = CuArray{TypeAlias}(undef, ny)
    devz = CuArray{TypeAlias}(undef, nz)
    devx .= range(0.0f0, Lx, nx)
    devy .= range(0.0f0, Ly, ny)
    devz .= range(0.0f0, Lz, nz)
    
    # NumPy's meshgrid equates to Julia's reshaping which is
    # automatically broadcasted during u0 and frequency grid calculations.
    X = reshape(devx, :, 1, 1)
    Y = reshape(devy, 1, :, 1)
    Z = reshape(devz, 1, 1, :)

    # Laplacian kernel
    kernel_array = CUDA.zeros(TypeAlias, 3, 3, 3)

    # Above, below (z-axis)
    kernel_array[0, 1, 1] = 1.0f0
    kernel_array[2, 1, 1] = 1.0f0

    # Up, down (y-axis)
    kernel_array[1, 0, 1] = 1.0f0
    kernel_array[1, 2, 1] = 1.0f0

    # Left, right (x-axis)
    kernel_array[1, 1, 0] = 1.0f0
    kernel_array[1, 1, 2] = 1.0f0

    # Center
    kernel_array[1, 1, 1] = -6.0f0
    kernel_array ./= dx^2

    # Initial condition (Gaussian bump)
    u0 = A .* exp.(
        -(
            (X .- x0).^2 .+
            (Y .- y0).^2 .+
            (Z .- z0).^2
        ) ./ (2 * sigma_initial^2)
    )

    # Kernel
    _ = kernel(u0, alpha, kernel_array, dt, num_steps)
    u = kernel(u0, alpha, kernel_array, dt, num_steps)
    
    return
end

end
