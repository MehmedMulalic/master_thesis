import cupy as cp
import numpy as np
import time
import StabilityException

MAX_NNZ = 4_000_000_000     # 32GB VRAM
K_E = 8.987551786214e9      # https://en.wikipedia.org/wiki/Coulomb%27s_law

def cupy_monte_carlo(args, kernel_start_event):
    """
    Monte Carlo integration of the Gaussian integral.
    The theoretical value should converge to pi ** (d / 2) as N approaches infinity.

    1. Draw samples
    2. Normalization
    3. Evaluate f(x_i)
    4. MC estimator
    """

    print(f"\nProblem size N:     {args.N}")
    print(f"Dimension d:        {args.d}")

    def kernel(seed=None):
        if seed is None:
            cp.random.seed()
        else:
            cp.random.seed(seed)

        X = cp.random.uniform(-args.bound, args.bound, size=(args.N, args.d), dtype=cp.float32)
        sq_norms = cp.sum(X**2, axis=1)
        f_vals = cp.exp(-sq_norms)

        volume = (2 * args.bound) ** args.d
        I_estimate = volume * cp.mean(f_vals)

        return I_estimate

    _ = kernel()

    start = cp.cuda.Event()
    end = cp.cuda.Event()

    kernel_start_event.set()
    start.record()

    I_estimate = kernel(42)

    end.record()
    end.synchronize()

    elapsed = cp.cuda.get_elapsed_time(start, end)

    return elapsed, I_estimate

def cupy_conjugate_gradient(args, kernel_start_event):
    """
    Using Conjugate Gradient to solve -d^2u/dx^2 = sin(pi * x) with zero boundary conditions
    whose exact solution is known to be u = sin(pi*x)/pi^2. Initial x is set to 0.
    1. Build the sparse Laplacian matrix A
    2. Calculate the solution b
    3. Iterate until convergence
    """

    print(f"\nProblem size N:             {args.N}")
    print(f"Maximum iteration count:    {args.maxiter}")

    from cupyx.scipy.sparse import diags as sparse_diags
    from cupyx.scipy.sparse.linalg import cg

    h = 1.0 / (args.N + 1)
    main_diag = 2.0 * cp.ones(args.N)
    off_diag = -1.0 * cp.ones(args.N - 1)
    A = sparse_diags([off_diag, main_diag, off_diag], offsets=[-1, 0, 1], format="csr")
    A = A / (h ** 2)

    grid = cp.linspace(0, 1, args.N + 2)[1:-1]
    u = cp.sin(cp.pi * grid)
    b = A @ u

    # Kernel
    _ = cg(A, b, rtol=1e-10, maxiters=args.maxiter)

    start = cp.cuda.Event()
    end = cp.cuda.Event()

    kernel_start_event.set()
    start.record()

    x = cg(A, b, rtol=1e-10, maxiter=args.maxiter)
    
    end.record()
    end.synchronize()

    elapsed = cp.cuda.get_elapsed_time(start, end)

    return elapsed, x

def cupy_dcs(args, kernel_start_event):
    """
    Implementation of the Direct Coulomb Summation.
    Defines atoms as an array of [x, y, z, q] where x, y, z are the position and q is the charge.
    1. Sample random atom values
    2. Calculate U = k_e * (Ei qi/ri)
    """

    print(f"\nGrid size:              [{args.grid_size}, {args.grid_size}, {args.grid_size}]")
    print(f"Amount of particles:    {args.particle_count}")

    def kernel(atoms, chunk=1 << 16, k_e=K_E):
        nx = ny = nz = args.grid_size
        total = nx * ny * nz
        result = cp.empty(total, dtype=cp.float32)
        pos, q = atoms[:, :3], atoms[:, 3]

        for start in range(0, total, chunk):
            idx = cp.arange(start, min(start + chunk, total), dtype=cp.int32)
            gx = (idx % nx).astype(cp.float32)
            gy = ((idx // nx) % ny).astype(cp.float32)
            gz = (idx // (nx * ny)).astype(cp.float32)

            dx = gx[:, None] - pos[None, :, 0]
            dy = gy[:, None] - pos[None, :, 1]
            dz = gz[:, None] - pos[None, :, 2]
            r = cp.sqrt(dx*dx + dy*dy + dz*dz)

            contrib = cp.true_divide(q[None, :], r, out=cp.zeros_like(r, dtype=cp.float32), where=(r > 0))
            result[start:start + len(idx)] = k_e * contrib.sum(axis=1)

        return result.reshape(nz, ny, nx)

    rng = np.random.default_rng(42)

    atoms = np.empty((args.particle_count, 4), dtype=np.float32)
    atoms[:, :3] = rng.integers(0, args.grid_size, size=(args.particle_count, 3), endpoint=True)
    atoms[:, 3] = rng.uniform(-5e-6, 5e-6, size=args.particle_count)
    atoms = cp.asarray(atoms)

    # Kernel
    start = cp.cuda.Event()
    end = cp.cuda.Event()

    kernel_start_event.set()
    start.record()

    q = kernel(atoms)

    end.record()
    end.synchronize()

    elapsed = cp.cuda.get_elapsed_time(start, end)
    sum_total = cp.sum(q).item()

    return elapsed, sum_total

def cupy_spectral(args, kernel_start_event):
    """
    Fourier Transformation.
    1.
    2.
    """

    print(f"\nGrid size:          [{args.grid_size}, {args.grid_size}, {args.grid_size}]")

    from cupyx.scipy.fft import dctn, idctn

    def kernel(u0, alpha, k_squared, dt, num_steps):
        u_hat = dctn(u0, norm='ortho')
        evolution_operator = cp.exp(alpha * k_squared * dt)
        
        for _ in range(num_steps):
            u_hat = cp.multiply(u_hat, evolution_operator, out=u_hat)
        
        u_final = idctn(u_hat, norm='ortho').real
        return u_final

    A = 100.0                       # Peak temperature (°C)
    sigma_initial = 0.2             # Initial width (m)
    alpha = 1.6563e-4               # Thermal diffusivity of pure silver (99.9%) (m^2/s)

    x0 = y0 = z0 = 0.5              # Center
    Lx = Ly = Lz = 1.0              # Domain limit (metres)
    nx = ny = nz = args.grid_size

    dx = Lx / (nx - 1)              # x spacing
    dy = Ly / (ny - 1)              # y spacing
    dz = Lz / (nz - 1)              # z spacing
    dt = 0.0003                     # Time step (s)
    t_final = 60                    # Final time step (s)
    num_steps = int(t_final / dt)

    x = cp.linspace(0.0, Lx, nx)
    y = cp.linspace(0.0, Ly, ny)
    z = cp.linspace(0.0, Lz, nz)
    X, Y, Z = cp.meshgrid(x, y, z)

    # Initial condition (Gaussian bump)
    u0 = A * cp.exp( -((X - x0)**2 + (Y - y0)**2 + (Z - z0)**2) / (2 * sigma_initial**2) )
    N = u0.shape[0]
            
    # Frequency grid
    kx = cp.pi * cp.arange(N) / N
    ky = cp.pi * cp.arange(N) / N
    kz = cp.pi * cp.arange(N) / N
    KX, KY, KZ = cp.meshgrid(kx, ky, kz)
    
    # Eigenvalues for Neumann BC
    k_squared = (
        2 * (cp.cos(KX) - 1) / dx**2 +
        2 * (cp.cos(KY) - 1) / dy**2 +
        2 * (cp.cos(KZ) - 1) / dz**2
    )

    # Kernel
    _ = kernel(u0, alpha, k_squared, dt, 1)

    start = cp.cuda.Event()
    end = cp.cuda.Event()

    kernel_start_event.set()
    start.record()

    u_final = kernel(u0, alpha, k_squared, dt, num_steps)

    end.record()
    end.synchronize()

    elapsed = cp.cuda.get_elapsed_time(start, end)
    sum_total = cp.sum(u_final).item()

    return elapsed, sum_total

def cupy_spmv(args, kernel_start_event):
    """
    Calculates the sparse matrix vector multiplication y = Ax.
    The kernel utilizes power iteration to increase its duration.
    """

    print(f"\nAmount of iterations:       {args.iterations}")

    import cupyx.scipy.sparse as csp

    def kernel(A_csr, x):
        for _ in range(args.iterations):
            x = A_csr @ x
            x = x / cp.linalg.norm(x)

        return x

    N = 300_000
    density = 0.01
    nnz = int(N * N * density)
    if nnz > MAX_NNZ:
        raise ValueError(
            f"Requested matrix would contain {nnz:,} nonzeros.\n"
            f"Maximum allowed is {MAX_NNZ:,}."
        )

    cp.random.seed(42)

    rows = cp.random.randint(0, N, size=nnz, dtype=cp.int32)
    cols = cp.random.randint(0, N, size=nnz, dtype=cp.int32)
    vals = cp.random.rand(nnz, dtype=cp.float32)

    A = csp.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
    x = cp.random.rand(N, dtype=cp.float32)

    cp.cuda.Stream.null.synchronize()

    # Kernel
    _ = kernel(A, x)

    start = cp.cuda.Event()
    end = cp.cuda.Event()

    kernel_start_event.set()
    start.record()

    y = kernel(A, x)

    end.record()
    end.synchronize()

    elapsed = cp.cuda.get_elapsed_time(start, end)
    sum_total = cp.sum(y).item()

    return elapsed, sum_total

def cupy_stencil(args, kernel_start_event):
    """
    Explicit 7-point stencil for the 3D heat equation.
    Assumes equal spacing in all three dimensions (dx=dy=dz). Will not work for unequal spacing.
    
    Stability, for equal spacing, is defined as r = alpha*dt / spacing^2 for which, in 3D, must be r <= 1/6.

    The stencil is calculated by convolving the laplacian kernel.
    Uses the 'reflect' mode as per defined Neumann boundary condition.
    """

    from cupyx.scipy.ndimage import convolve

    def kernel(u0, alpha, kernel_array, dt, num_steps):
        for _ in range(num_steps):
            laplacian = convolve(u0, kernel_array)
            u0 += alpha * dt * laplacian

        return u0

    A = 100.0                       # Peak temperature (°C)
    sigma_initial = 0.2             # Initial width (m)
    alpha = 1.6563e-4               # Thermal diffusivity of pure silver (99.9%) (m^2/s)

    x0 = y0 = z0 = 0.5              # Center
    Lx = Ly = Lz = 1.0              # Domain limit (metres)
    nx = ny = nz = args.grid_size

    dx = Lx / (nx - 1)              # x spacing (we assume dx=dy=dz in this case)
    dt = 0.0003                     # Time step (s)
    t_final = 60                    # Final time step (s)
    num_steps = int(t_final / dt)

    # Stability
    r = alpha * dt / dx**2          # Holds only if equal spacing dx=dy=dz
    if (r > 1/6):                   # Coefficient for 3D space
        raise StabilityException(f"Unstable simulation; must be below 0.167. Current r: {r}")

    x = cp.linspace(0.0, Lx, nx)
    y = cp.linspace(0.0, Ly, ny)
    z = cp.linspace(0.0, Lz, nz)
    X, Y, Z = cp.meshgrid(x, y, z)

    # Laplacian kernel
    kernel_array = cp.zeros((3, 3, 3), dtype=cp.float32)

    # Above, below (z-axis)
    kernel_array[0, 1, 1] = 1
    kernel_array[2, 1, 1] = 1

    # Up, down (y-axis)
    kernel_array[1, 0, 1] = 1
    kernel_array[1, 2, 1] = 1

    # Left, right (x-axis)
    kernel_array[1, 1, 0] = 1
    kernel_array[1, 1, 2] = 1

    # Center
    kernel_array[1, 1, 1] = -6
    kernel_array /= dx**2

    # Initial condition (Gaussian bump)
    u0 = A * cp.exp( -((X - x0)**2 + (Y - y0)**2 + (Z - z0)**2) / (2 * sigma_initial**2) )
            
    # Kernel
    _ = kernel(u0, alpha, kernel_array, dt, 1)

    start = cp.cuda.Event()
    end = cp.cuda.Event()

    kernel_start_event.set()
    start.record()

    u = kernel(u0, alpha, kernel_array, dt, num_steps)

    end.record()
    end.synchronize()

    elapsed = cp.cuda.get_elapsed_time(start, end)
    sum_total = cp.sum(u).item()

    return elapsed, sum_total
