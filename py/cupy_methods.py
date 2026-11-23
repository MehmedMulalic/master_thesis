import cupy as cp
import numpy as np
import StabilityException, ConvergenceException

TYPE_ALIAS = cp.float32

INTEGER_TYPE = {
    cp.float16: cp.int16,
    cp.float32: cp.int32,
    cp.float64: cp.int64
}[TYPE_ALIAS]

SIZEOF_TYPE = {
    cp.float16: 2,
    cp.float32: 4,
    cp.float64: 8
}[TYPE_ALIAS]

K_E = 8.987551786214e9      # https://en.wikipedia.org/wiki/Coulomb%27s_law
ALPHA = 1.6563e-4           # Thermal diffusivity of pure silver (99.9%) (m^2/s)
DT = 0.0003                 # Time step (s)

def check_memory_usage_(array_size):
    free, _ = cp.cuda.Device().mem_info
    print(f"Free VRAM: {free/1024**2} MB")
    
    safety_margin = 1.2
    required_bytes = SIZEOF_TYPE * array_size * safety_margin

    if required_bytes > free:
        raise MemoryError(
            f"ERROR: Requested parameter\n"
            f"REQUIRES: {required_bytes/1024**2} MB. Currently\n"
            f"AVAILABLE: {free/1024**2} MB."
        )

def init_heat_equation_(args, stability=True):
    A = 100.0                       # Peak temperature (°C)
    sigma_initial = 0.2             # Initial width (m)

    x0 = y0 = z0 = 0.5              # Center
    Lx = Ly = Lz = 1.0              # Domain limit (metres)
    nx = ny = nz = args.grid_size

    dx = Lx / (nx - 1)              # x spacing (we assume dx=dy=dz in this case)
    t_final = 60                    # Final time step (s)
    num_steps = round(t_final / DT)

    # Stability
    r = ALPHA * DT / dx**2          # Holds only if equal spacing dx=dy=dz
    if (r > 1/6) and stability:     # Coefficient for 3D space
        raise StabilityException(f"Unstable simulation; must be below 0.167. Current r: {r}")

    x = cp.linspace(0.0, Lx, nx)
    y = cp.linspace(0.0, Ly, ny)
    z = cp.linspace(0.0, Lz, nz)
    X, Y, Z = cp.meshgrid(x, y, z)

    # Initial condition (Gaussian bump)
    u0 = A * cp.exp( -((X - x0)**2 + (Y - y0)**2 + (Z - z0)**2) / (2 * sigma_initial**2) )

    return u0, dx, num_steps

def init_laplacian_array_(dx):
    kernel_array = cp.zeros((3, 3, 3), dtype=TYPE_ALIAS)

    # Above, below (z-axis)
    kernel_array[0, 1, 1] = 1.0
    kernel_array[2, 1, 1] = 1.0

    # Up, down (y-axis)
    kernel_array[1, 0, 1] = 1.0
    kernel_array[1, 2, 1] = 1.0

    # Left, right (x-axis)
    kernel_array[1, 1, 0] = 1.0
    kernel_array[1, 1, 2] = 1.0

    # Center
    kernel_array[1, 1, 1] = -6.0
    kernel_array /= dx**2

    return kernel_array

def cupy_monte_carlo(args, kernel_start_event):
    """
    Monte Carlo integration of the Gaussian integral.
    The theoretical value should converge to pi ** (d / 2) as N approaches infinity.

    1. Draw samples
    2. Normalization
    3. Evaluate f(x_i)
    4. MC estimator
    """

    print()
    print(f"Problem size N:     {args.N}")
    print(f"Dimension d:        {args.d}")

    check_memory_usage_(args.N*args.d)

    def kernel(seed=None):
        if seed is None:
            cp.random.seed()
        else:
            cp.random.seed(seed)

        X = cp.random.uniform(-args.bound, args.bound, size=(args.N, args.d), dtype=TYPE_ALIAS)
        cp.square(X, out=X)

        sq_norms = cp.sum(X, axis=1)
        f_vals = cp.exp(-sq_norms)

        volume = (2 * args.bound) ** args.d
        I_estimate = volume * cp.mean(f_vals)

        return I_estimate

    _ = kernel()
    cp.cuda.Stream.null.synchronize()

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
    Implementation which uses Conjugate Gradient to solve -d^2u/dx^2 = sin(pi * x) with
    Dirichlet boundary conditions, u(0) = u(1) = 1, whose exact solution is known to be u = sin(pi*x) / pi**2.
    
    Initial x is set to 0.
    1. Build the sparse Laplacian matrix A
    2. Calculate the solution b
    3. Iterate until convergence
    """

    from cupyx.scipy.sparse import diags as sparse_diags
    from cupyx.scipy.sparse.linalg import cg

    print()
    print(f"Problem size N:             {args.N}")
    print(f"Maximum iteration count:    {args.maxiter}")

    check_memory_usage_(args.N*6)

    # Laplacian representation with spacing h
    # -d^2u/dx^2 = -u[i-1] + 2u[i] - u[i+1], thus off-diagonals are -1 and diagonals are 2
    h = 1.0 / (args.N + 1)
    main_diag = 2.0 * cp.ones(args.N)
    off_diag = -1.0 * cp.ones(args.N - 1)
    
    # A = -d^2/dx^2; Au = -d^2u/dx^2
    A = sparse_diags([off_diag, main_diag, off_diag], offsets=[-1, 0, 1], format="csr")
    A /= (h ** 2)

    # Grid skips the 0 boundary values
    # Au = sin(pi * x); b = sin(pi * x); Au = b
    grid = cp.linspace(0, 1, args.N + 2)[1:-1]
    b = cp.sin(cp.pi * grid)

    # Kernel
    cp.cuda.Stream.null.synchronize()
    _ = cg(A, b, rtol=1e-10, maxiter=args.maxiter)
    cp.cuda.Stream.null.synchronize()

    start = cp.cuda.Event()
    end = cp.cuda.Event()

    kernel_start_event.set()
    start.record()

    u, info = cg(A, b, rtol=1e-10, maxiter=args.maxiter)
    
    end.record()
    end.synchronize()

    if info != 0:
        raise ConvergenceException("ERROR: Convergence to tolerance not achieved.")

    elapsed = cp.cuda.get_elapsed_time(start, end)

    return elapsed, u

def cupy_dcs(args, kernel_start_event):
    """
    Implementation of the Direct Coulomb Summation.
    Due to a large 3D grid, batch processing is required with a default batch size of 65,536.
    Atoms are defined as an array [x, y, z, q] where x, y, z represent the position and q is the charge.

    The implementation uses NumPy's RNG Generator for deterministic values with a seed of 42.
    To provide equal data across frameworks, the host generates the values with NumPy which are
    then communicated to the device.

    The algorithm is defined with the following steps:
    1. Sample random atom values
    2. Calculate U = k_e * (∑_i q_i / r_i)
    """

    print()
    print(f"Grid size:              [{args.grid_size}, {args.grid_size}, {args.grid_size}]")
    print(f"Number of particles:    {args.particle_count}")

    # check_memory_usage_(, TYPE_ALIAS)

    def kernel(atoms, chunk=1 << 16, k_e=K_E):
        nx = ny = nz = args.grid_size
        total = nx * ny * nz
        
        result = cp.empty(total, dtype=TYPE_ALIAS)
        pos, q = atoms[:, :3], atoms[:, 3]

        for start in range(0, total, chunk):
            idx = cp.arange(start, min(start + chunk, total), dtype=INTEGER_TYPE)
            gx = (idx % nx).astype(TYPE_ALIAS)
            gy = ((idx // nx) % ny).astype(TYPE_ALIAS)
            gz = (idx // (nx * ny)).astype(TYPE_ALIAS)

            # Broadcasting: shape (chunk_size, 1) - shape (1, particle_count)
            # shape (chunk_size, particle_count)
            dx = gx[:, None] - pos[None, :, 0]
            dy = gy[:, None] - pos[None, :, 1]
            dz = gz[:, None] - pos[None, :, 2]
            r = cp.sqrt(dx*dx + dy*dy + dz*dz)

            contrib = cp.true_divide(q[None, :], r, out=cp.zeros_like(r, dtype=TYPE_ALIAS), where=(r > 0))
            result[start:start + len(idx)] = k_e * contrib.sum(axis=1)

        return result.reshape(nz, ny, nx)

    rng = np.random.default_rng(42)

    #* this might cause host errors (not enough RAM)
    atoms = np.empty((args.particle_count, 4), dtype=np.float32)
    atoms[:, :3] = rng.integers(0, args.grid_size, size=(args.particle_count, 3), endpoint=True)
    atoms[:, 3] = rng.uniform(-5e-6, 5e-6, size=args.particle_count)
    
    d_atoms = cp.asarray(atoms)

    # Kernel
    cp.cuda.Stream.null.synchronize()
    _ = kernel(d_atoms)
    cp.cuda.Stream.null.synchronize()
    
    start = cp.cuda.Event()
    end = cp.cuda.Event()

    kernel_start_event.set()
    start.record()

    q = kernel(d_atoms)

    end.record()
    end.synchronize()

    elapsed = cp.cuda.get_elapsed_time(start, end)
    sum_total = cp.sum(q).item()

    return elapsed, sum_total

def cupy_spectral(args, kernel_start_event):
    """
    Solve the 3D heat equation using spectral evolution.
    Assumes equal spacing in all three dimensions (dx=dy=dz). Will not work for unequal spacing.

    Stability is unconditionally stable for this numerical method.

    1. Transform the initial temperature field to frequency space
    2. Evolve the frequency modes over time
    3. Transform back to the spatial domain
    """

    from cupyx.scipy.fft import dctn, idctn

    print(f"\nGrid size:              [{args.grid_size}, {args.grid_size}, {args.grid_size}]")

    array_size = (args.N*3) + (args.N**2 * 4)
    check_memory_usage_(array_size)

    def kernel(u0, k_squared, num_steps, alpha, dt):
        u_hat = dctn(u0)
        evolution_operator = cp.exp(alpha * k_squared * dt)
        
        for _ in range(num_steps):
            u_hat = cp.multiply(u_hat, evolution_operator, out=u_hat)
        
        u_final = idctn(u_hat).real
        return u_final

    u0, dx, num_steps = init_heat_equation_(args, stability=False)
    dz = dy = dx
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
    cp.cuda.Stream.null.synchronize()
    _ = kernel(u0, k_squared, 1, ALPHA, DT)
    cp.cuda.Stream.null.synchronize()

    start = cp.cuda.Event()
    end = cp.cuda.Event()

    kernel_start_event.set()
    start.record()

    u_final = kernel(u0, k_squared, num_steps, ALPHA, DT)

    end.record()
    end.synchronize()

    elapsed = cp.cuda.get_elapsed_time(start, end)
    sum_total = cp.sum(u_final).item()

    return elapsed, sum_total

def cupy_spmv(args, kernel_start_event):
    """
    Calculates the heat equation through sparse matrix vector multiplication y = ax.
    Assumes equal spacing in all three dimensions (dx=dy=dz). Will not work for unequal spacing.

    The equation is defined as u_{n+1} = M * u_{n} where M is the sparse matrix defined as
    M = (I + alpha * delta t * laplacian) * u_n, where u_n is the flattened temperature field vector.

    Stability, for equal spacing, is defined as r = alpha*dt / spacing^2 for which, in 3D, must be r <= 1/6.
    """

    from cupyx.scipy.sparse import diags, identity, kron

    print()
    print(f"Grid size:               [{args.grid_size}, {args.grid_size}, {args.grid_size}]")
    print(f"Number of iterations:    {args.iterations}")

    check_memory_usage_(args.grid_size**6 + args.grid_size**3)

    def init_sparse_laplacian_array_(n, dx):
        lower  = cp.ones(n-1, dtype=TYPE_ALIAS)
        middle = cp.full(n, -2, dtype=TYPE_ALIAS)
        upper  = cp.ones(n-1, dtype=TYPE_ALIAS)

        # Neumann BC
        lower[-1] = 2
        upper[0] = 2
        
        D = diags([lower, middle, upper], [-1, 0, 1], shape=(n, n), format="csr", dtype=TYPE_ALIAS)
        I = identity(n, format="csr", dtype=TYPE_ALIAS)

        L = (
            kron(kron(D, I), I) +
            kron(kron(I, D), I) +
            kron(kron(I, I), D)
        )

        return (L / dx**2).tocsr()

    def kernel(u, M, num_steps):
        for _ in range(num_steps):
            u = M @ u

        return u

    u0, dx, num_steps = init_heat_equation_(args)
    L = init_sparse_laplacian_array_(args.grid_size, dx)
    N = u0.size
    
    u0 = cp.ravel(u0)
    I = identity(N, format="csr", dtype=TYPE_ALIAS)
    M = I + (ALPHA * DT * L)
    
    # Kernel
    cp.cuda.Stream.null.synchronize()
    _ = kernel(u0, M, 1)
    cp.cuda.Stream.null.synchronize()

    start = cp.cuda.Event()
    end = cp.cuda.Event()

    kernel_start_event.set()
    start.record()

    y = kernel(u0, M, num_steps)

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

    print(f"\nGrid size:              [{args.grid_size}, {args.grid_size}, {args.grid_size}]")

    def kernel(u, kernel_array, num_steps, alpha, dt):
        for _ in range(num_steps):
            laplacian = convolve(u, kernel_array)
            u += alpha * dt * laplacian

        return u

    u0, dx, num_steps = init_heat_equation_(args)
    kernel_array = init_laplacian_array_(dx)

    # Kernel
    cp.cuda.Stream.null.synchronize()
    _ = kernel(u0, kernel_array, 1, ALPHA, DT)
    cp.cuda.Stream.null.synchronize()

    start = cp.cuda.Event()
    end = cp.cuda.Event()

    kernel_start_event.set()
    start.record()

    u = kernel(u0, kernel_array, num_steps, ALPHA, DT)

    end.record()
    end.synchronize()

    elapsed = cp.cuda.get_elapsed_time(start, end)
    sum_total = cp.sum(u).item()

    return elapsed, sum_total


if __name__ == "__main__":
    raise SystemExit(
        "ERROR: this module is not intended to be executed directly.\n"
        "Please run solver.py instead."
    )
