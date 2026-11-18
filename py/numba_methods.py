from numba import cuda, float32
import numpy as np
import math
import StabilityException, ConvergenceException

MAX_NNZ = 4_000_000_000     #! ovo zapravo treba da bude 4G * X bytes jer ovo nije 32GB VRAM
K_E = 8.987551786214e9      # https://en.wikipedia.org/wiki/Coulomb%27s_law

#! podesi bpg, tpb u ostatku koda
BPG = 1
TPB = 256

@cuda.jit
def axpy_kernel(a, x, y, out):
    i = cuda.grid(1)
    if i < out.size:
        out[i] = a * x[i] + y[i]

@cuda.jit
def spmv_csr(rowptr, col, val, x, y):
    r = cuda.grid(1)
    if r < y.size:
        acc = 0.0
        for j in range(rowptr[r], rowptr[r + 1]):
            acc += val[j] * x[col[j]]
        y[r] = acc

@cuda.jit
def dot_kernel(x, y, out):
    """
    The dot product can be defined naively through two kernels, element multiplication and sum reduction.
    Element multiplication would be O(3N) : out[i] = a[i] * b[i]; 0 <= i < N.
    Sum reduction would be O(1N): sum += out[i]; 0 <= i < N.
    Thus the naive kernel would be O(4N) which can be reduced to O(2N) through the following dot kernel.
    #! ovdje objasni razliku izmedju naive i ovog novog

    The dot product kernel took inspiration from the following websites:
    https://doc.sling.si/en/workshops/programming-gpu-cuda/04-programming/03-dotprod/#dot-product-optimized-usage-of-shared-memory-and-reduction
    https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
    """

    sh     = cuda.shared.array(TPB, float32)
    tid    = cuda.threadIdx.x
    idx    = cuda.grid(1)
    stride = cuda.gridsize(1)

    # Element wise multiplication
    acc = 0.0
    while idx < x.size:
        acc += x[idx] * y[idx]
        idx += stride
    sh[tid] = acc

    cuda.syncthreads()

    # Sum reduction (tree-based approach) #* can be optimized further
    s = TPB // 2
    while s > 0:
        if tid < s:
            sh[tid] += sh[tid + s]
        cuda.syncthreads()
        s //= 2
    if tid == 0:
        out[cuda.blockIdx.x] = sh[0]

def axpy(a, x, y):
    d_out = cuda.device_array(y.size)
    bpg = (y.size + TPB - 1) // TPB
    axpy_kernel[bpg, TPB](a, x, y, d_out)

    out = d_out.copy_to_host()
    return out

def spmv(A, x):
    y = cuda.device_array(x.size)
    bpg = (x.size + TPB - 1) // TPB
    spmv_csr[bpg, TPB](A.indptr, A.indices, A.data, x, y)

    out = y.copy_to_host()
    return y

def dot(x, y):
    """ #! both x and y should be flattened """
    d_out = cuda.device_array(x.size, dtype=np.float32)
    bpg = (x.size + TPB - 1) // TPB
    dot_kernel[bpg, TPB](x, y, d_out)

    out = d_out.copy_to_host()
    return out.sum() #* CUDA: for (int b = 0; b < NBLK; ++b) result += h_partial[b];

def numba_monte_carlo(args, kernel_start_event):
    """
    Monte Carlo integration of the Gaussian integral.
    The theoretical value should converge to pi ** (d / 2) as N approaches infinity.

    1. Draw samples
    2. Normalization
    3. Evaluate f(x_i)
    4. MC estimator through the following steps:
        - atomic reduction of the sum of f(x_i),
        - D2H transfer of the sum
        - dividing by N on the host to get the mean
    """
    from numba.cuda.random import (
        create_xoroshiro128p_states,
        xoroshiro128p_uniform_float32
    )

    @cuda.jit
    def kernel(rng_states, bound, N, total):
        tid = cuda.grid(1)
        acc = 0.0
        for _ in range(tid, N, cuda.gridsize(1)):
            sq = 0.0
            for _ in range(d):
                # Bound from [-bound, bound)
                # defined as x * (up - lb) + lb
                x = xoroshiro128p_uniform_float32(rng_states, tid)
                x = x * (2 * bound) - bound
                sq += x * x

            acc += math.exp(-sq)
        cuda.atomic.add(total, 0, acc)

    rng_states = create_xoroshiro128p_states(BPG*TPB, seed=42)
    f_vals_sum = cuda.to_device(np.zeros(1, dtype=np.float32))

    # Warm up
    kernel[BPG, TPB](rng_states, args.bound, args.N, f_vals_sum)
    f_vals_sum = cuda.to_device(np.zeros(1, dtype=np.float32))

    start = cuda.event()
    end = cuda.event()

    kernel_start_event.set()
    start.record()

    kernel[BPG, TPB](rng_states, args.bound, args.N, f_vals_sum)
    volume = (2 * args.bound) ** args.d
    I_estimate = volume * f_vals_sum.copy_to_host() / args.N

    end.record()
    end.synchronize()

    elapsed = cuda.event_elapsed_time(start, end)

    return elapsed, I_estimate

"""
# CUSTOM CG IMPLEMENTATION FOR TESTING
def custom_cg(A, b, rtol=1e-5, atol=0., maxiter=200):
    N = A.shape[0] # int
    x = np.zeros_like(b) # vector
    r = b - (A@x) # vector
    p = np.copy(r)

    r_old = r.T@r #2 reduction
    
    for iteration in range(maxiter):
        if np.linalg.norm(r) <= max(rtol * np.linalg.norm(b), atol):
            return x

        Ap = A@p #1 parallel
        alpha = r_old / (p.T @ Ap) #3 sequential

        x = x + (alpha * p) #4 parallel
        r = r - (alpha * Ap) #5 parallel
        
        r_new = (r.T@r) #6 reduction

        beta = r_new / r_old #7 sequential
        p = r + (beta*p) #8 parallel

        r_old = r_new #(9) pointer switch

    return x
"""

def numba_conjugate_gradient(args, kernel_start_event):
    """
    Implementation which uses Conjugate Gradient to solve -d^2u/dx^2 = sin(pi * x) with
    Dirichlet boundary conditions, u(0) = u(1) = 1, whose exact solution is known to be u = sin(pi*x) / pi**2.

    The cg algorithm is done by the host but the computation is done by the GPU through
    smaller mathematical kernels like axpy, spmv, and dot product.
    
    Initial x is set to 0.

    The algorithm is defined through the following steps:
    1. Build the sparse Laplacian matrix A
    2. Define the right hand side b
    3. Iterate until x converges
    """

    from scipy.sparse import diags as sparse_diags

    print(f"\nProblem size N:             {args.N}")
    print(f"Maximum iteration count:    {args.maxiter}")

    def cg(A, b, rtol=1e-5, atol=0., maxiter=200):
        """ x0 is always set to zero. """

        #! Add error check for SPD ([nxn], xTAx),
        #! check if others require error checking as well
        b_sqared = dot(b.T, b)
        N = A.shape[0]
        x = np.zeros_like(b)

        r = b - (spmv(A, x))
        p = np.copy(r)

        r_old = dot(r.T, r)

        for _ in range(maxiter):
            if r_old <= max(rtol**2 * b_sqared, atol):
                return x

            Ap = spmv_csr(A, p)
            alpha = r_old / dot(p.T, Ap)

            x = axpy(alpha, p, x)
            r = axpy(-alpha, Ap, r)

            r_new = dot(r.T, r)

            beta = r_new / r_old
            p = axpy(beta, p, r)

            r_old = r_new

        raise ConvergenceException("ERROR: Convergence to tolerance not achieved")

    # Laplacian representation with spacing h
    # -d^2u/dx^2 = -u[i-1] + 2u[i] - u[i+1], thus off-diagonals are -1 and diagonals are 2
    h = 1.0 / (args.N + 1)
    main_diag = 2.0 * np.ones(args.N)
    off_diag = -1.0 * np.ones(args.N - 1)
    
    # A = -d^2/dx^2; Au = -d^2u/dx^2
    A = sparse_diags([off_diag, main_diag, off_diag], offsets=[-1, 0, 1], format="csr")
    A /= (h ** 2)

    # Grid skips the 0 boundary values
    # Au = sin(pi * x); b = sin(pi * x); Au = b
    grid = np.linspace(0, 1, args.N + 2)[1:-1]
    b = np.sin(np.pi * grid)

    # Warm up
    _ = cg(A, b, rtol=1e-10, maxiter=1)

    start = cuda.event()
    end = cuda.event()

    kernel_start_event.set()
    start.record()

    x = cg(A, b, rtol=1e-10, maxiter=args.maxiter)
    
    end.record()
    end.synchronize()

    elapsed = cuda.event_elapsed_time(start, end)

    return elapsed, x.sum().item()

def numba_dcs(args, kernel_start_event):
    """
    Implementation of the Direct Coulomb Summation.

    Defines atoms as an array of [x, y, z, q] where x, y, z are the position and q is the charge.
    1. Sample random atom values
    2. Calculate U = k_e * (Ei qi/ri)
    """

    print(f"\nGrid size:              [{args.grid_size}, {args.grid_size}, {args.grid_size}]")
    print(f"Amount of particles:    {args.particle_count}")
    
    @cuda.jit
    def kernel():
        tid = cuda.grid(1)

    rng = np.random.default_rng(42)

    atoms = np.empty((args.particle_count, 4), dtype=np.float32)
    atoms[:, :3] = rng.integers(0, args.grid_size, size=(args.particle_count, 3), endpoint=True)
    atoms[:, 3] = rng.uniform(-5e-6, 5e-6, size=args.particle_count)

    d_atoms = cuda.to_device(atoms)
    d_output = cuda.to_device(np.zeros_like(atoms))

    # Warm up
    kernel[BPG, TPB](d_atoms, d_output) #! pretjerano velik array
    d_output = cuda.to_device(np.zeros_like(atoms))

    start = cuda.event()
    end = cuda.event()

    kernel_start_event.set()
    start.record()

    kernel[BPG, TPB](d_atoms, d_output)

    end.record()
    end.synchronize()

    elapsed = cuda.event_elapsed_time(start, end)
    sum_total = d_output.copy_to_host().sum().item()

    return elapsed, sum_total

def numba_spectral(args, kernel_start_event):
    """
    Solve the 3D heat equation using a spectral method.
    1. Transform the initial temperature field to frequency space
    2. Evolve the frequency modes over time and transform back to the spatial domain
    """
    @cuda.jit
    def kernel(output, u0, alpha, k_squared, dt, num_steps):
        x, y, z = cuda.grid(3)
        """
        Ovdje bi najbolje bilo iskoristiti cupy fft i onda numba kernel za norm.
        A zasto bas ovako kad bi se pitalo, objasnjenje je jednostavno.
        Nece niko FFT praviti rucno kad je to vec odradjena funkcija, postoje vec
        optimalni algoritmi za to tako da ako mi treba FFT koristicu neki library.
        Eh sad, numba nema nikakv library za to, a cesto se desi da cupy i numba
        zajedno rade, tako da je smisleno koristiti cupy u ovom slucaju cisto da
        bi se cuFFT invoke-o jer Numba nema funkcionalnost da zove NVIDIA APIs direktno.
        Tako da se moze zakljuciti iz navedenog da CuPy u ovom slucaju se koristi kao
        zamjena da se invoke-a NVIDI API, cuFFT u ovo slucaju, te ostatak koda
        koristi Numba-u tako da se moze zakljuciti da ovo ne utice na Numba istrazivanje.
        """

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
    num_steps = round(t_final / dt)

    x = np.linspace(0.0, Lx, nx)
    y = np.linspace(0.0, Ly, ny)
    z = np.linspace(0.0, Lz, nz)
    X, Y, Z = np.meshgrid(x, y, z)

    # Initial condition (Gaussian bump)
    u0 = A * np.exp( -((X - x0)**2 + (Y - y0)**2 + (Z - z0)**2) / (2 * sigma_initial**2) )
    d_u0 = cuda.to_device(u0)
    N = u0.shape[0]
            
    # Frequency grid
    kx = np.pi * np.arange(N) / N
    ky = np.pi * np.arange(N) / N
    kz = np.pi * np.arange(N) / N
    KX, KY, KZ = np.meshgrid(kx, ky, kz)
    
    # Eigenvalues for Neumann BC
    k_squared = (
        2 * (np.cos(KX) - 1) / dx**2 +
        2 * (np.cos(KY) - 1) / dy**2 +
        2 * (np.cos(KZ) - 1) / dz**2
    )

    u_final = cuda.device_array((nx, ny, nz), dtype=np.float32)

    # Warm up
    kernel[BPG, TPB](u_final, u0, alpha, k_squared, dt, 1)
    u_final = cuda.to_device(np.zeros())

    start = cuda.event()
    end = cuda.event()

    kernel_start_event.set()
    start.record()

    kernel[BPG, TPB](u0, alpha, k_squared, dt, num_steps)

    end.record()
    end.synchronize()

    elapsed = cuda.event_elapsed_time(start, end)
    sum_total = u_final.copy_to_host().sum().item()

    return elapsed, sum_total

def numba_spmv(args, kernel_start_event):
    """
    Calculates the sparse matrix vector multiplication y = Ax.

    The kernel utilizes power iteration to increase its duration.
    """

    @cuda.jit
    def kernel():
        row = cuda.grid(1)

    # Warm up
    # kernel[]()

    start = cuda.event()
    end = cuda.event()

    kernel_start_event.set()
    start.record()

    # kernel[]()

    end.record()
    end.synchronize()

    elapsed = cuda.event_elapsed_time(start, end)
    # sum_total = .sum().item()

    return elapsed, sum_total

def numba_stencil(args, kernel_start_event):
    """
    Explicit 7-point stencil for the 3D heat equation.
    Assumes equal spacing in all three dimensions (dx=dy=dz). Will not work for unequal spacing.

    Stability, for equal spacing, is defined as r = alpha*dt / spacing^2 for which, in 3D, must be r <= 1/6.

    The stencil is calculated by convolving the laplacian kernel.
    Uses the 'reflect' mode as per defined Neumann boundary condition.
    """
    @cuda.jit
    def kernel(u, u_step, alpha, dx, dy, dz, grid_size, dt):
        i, j, k = cuda.grid(3)

        # 1 <= i, j, k < last_index; where last index is grid_size-1
        if 1 <= i < grid_size-1 and 1 <= j < grid_size-1 and 1 <= k < grid_size-1:
            dudx2 = (u[i+1, j, k] - 2*u[i, j, k] + u[i-1, j, k]) / dx**2
            dudy2 = (u[i, j+1, k] - 2*u[i, j, k] + u[i, j-1, k]) / dy**2
            dudz2 = (u[i, j, k+1] - 2*u[i, j, k] + u[i, j, k-1]) / dz**2

            u_step[i, j] = u[i, j, z] + alpha * dt * (dudx2 + dudy2 + dudz2)

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
    num_steps = round(t_final / dt)

    # Stability
    r = alpha * dt / dx**3
    if (r > 1/6):
        raise StabilityException(f"Unstable simulation; must be below 0.167. Current r: {r}")

    x = np.linspace(0.0, Lx, nx)
    y = np.linspace(0.0, Ly, ny)
    z = np.linspace(0.0, Lz, nz)
    X, Y, Z = np.meshgrid(x, y, z)

    # Initial condition (Gaussian bump)
    u0 = A * np.exp( -((X - x0)**2 + (Y - y0)**2 + (Z - z0)**2) / (2 * sigma_initial**2) )

    u_gpu = cuda.to_device(u0.copy())
    u_step_gpu = cuda.to_device(u0.copy())

    # Kernel
    tpb = (8, 8, 8)
    bpg_x = (args.grid_size + (tpb[0] - 1)) // tpb[0]
    bpg_y = (args.grid_size + (tpb[1] - 1)) // tpb[1]
    bpg_z = (args.grid_size + (tpb[2] - 1)) // tpb[2]
    bpg = (bpg_x, bpg_y, bpg_z)

    # Warm up
    kernel[bpg, tpb](u_gpu, u_step_gpu, alpha, dx, dy, dz, 3, dt)
    u_gpu = cuda.to_device(u0.copy())
    u_step_gpu = cuda.to_device(u0.copy())

    start = cuda.event()
    end = cuda.event()

    kernel_start_event.set()
    start.record()

    for step in range(num_steps):
        kernel[bpg, tpb](u_gpu, u_step_gpu, alpha, dx, dy, dz, args.grid_size, dt)
        u_gpu, u_step_gpu = u_step_gpu, u_gpu

    end.record()
    end.synchronize()

    elapsed = cuda.event_elapsed_time(start, end)
    sum_total = cp_sum(u_gpu.copy_to_host()).item()

    return elapsed, sum_total
