from pathlib import Path
import cupy as cp
import numpy as np
import time

MAX_NNZ = 4_000_000_000    # 32GB VRAM

#! ################
# da li je potrebno warm-up kernel uraditi na svakom kernel-u? vjerujem da treba
# tako da treba u monte carlo ubaciti warm up kernel
#! ################

def cupy_monte_carlo(args, kernel_start_event):
    """
    Monte Carlo integration of the Gaussian integral.
    The theoretical value should converge to pi ** (d / 2) as N approaches infinity.

    1. Draw samples
    2. Normalization
    3. Evaluate f(x_i)
    4. MC estimator
    """
    cp.random.seed(42)

    start = cp.cuda.Event()
    end = cp.cuda.Event()

    kernel_start_event.set()
    start.record()

    X = cp.random.uniform(-args.bound, args.bound, size=(args.N, args.d), dtype=cp.float32)
    sq_norms = cp.sum(X**2, axis=1)
    f_vals = cp.exp(-sq_norms)

    volume = (2 * args.bound) ** args.d
    I_estimate = volume * cp.mean(f_vals)

    end.record()
    end.synchronize()

    elapsed = cp.cuda.get_elapsed_time(start, end)

    return elapsed, I_estimate

def cupy_conjugate_gradient(args, kernel_start_event):
    """
    Conjugate Gradient. Uses sin(pi * x). Initial x is 0.
    1. Build sparse Laplacian matrix
    2. 
    """ #TODO: zavrsi comment

    from cupyx.scipy.sparse import diags as sparse_diags
    from cupyx.scipy.sparse.linalg import cg

    # Laplacian matrix
    h = 1.0 / (args.N + 1)
    main_diag = 2.0 * cp.ones(args.N)
    off_diag = -1.0 * cp.ones(args.N - 1)
    A = sparse_diags([off_diag, main_diag, off_diag], offsets=[-1, 0, 1], format="csr")
    A = A / (h ** 2)

    # Calculate known solution
    grid = cp.linspace(0, 1, args.N + 2)[1:-1]
    u = cp.sin(cp.pi * grid)
    b = A @ u

    # Kernel
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

    """
    # Kernel
    start = cp.cuda.Event()
    end = cp.cuda.Event()

    kernel_start_event.set()
    start.record()

    q = kernel()

    end.record()
    end.synchronize()

    elapsed = cp.cuda.get_elapsed_time(start, end)

    return elapsed, q

#* Finished!
def cupy_spectral(args, kernel_start_event):
    """
    Fourier Transformation.
    1.
    2.
    """

    from cupyx.scipy.fft import dctn, idctn

    def kernel(u0, alpha, dx, dy, dz, dt, num_steps):
        N = u0.shape[0]
        
        # Frequency grid for DCT
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

        u_hat = dctn(u0, norm='ortho')
        evolution_operator = cp.exp(alpha * k_squared * dt)
        
        for _ in range(num_steps):
            u_hat = u_hat * evolution_operator
        
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

    # Kernel
    _ = kernel(u0, alpha, dx, dy, dz, dt, num_steps)

    start = cp.cuda.Event()
    end = cp.cuda.Event()

    kernel_start_event.set()
    start.record()

    u_final = kernel(u0, alpha, dx, dy, dt, num_steps)

    end.record()
    end.synchronize()

    elapsed = cp.cuda.get_elapsed_time(start, end)
    sum_total = cp.sum(u_final).item()

    return elapsed, sum_total

#TODO: dodaj optional argument za spmv, formatiraj
def cupy_spmv(args, kernel_start_event):
    """
    Calculates the sparse matrix vector multiplication y = Ax.
    The kernel utilizes power iteration to increase its duration.
    """

    import cupyx.scipy.sparse as csp

    def kernel(A_csr, x, iterations=200):
        for _ in range(iterations):
            x = A_csr @ x
            x = x / cp.linalg.norm(x)
        cp.cuda.Stream.null.synchronize()
        return x

    N = 300_000
    density = 0.01
    nnz = int(N * N * density)
    if nnz > MAX_NNZ:
        raise ValueError(
            f"Requested matrix would contain {nnz:,} nonzeros.\n"
            f"Maximum allowed is {MAX_NNZ:,}."
        )

    start = cp.cuda.Event()
    end = cp.cuda.Event()
    cp.random.seed(42)

    rows = cp.random.randint(0, N, size=nnz, dtype=cp.int32)
    cols = cp.random.randint(0, N, size=nnz, dtype=cp.int32)
    vals = cp.random.rand(nnz, dtype=cp.float32)

    A_gpu = csp.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
    x_gpu = cp.random.rand(N, dtype=cp.float32)
    cp.cuda.Stream.null.synchronize()

    kernel_start_event.set()
    start.record()
    # t0 = time.perf_counter()

    y_gpu = kernel(A_gpu, x_gpu)
    #TODO: transfer to cpu? include in timing?

    # t1 = time.perf_counter()
    end.record()
    end.synchronize()

    elapsed = cp.cuda.get_elapsed_time(start, end)

    return elapsed, 

def cupy_stencil(args, kernel_start_event):
    """

    """

    # Kernel
    start = cp.cuda.Event()
    end = cp.cuda.Event()

    kernel_start_event.set()
    start.record()

    end.record()
    end.synchronize()

    elapsed = cp.cuda.get_elapsed_time(start, end)

    return elapsed, 
