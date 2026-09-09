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
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    cp.random.seed(42)

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
    start = cp.cuda.Event()
    end = cp.cuda.Event()

    kernel_start_event.set()
    start.record()

    # kernel

    end.record()
    end.synchronize()

    elapsed = cp.cuda.get_elapsed_time(start, end)

    return elapsed, 

def cupy_spectral(args, kernel_start_event):
    """

    """
    from cupyx.scipy.fft import dctn, idctn

    def kernel(u0, alpha, dx, dt, num_steps):
        N = u0.shape[0]
        
        # Frequency grid for DCT
        k = cp.pi * cp.arange(N) / N
        KX, KY, KZ = cp.meshgrid(k, k, k)
        
        # Eigenvalues for Neumann BC
        k_squared = (2 / dx**2) * (cp.cos(KX) + cp.cos(KY) + cp.cos(KZ) - 2)
        
        u_hat = dctn(u0, norm='ortho')
        evolution_operator = cp.exp(alpha * k_squared * dt)
        
        for step in range(num_steps):
            u_hat = u_hat * evolution_operator
        
        u_final = idctn(u_hat, norm='ortho').real    
        return u_final


    A = 100.0                   # Peak temperature (°C)
    sigma_initial = 0.2         # Initial width (m)
    x0 = 0.5                    # Centre
    alpha = 1.6563e-4           # Thermal diffusivity of pure silver (m^2/s)
    L = 1.0                     # Domain limit (m)

    spacing = L / (args.grid_size - 1)
    dt = 0.0003  #! zasto ovaj broj
    t_final = 60
    num_steps = int(t_final / dt)

    x = cp.linspace(0.0, L, args.grid_size)
    y = cp.linspace(0.0, L, args.grid_size)
    z = cp.linspace(0.0, L, args.grid_size)
    X, Y, Z = cp.meshgrid(x, y, z)

    # Initial condition (Gaussian bump)
    u0 = A * cp.exp( -((X - x0)**2 + (Y - x0)**2 + (Z - x0)**2) / (2 * sigma_initial**2) )

    # Discarded warm-up run
    kernel()

    t0 = time.perf_counter()

    u_final = kernel()
    u_cpu = cp.asnumpy(u_final)

    t1 = time.perf_counter()
    print(f"Time taken: {t1-t0:.5}s")

    filename_dir = Path(f"results")
    filename_dir.mkdir(exist_ok=True)
    filename = filename_dir / f"cupy_spectral_{args.grid_size}.csv"
    np.savetxt(filename, u_cpu, delimiter=",")

#TODO: dodaj optional argument za spmv
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
    start = cp.cuda.Event()
    end = cp.cuda.Event()

    kernel_start_event.set()
    start.record()

    # kernel

    end.record()
    end.synchronize()

    elapsed = cp.cuda.get_elapsed_time(start, end)

    return elapsed, 
