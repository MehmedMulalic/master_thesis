from pathlib import Path
import cupy as cp
import numpy as np
import time

def cupy_monte_carlo(args):
    """
    Monte Carlo integration of the Gaussian integral.
    The theoretical value should converge to pi^(d/2) as N and d approach infinity.
    """
    def kernel(d, N, bound=6.0):
        X = cp.random.uniform(-bound, bound, size=(N, d), dtype=cp.float32)
        f_vals = cp.exp(-cp.sum(X**2, axis=1))
        volume = (2 * bound) ** d

        I_estimate = volume * cp.mean(f_vals)
        return I_estimate

    

def cupy_conjugate_gradient(args):
    """

    """
    pass

def cupy_dcs(args):
    """

    """
    pass

def cupy_spectral(args):
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

def cupy_spmv(args):
    """
    Calculates the sparse matrix vector multiplication y = Ax.
    """
    import cupyx.scipy.sparse as csp
    from scipy.sparse import random

    def kernel(A_csr, x):
        y = A_csr @ x
        return y

    rng = np.random.default_rng(seed=42)
    N = 100_000
    density = 0.001

    A_host = random(N, N, density=density, format='csr', rng=rng, dtype=cp.float32)
    A_gpu = csp.csr_matrix(A_host)
    x_gpu = cp.random.rand(N, dtype=cp.float32)

    y_gpu = kernel(A_gpu, x_gpu)
    y_cpu = cp.asnumpy(y_gpu)
    
    filename_dir = Path(f"results")
    filename_dir.mkdir(exist_ok=True)
    filename = filename_dir / f"cupy_spmv_XYZ.csv" #!change
    np.savetxt(filename, y_cpu, delimiter=",")

    ### DEBUG ###
    # A = cp.random.random((4096, 4096), dtype=cp.float32)
    # B = cp.random.random((4096, 4096), dtype=cp.float32)

    # for _ in range(10):
    #     C = A @ B
    #     cp.cuda.Stream.null.synchronize()
    ### END ###

    return "done"

def cupy_stencil(args):
    """

    """
    pass
