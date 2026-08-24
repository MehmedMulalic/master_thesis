from numba import cuda
from pathlib import Path
import numpy as np
import time

def numba_monte_carlo(args):
    """
    Monte Carlo integration of the Gaussian integral.
    The theoretical value should converge to pi^(d/2) as N and d approach infinity.
    """
    from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32

    @cuda.jit
    def kernel(rng_states, output):
        i = cuda.grid(1)

        if i < output.size:
            x = xoroshiro128p_uniform_float32(rng_states, i)
            output[i] = x

    #! args nek ima optional bound default 6, mandatory N & d
    X = np.random.uniform(-args.bound, args.bound, size=(args.N, args.d), dtype=np.float32)
    f_vals = np.exp(-np.sum(X**2, axis=1))
    volume = (2 * args.bound) ** args.d

    I_estimate = volume * np.mean(f_vals)
    
def numba_conjugate_gradient(args):
    """

    """
    @cuda.jit
    def kernel():
        pass

    pass

def numba_dcs(args):
    """

    """
    @cuda.jit
    def kernel():
        pass

    pass

def numba_spectral(args):
    """

    """
    @cuda.jit
    def kernel():
        pass

    pass

def numba_spmv(args):
    """
    Calculates the sparse matrix vector multiplication y = Ax.
    """

    @cuda.jit
    def kernel():
        row = cuda.grid(1)

    threadsperblock = 256
    blockspergrid = (threadsperblock - 1) // threadsperblock

    kernel[blockspergrid, threadsperblock]()

def numba_stencil(args):
    """

    """
    @cuda.jit
    def kernel(u, u_step, alpha, dx, dy, grid_size, dt):
        i, j, k = cuda.grid(3)

        if 1 <= i < grid_size-1 and 1 <= j < grid_size-1 and 1 <= k < grid_size-1:
            dudx2 = (u[i+1, j] - 2*u[i, j] + u[i-1, j]) / dx**2
            dudy2 = (u[i, j+1] - 2*u[i, j] + u[i, j-1]) / dy**2

            u_step[i, j] = u[i, j] + alpha * dt * (dudx2 + dudy2)

    A = 100.0                   # Peak temperature (°C)
    sigma_initial = 0.2         # Initial width (m)
    x0 = 0.5                    # Centre
    alpha = 1.6563e-4           # Thermal diffusivity of pure silver (m^2/s)
    L = 1.0                     # Domain limit (m)

    spacing = L / (args.grid_size - 1)
    dt = 0.0003  #! zasto ovaj broj
    t_final = 60
    num_steps = int(t_final / dt)

    # Stability
    r = alpha * dt / spacing**2
    if (r > 0.25):
        raise Exception("Unstable simulation")

    x = np.linspace(0.0, L, args.grid_size)
    y = np.linspace(0.0, L, args.grid_size)
    z = np.linspace(0.0, L, args.grid_size)
    X, Y, Z = np.meshgrid(x, y, z)

    # Initial condition (Gaussian bump)
    u0 = A * np.exp( -((X - x0)**2 + (Y - x0)**2 + (Z - x0)**2) / (2 * sigma_initial**2) )

    threadsperblock = (8, 8, 8)
    blockspergrid_x = (args.grid_size + (threadsperblock[0] - 1)) // threadsperblock[0]
    blockspergrid_y = (args.grid_size + (threadsperblock[1] - 1)) // threadsperblock[1]
    blockspergrid_z = (args.grid_size + (threadsperblock[1] - 1)) // threadsperblock[2]
    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

    u_gpu = cuda.to_device(u0.copy())
    u_step_gpu = cuda.to_device(u0.copy())

    #! treba li discarded warm up run
    t0 = time.perf_counter()
#! dx je sada spacing promijeni kernel da odgovara
    for step in range(num_steps):
        kernel[blockspergrid, threadsperblock](u_gpu, u_step_gpu, alpha, dx, dy, nx, ny, dt)
        u_gpu, u_step_gpu = u_step_gpu, u_gpu

    u_cpu = u_gpu.copy_to_host()

    t1 = time.perf_counter()
    print(f"Time taken: {t1-t0:.5}s")

    filename_dir = Path(f"results")
    filename_dir.mkdir(exist_ok=True)
    filename = filename_dir / f"numba_stencil_explicit_{args.grid_size}.csv"
    np.savetxt(filename, u_cpu, delimiter=",")
