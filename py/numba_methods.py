from numba import cuda
from pathlib import Path
from cupy import sum as cp_sum
import numpy as np
import math
import StabilityException

MAX_NNZ = 4_000_000_000     # 32GB VRAM
K_E = 8.987551786214e9      # https://en.wikipedia.org/wiki/Coulomb%27s_law

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
    def kernel(rng_states, total):
        tid = cuda.grid(1)
        acc = 0.0
        for _ in range(tid, args.N, cuda.gridsize(1)):
            sq = 0.0
            for _ in range(d):
                # Bound from [-bound, bound)
                # defined as x * (up - lb) + lb
                x = xoroshiro128p_uniform_float32(rng_states, tid)
                x = x * (2 * args.bound) - args.bound
                sq += x * x

            acc += math.exp(-sq)
        cuda.atomic.add(total, 0, acc)

    rng_states = create_xoroshiro128p_states(n_threads, seed=42)
    f_vals_sum = cuda.to_device(np.zeros(1, dtype=np.float32))

    # Kernel
    bpg = 1
    tpb = 32
    n_threads = bpg * tpb

    kernel[bpg, tpb](rng_states, f_vals_sum)
    f_vals_sum = cuda.to_device(np.zeros(1, dtype=np.float32))

    start = cuda.event()
    end = cuda.event()

    kernel_start_event.set()
    start.record()

    kernel[bpg, tpb](rng_states, f_vals_sum)
    volume = (2 * args.bound) ** args.d
    I_estimate = volume * f_vals_sum.copy_to_host() / args.N

    end.record()
    end.synchronize()

    elapsed = cuda.event_elapsed_time(start, end)

    return elapsed, I_estimate
    
def numba_conjugate_gradient(args):
    """
    Using Conjugate Gradient to solve -d^2u/dx^2 = sin(pi * x) with zero boundary conditions
    whose exact solution is known to be u = sin(pi*x)/pi^2.
    
    Initial x is set to 0.
    1. Build the sparse Laplacian matrix A
    2. Calculate the solution b
    3. Iterate until convergence
    """
    @cuda.jit
    def kernel():
        pass

    pass

def numba_dcs(args):
    """
    Implementation of the Direct Coulomb Summation.

    Defines atoms as an array of [x, y, z, q] where x, y, z are the position and q is the charge.
    1. Sample random atom values
    2. Calculate U = k_e * (Ei qi/ri)
    """
    @cuda.jit
    def kernel():
        pass

    pass

def numba_spectral(args):
    """
    Solve the 3D heat equation using a spectral method.
    1. Transform the initial temperature field to frequency space
    2. Evolve the frequency modes over time and transform back to the spatial domain
    """
    @cuda.jit
    def kernel():
        pass

    pass

def numba_spmv(args):
    """
    Calculates the sparse matrix vector multiplication y = Ax.

    The kernel utilizes power iteration to increase its duration.
    """

    @cuda.jit
    def kernel():
        row = cuda.grid(1)

    threadsperblock = 256
    blockspergrid = (threadsperblock - 1) // threadsperblock

    kernel[blockspergrid, threadsperblock]()

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
