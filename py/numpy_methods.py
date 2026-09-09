import numpy as np
from argparse import ArgumentParser
from scipy.sparse.linalg import cg
from scipy.sparse import diags
from scipy.fft import dctn, idctn

def numpy_theoretical_mc(args):
    true_val = np.pi ** (args.d / 2)

    return true_val

def numpy_cg(args):
    # Laplacian matrix
    h = 1.0 / (args.N + 1)
    main_diag = 2.0 * np.ones(args.N)
    off_diag = -1.0 * np.ones(args.N - 1)
    A = diags([off_diag, main_diag, off_diag], offsets=[-1, 0, 1], format="csr")
    A = A / (h ** 2)

    # Calculate known solution
    grid = np.linspace(0, 1, args.n + 1)[1:-1]
    u = np.sin(np.pi * grid)
    b = A @ u

    x = cg(A, b, rtol=1e-10)

    return x

def numpy_dcs(args):
    pass

def numpy_spectral(args):
    A = 100.0               # Peak temperature (°C)
    sigma_initial = 0.2     # Initial width (m)
    x0, y0 = 0.5, 0.5       # Centre
    alpha = 1.6563e-4       # Thermal diffusivity of pure silver (99.9%) (m^2/s)
    Lx, Ly = 1.0, 1.0       # Domain limit (metres)

    # Grid points
    nx = args.grid_size
    ny = args.grid_size

    dx = Lx / (nx - 1)      # x spacing
    dy = Ly / (ny - 1)      # y spacing
    dt = 0.0003             # Time step (s)
    t_final = 60            # Final time step (s)
    num_steps = int(t_final / dt)

    x = np.linspace(0.0, Lx, nx)
    y = np.linspace(0.0, Ly, ny)
    X, Y = np.meshgrid(x, y)

    # Initial condition (Gaussian bump)
    u0 = A * np.exp( -((X - x0)**2 + (Y - y0)**2) / (2 * sigma_initial**2) )

    N = u0.shape[0]
    
    # Frequency grid for DCT
    kx = np.pi * np.arange(N) / N
    ky = np.pi * np.arange(N) / N
    KX, KY = np.meshgrid(kx, ky)
    
    # Eigenvalues for Neumann BC
    k_squared = (2 / dx*dy) * (np.cos(KX) + np.cos(KY) - 2)
    
    u_hat = dctn(u0, norm='ortho')
    evolution_operator = np.exp(alpha * k_squared * dt)
    
    for _ in range(num_steps):
        u_hat = u_hat * evolution_operator

    u_final = idctn(u_hat, norm='ortho').real
    return np.sum(u_final)

def numpy_spmv(args):
    pass

def numpy_stencil(args):
    pass
