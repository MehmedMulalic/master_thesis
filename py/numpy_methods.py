import numpy as np
from scipy.sparse.linalg import cg
from scipy.sparse import diags

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
