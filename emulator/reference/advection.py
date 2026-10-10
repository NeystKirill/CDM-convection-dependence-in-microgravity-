import numpy as np


def upwind_term(grid, phi, u, phi_rb):
    grad = np.empty(grid.n)
    dr0 = grid.r_c[0] - grid.r_face[0]
    grad[0] = (phi[0] - phi_rb) / dr0
    dr = grid.r_c[1:] - grid.r_c[:-1]
    grad[1:] = (phi[1:] - phi[:-1]) / dr
    return -u * grad
