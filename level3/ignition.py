import numpy as np


def apply_ignition(T, grid, params, t):
    if t >= params.tign:
        return T
    radial = (grid.X_c - params.wick_x0) ** 2 + (grid.Y_c - params.wick_y0) ** 2
    kernel = (radial <= params.ignition_kernel_r ** 2) & (grid.Z_c <= params.ignition_kernel_z)
    return np.where(kernel, np.maximum(T, params.Tign), T)
