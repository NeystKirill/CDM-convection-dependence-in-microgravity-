import numpy as np


def apply_ignition(T, grid, params, t):
    if t >= params.tign:
        return T
    kernel = (grid.R_c <= params.ignition_kernel_r) & (grid.Z_c <= params.ignition_kernel_z)
    return np.where(kernel, np.maximum(T, params.Tign), T)
