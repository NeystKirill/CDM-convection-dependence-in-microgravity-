import numpy as np


def apply_ignition(T, grid, params, t):
    if t >= params.tign:
        return T
    kernel = grid.r_c <= grid.r_face[0] + params.ignition_kernel_Rb * params.Rb
    return np.where(kernel, np.maximum(T, params.Tign), T)
