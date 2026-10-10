import numpy as np


def diffusion_rhs(grid, params):
    q = grid.q
    ng = grid.ng
    nx, ny = grid.nx, grid.ny
    dx, dy = grid.dx, grid.dy

    jc, ic = slice(ng, ng + ny), slice(ng, ng + nx)
    center = q[:, jc, ic]
    left = q[:, jc, ng - 1 : ng - 1 + nx]
    right = q[:, jc, ng + 1 : ng + 1 + nx]
    down = q[:, ng - 1 : ng - 1 + ny, ic]
    up = q[:, ng + 1 : ng + 1 + ny, ic]

    lap = (left - 2.0 * center + right) / dx**2 + (down - 2.0 * center + up) / dy**2

    coef = np.array([params.alpha, params.D, params.D]).reshape(3, 1, 1)
    return coef * lap
