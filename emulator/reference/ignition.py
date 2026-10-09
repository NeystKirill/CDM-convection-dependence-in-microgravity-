import numpy as np

from .grid import T as IDX_T


def apply_ignition(grid, params, t):
    if t >= params.ign_t:
        return
    j, i = grid.interior
    x = grid.x[i]
    y = grid.y[j]
    in_x = (x >= params.ign_x0) & (x <= params.ign_x1)
    in_y = y <= params.ign_y1
    mask = in_y[:, None] & in_x[None, :]

    Tfield = grid.q[IDX_T, j, i]
    grid.q[IDX_T, j, i] = np.where(mask, np.maximum(Tfield, params.ign_T), Tfield)
