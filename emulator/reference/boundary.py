import numpy as np

from .grid import T, F, O


def apply_bc(grid, params):
    q = grid.q
    ng = grid.ng

    q[T, :, :ng] = params.T_inf
    q[F, :, :ng] = 0.0
    q[O, :, :ng] = params.Yo_inf

    q[:, :, -ng:] = q[:, :, -ng - 1 : -ng]

    q[:, -ng:, :] = q[:, -ng - 1 : -ng, :]

    on_fuel = ((grid.x >= params.fuel_x0) & (grid.x <= params.fuel_x1))[None, None, :]
    bottom_neumann = q[:, ng : ng + 1, :]

    bottom_dirichlet = np.empty((3, 1, grid.x.size))
    bottom_dirichlet[T] = params.Ts
    bottom_dirichlet[F] = params.Yf_s
    bottom_dirichlet[O] = 0.0

    q[:, :ng, :] = np.where(on_fuel, bottom_dirichlet, bottom_neumann)
