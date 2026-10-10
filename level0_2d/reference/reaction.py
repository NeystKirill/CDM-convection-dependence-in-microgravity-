import numpy as np

from .grid import T as IDX_T, F as IDX_F, O as IDX_O

N_ITER = 30


def react_step(grid, params, dt_half):
    j, i = grid.interior
    q = grid.q

    T0 = q[IDX_T, j, i]
    Yf0 = q[IDX_F, j, i]
    Yo0 = q[IDX_O, j, i]
    B, Ta, Qcp, s = params.B, params.Ta, params.Qcp, params.s

    xi_max = np.maximum(np.minimum(Yf0, Yo0 / s), 0.0)
    xi = np.zeros_like(T0)

    for _ in range(N_ITER):
        Tc = T0 + Qcp * xi
        Yfc = Yf0 - xi
        Yoc = Yo0 - s * xi
        expo = np.exp(-Ta / Tc)
        omega = B * Yfc * Yoc * expo

        residual = xi - dt_half * omega
        domega_dxi = B * expo * (
            -Yoc - s * Yfc + Yfc * Yoc * Qcp * Ta / Tc**2
        )
        jac = 1.0 - dt_half * domega_dxi
        jac = np.where(np.abs(jac) < 1e-30, 1e-30, jac)

        xi = np.clip(xi - residual / jac, 0.0, xi_max)

    q[IDX_T, j, i] = T0 + Qcp * xi
    q[IDX_F, j, i] = Yf0 - xi
    q[IDX_O, j, i] = Yo0 - s * xi
