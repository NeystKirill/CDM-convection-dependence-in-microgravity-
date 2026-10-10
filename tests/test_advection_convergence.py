import math
from types import SimpleNamespace

import numpy as np

from level0_2d.reference.grid import Grid, T as IDX_T
from level0_2d.reference.advection import advection_rhs

U = 1.0
A = 1.0
T_INF = 0.0
SIGMA0 = 0.1
XC = 0.2
CFL = 0.4
T_FINAL = 0.3


def _exact(x, t):
    return T_INF + A * np.exp(-((x - U * t - XC) ** 2) / (2 * SIGMA0**2))


def _l2_error(n):
    p = SimpleNamespace(nx=n, ny=4, dx=1.0 / n, dy=1.0 / n, U=U)
    g = Grid(p)
    dt = CFL * p.dx / U
    nsteps = max(1, math.ceil(T_FINAL / dt))
    dt = T_FINAL / nsteps

    def fill_exact(t):
        g.q[IDX_T] = _exact(g.x[None, :], t) * np.ones((g.q.shape[1], 1))

    fill_exact(0.0)
    j, i = g.interior
    t = 0.0
    for _ in range(nsteps):
        rhs_t = advection_rhs(g, p)[0]
        interior_new = g.q[IDX_T, j, i] + dt * rhs_t
        t += dt
        fill_exact(t)
        g.q[IDX_T, j, i] = interior_new

    exact_final = _exact(g.x[i][None, :], T_FINAL) * np.ones((g.ny, 1))
    return float(np.sqrt(np.mean((g.q[IDX_T, j, i] - exact_final) ** 2)))


def test_advection_converges_but_below_second_order():
    errors = {n: _l2_error(n) for n in (64, 128, 256, 512)}
    orders = [
        math.log2(errors[a] / errors[b])
        for a, b in zip((64, 128, 256), (128, 256, 512))
    ]

    assert all(0.8 < order < 2.0 for order in orders)
    assert errors[512] < errors[64] / 4
