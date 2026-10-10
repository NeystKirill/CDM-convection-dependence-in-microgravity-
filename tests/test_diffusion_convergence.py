import math
from types import SimpleNamespace

import numpy as np

from level0_2d.reference.grid import Grid, T as IDX_T
from level0_2d.reference.diffusion import diffusion_rhs

ALPHA = 1.0
A = 1.0
T_INF = 0.0
SIGMA0 = 0.08
XC = YC = 0.5
FO = 0.2
T_FINAL = 1e-3


def _exact(x, y, t):
    s2 = SIGMA0**2 + 2 * ALPHA * t
    return T_INF + A * (SIGMA0**2 / s2) * np.exp(-((x - XC) ** 2 + (y - YC) ** 2) / (2 * s2))


def _l2_error(n):
    p = SimpleNamespace(nx=n, ny=n, dx=1.0 / n, dy=1.0 / n, alpha=ALPHA, D=ALPHA)
    g = Grid(p)
    dt = FO * 0.5 / (ALPHA * (1.0 / p.dx**2 + 1.0 / p.dy**2))
    nsteps = max(1, math.ceil(T_FINAL / dt))
    dt = T_FINAL / nsteps

    def fill_exact(t):
        X, Y = np.meshgrid(g.x, g.y)
        g.q[IDX_T] = _exact(X, Y, t)

    fill_exact(0.0)
    j, i = g.interior
    t = 0.0
    for _ in range(nsteps):
        rhs_t = diffusion_rhs(g, p)[0]
        interior_new = g.q[IDX_T, j, i] + dt * rhs_t
        t += dt
        fill_exact(t)
        g.q[IDX_T, j, i] = interior_new

    X, Y = np.meshgrid(g.x[i], g.y[j])
    exact_final = _exact(X, Y, T_FINAL)
    return float(np.sqrt(np.mean((g.q[IDX_T, j, i] - exact_final) ** 2)))


def test_diffusion_is_second_order():
    errors = {n: _l2_error(n) for n in (32, 64, 128)}
    order_1 = math.log2(errors[32] / errors[64])
    order_2 = math.log2(errors[64] / errors[128])

    assert 1.8 < order_1 < 2.2
    assert 1.8 < order_2 < 2.2
