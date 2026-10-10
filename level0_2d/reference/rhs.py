import numpy as np

from .grid import T
from .advection import advection_rhs
from .diffusion import diffusion_rhs
from formulas import radiative_loss


def rhs(grid, params):
    out = advection_rhs(grid, params) + diffusion_rhs(grid, params)
    j, i = grid.interior
    Tfield = grid.q[T, j, i]
    qrad = radiative_loss(Tfield, params.T_inf, params.a_p)
    out[T] -= qrad / (params.rho * params.cp)
    return out
