import numpy as np

from .grid import T as IDX_T, F as IDX_F, O as IDX_O
from formulas import reaction_rate


def diagnostics(grid, params):
    j, i = grid.interior
    Tf = grid.q[IDX_T, j, i]
    Yf = grid.q[IDX_F, j, i]
    Yo = grid.q[IDX_O, j, i]
    omega = reaction_rate(Tf, Yf, Yo, params.B, params.Ta)

    return {
        "Tmax": float(np.max(Tf)),
        "Yf_min": float(np.min(Yf)),
        "Yo_min": float(np.min(Yo)),
        "w_max": float(np.max(omega)),
        "has_nan": bool(not np.all(np.isfinite(grid.q))),
    }
