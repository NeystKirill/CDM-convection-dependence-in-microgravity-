from types import SimpleNamespace

import numpy as np

from emulator.reference.grid import Grid, T as IDX_T, F as IDX_F, O as IDX_O
from emulator.reference.reaction import react_step


def _make_state(n, seed):
    rng = np.random.default_rng(seed)
    p = SimpleNamespace(nx=n, ny=n, dx=1.0 / n, dy=1.0 / n, B=5.0e7, Ta=15000.0, Qcp=8600.0, s=1.0)
    g = Grid(p)
    j, i = g.interior
    g.q[IDX_T, j, i] = rng.uniform(800.0, 2500.0, size=(n, n))
    g.q[IDX_F, j, i] = rng.uniform(0.0, 1.0, size=(n, n))
    g.q[IDX_O, j, i] = rng.uniform(0.0, 0.3, size=(n, n))
    return g, p


def test_reaction_conserves_invariants_and_stays_nonnegative():
    g, p = _make_state(n=20, seed=0)
    j, i = g.interior

    inv_T_before = g.q[IDX_T, j, i] + p.Qcp * g.q[IDX_F, j, i]
    inv_O_before = g.q[IDX_O, j, i] - p.s * g.q[IDX_F, j, i]

    react_step(g, p, dt_half=1e-3)

    inv_T_after = g.q[IDX_T, j, i] + p.Qcp * g.q[IDX_F, j, i]
    inv_O_after = g.q[IDX_O, j, i] - p.s * g.q[IDX_F, j, i]

    assert np.max(np.abs(inv_T_after - inv_T_before)) < 1e-6
    assert np.max(np.abs(inv_O_after - inv_O_before)) < 1e-9
    assert np.min(g.q[IDX_F, j, i]) >= -1e-12
    assert np.min(g.q[IDX_O, j, i]) >= -1e-12


def test_reaction_does_not_exceed_initial_reactants():
    g, p = _make_state(n=20, seed=1)
    j, i = g.interior
    Yf0 = g.q[IDX_F, j, i].copy()
    Yo0 = g.q[IDX_O, j, i].copy()

    react_step(g, p, dt_half=1e-2)

    assert np.all(g.q[IDX_F, j, i] <= Yf0 + 1e-12)
    assert np.all(g.q[IDX_O, j, i] <= Yo0 + 1e-12)
