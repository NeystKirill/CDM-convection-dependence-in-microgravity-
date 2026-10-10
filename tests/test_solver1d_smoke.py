import numpy as np

from emulator.reference import Params, Solver


def _params():
    p = Params.from_yaml("configs/sflame_ch4.yaml")
    p.n = 40
    return p


def test_solver1d_runs_without_nan():
    params = _params()
    solver = Solver(params)

    for _ in range(300):
        solver.step()
        state = solver.state()
        assert np.all(np.isfinite(state["T"]))
        assert np.all(np.isfinite(state["Yf"]))
        assert np.all(np.isfinite(state["Yo"]))


def test_solver1d_ignites_and_grows():
    params = _params()
    solver = Solver(params)

    for _ in range(600):
        solver.step()

    q = solver.qoi()
    assert q["T_max"] > params.T_inf + 500.0
    assert q["r_f"] >= params.Rb


def test_solver1d_species_bounded():
    params = _params()
    solver = Solver(params)

    for _ in range(300):
        solver.step()

    assert np.all(solver.Yf >= 0.0) and np.all(solver.Yf <= 1.0)
    assert np.all(solver.Yo >= 0.0) and np.all(solver.Yo <= 1.0)
