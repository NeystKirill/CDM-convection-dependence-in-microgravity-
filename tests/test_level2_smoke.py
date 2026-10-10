import numpy as np

from level2 import Params, Solver


def _params(config="configs/candle_l2_base.yaml"):
    p = Params.from_yaml(config)
    p.nr = 12
    p.nz = 20
    p.t_end = 0.05
    return p


def test_level2_runs_without_nan():
    params = _params()
    solver = Solver(params)

    for _ in range(40):
        solver.step()
        assert np.all(np.isfinite(solver.T))
        assert np.all(np.isfinite(solver.Yf))
        assert np.all(np.isfinite(solver.Yo))
        assert np.all(np.isfinite(solver.u_r))
        assert np.all(np.isfinite(solver.u_z))


def test_level2_projection_is_divergence_free():
    params = _params()
    solver = Solver(params)

    for _ in range(10):
        solver.step()
        assert solver.div_max < 1e-8


def test_level2_ignites():
    params = _params()
    solver = Solver(params)

    for _ in range(40):
        solver.step()

    q = solver.qoi()
    assert q["T_max"] > params.T_inf + 500.0


def test_level2_species_bounded():
    params = _params()
    solver = Solver(params)

    for _ in range(40):
        solver.step()

    assert np.all(solver.Yf >= 0.0) and np.all(solver.Yf <= 1.0)
    assert np.all(solver.Yo >= 0.0) and np.all(solver.Yo <= 1.0)


def test_level2_high_flow_has_lower_damkohler_than_base():
    base = Solver(_params("configs/candle_l2_base.yaml"))
    high_flow = Solver(_params("configs/candle_l2_high_flow.yaml"))

    for _ in range(40):
        base.step()
        high_flow.step()

    assert high_flow.qoi()["da"] < base.qoi()["da"]
