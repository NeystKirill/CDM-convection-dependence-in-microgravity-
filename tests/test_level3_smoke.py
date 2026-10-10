import numpy as np

from level3 import Params, Solver


def _params(config="configs/candle_l3_base.yaml"):
    p = Params.from_yaml(config)
    p.nx = 8
    p.ny = 8
    p.nz = 12
    p.t_end = 0.03
    return p


def test_level3_runs_without_nan():
    params = _params()
    solver = Solver(params)

    for _ in range(20):
        solver.step()
        assert np.all(np.isfinite(solver.T))
        assert np.all(np.isfinite(solver.Yf))
        assert np.all(np.isfinite(solver.Yo))
        assert np.all(np.isfinite(solver.u_x))
        assert np.all(np.isfinite(solver.u_y))
        assert np.all(np.isfinite(solver.u_z))


def test_level3_projection_is_divergence_free():
    params = _params()
    solver = Solver(params)

    for _ in range(10):
        solver.step()
        assert solver.div_max < 1e-8


def test_level3_ignites():
    params = _params()
    solver = Solver(params)

    for _ in range(20):
        solver.step()

    q = solver.qoi()
    assert q["T_max"] > params.T_inf + 500.0


def test_level3_species_bounded():
    params = _params()
    solver = Solver(params)

    for _ in range(20):
        solver.step()

    assert np.all(solver.Yf >= 0.0) and np.all(solver.Yf <= 1.0)
    assert np.all(solver.Yo >= 0.0) and np.all(solver.Yo <= 1.0)


def test_level3_offcenter_wick_shifts_flame_location():
    params = _params("configs/candle_l3_offcenter.yaml")
    solver = Solver(params)

    for _ in range(20):
        solver.step()

    q = solver.qoi()
    domain_center_x = params.Lx / 2.0
    assert abs(q["x_f"] - params.wick_x0) < abs(q["x_f"] - domain_center_x)
