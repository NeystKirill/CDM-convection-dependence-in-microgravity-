from level0_2d.reference import Params, Solver


def _small_params():
    return Params(
        nx=40, ny=20, Lx=0.10, Ly=0.04,
        rho=0.3, cp=1200.0, lam=0.07,
        U=0.20, T_inf=300.0, Yo_inf=0.233,
        fuel_x0=0.02, fuel_x1=0.08, Ts=600.0, Yf_s=1.0,
        B=5.0e7, Ta=15000.0, Qcp=8600.0, s=1.0,
        a_p=1.0,
        ign_T=1600.0, ign_t=0.15, ign_x0=0.02, ign_x1=0.035, ign_y1=0.006,
        t_end=1.0, cfl=0.4, fo=0.2,
    )


def test_solver_runs_without_nan_and_heats_up():
    params = _small_params()
    solver = Solver(params)

    diag0 = solver.diagnostics()
    assert not diag0["has_nan"]

    for _ in range(200):
        solver.step()

    diag = solver.diagnostics()
    assert not diag["has_nan"]
    assert diag["Tmax"] > params.T_inf + 500.0
    assert diag["Yf_min"] >= -1e-9
    assert diag["Yo_min"] >= -1e-9
