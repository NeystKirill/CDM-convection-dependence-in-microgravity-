import numpy as np
import pytest

ffcore = pytest.importorskip("emulator._ffcore")

from emulator.reference import Params as PyParams, Solver as PySolver

KWARGS = dict(
    fuel="CH4",
    n=40, Rb=0.00635, Rout=0.19, stretch=1.0,
    p=101325.0, mdot=2.0e-6, YF_inj=1.0,
    XO2_inf=0.21, T_inf=300.0, Tb=400.0,
    Tign=1600.0, tign=0.05, ignition_kernel_Rb=2.0,
    rad_enabled=True, rad_mult=1.0, a_p_ref=0.6,
    lam_ref=0.07, lam_exponent=0.7, cp=1167.0,
    Le_fuel=1.0, Le_o2=1.1, Le_product=1.2,
    Mo=0.032, Mn2=0.028, Mp=0.029,
    B=5.0e7, Ta=15000.0, Qcp=8600.0, s=4.0, Mf=0.01604,
    cfl=0.4, t_end=0.4, n_iter=12,
)

CPP_KWARGS = {k: v for k, v in KWARGS.items() if k != "fuel"}


def test_cpp_matches_python_reference_fields():
    py_solver = PySolver(PyParams(**KWARGS))
    cpp_solver = ffcore.Solver(ffcore.Params(**CPP_KWARGS))

    for _ in range(200):
        py_solver.step()
        cpp_solver.step()

    py_state = py_solver.state()
    cpp_state = cpp_solver.state()

    for key in ("T", "Yf", "Yo", "rho"):
        diff = np.abs(py_state[key] - np.asarray(cpp_state[key]))
        assert diff.max() < 1e-6


def test_cpp_matches_python_reference_qoi():
    py_solver = PySolver(PyParams(**KWARGS))
    cpp_solver = ffcore.Solver(ffcore.Params(**CPP_KWARGS))

    for _ in range(200):
        py_solver.step()
        cpp_solver.step()

    py_q = py_solver.qoi()
    cpp_q = cpp_solver.qoi()

    assert abs(py_q["T_max"] - cpp_q["T_max"]) < 1e-6
    assert abs(py_q["r_f"] - cpp_q["r_f"]) < 1e-9
    assert abs(py_q["chi_rad"] - cpp_q["chi_rad"]) < 1e-6
    assert py_q["regime"] == cpp_q["regime"]
