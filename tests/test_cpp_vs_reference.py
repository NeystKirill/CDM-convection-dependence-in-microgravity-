import numpy as np
import pytest

emu = pytest.importorskip("level0_2d._emu")

from level0_2d.reference import Params as PyParams, Solver as PySolver

KWARGS = dict(
    nx=40, ny=20, Lx=0.10, Ly=0.04,
    rho=0.3, cp=1200.0, lam=0.07,
    U=0.20, T_inf=300.0, Yo_inf=0.233,
    fuel_x0=0.02, fuel_x1=0.08, Ts=600.0, Yf_s=1.0,
    B=5.0e7, Ta=15000.0, Qcp=8600.0, s=1.0,
    a_p=1.0,
    ign_T=1600.0, ign_t=0.15, ign_x0=0.02, ign_x1=0.035, ign_y1=0.006,
    t_end=1.0, cfl=0.4, fo=0.2,
)


def test_cpp_matches_python_reference_fields():
    py_solver = PySolver(PyParams(**KWARGS))
    cpp_solver = emu.Solver(emu.Params(**KWARGS))

    for _ in range(300):
        py_solver.step()
        cpp_solver.step()

    py_fields = py_solver.fields()
    cpp_fields = cpp_solver.fields()

    for key in ("T", "Yf", "Yo"):
        diff = np.abs(py_fields[key] - cpp_fields[key])
        assert diff.max() < 1e-6


def test_cpp_matches_python_reference_diagnostics():
    py_solver = PySolver(PyParams(**KWARGS))
    cpp_solver = emu.Solver(emu.Params(**KWARGS))

    for _ in range(300):
        py_solver.step()
        cpp_solver.step()

    py_diag = py_solver.diagnostics()
    cpp_diag = cpp_solver.diagnostics()

    assert abs(py_diag["Tmax"] - cpp_diag["Tmax"]) < 1e-6
    assert py_diag["has_nan"] is False
    assert cpp_diag["has_nan"] is False
