#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "solver.hpp"

namespace py = pybind11;
using namespace blueball;

namespace {

py::array_t<double> extract_field(const Solver& solver, int field) {
    const Grid& g = solver.grid();
    py::array_t<double> arr({g.ny, g.nx});
    auto buf = arr.mutable_unchecked<2>();
    for (int j = 0; j < g.ny; ++j) {
        const int gj = g.ng + j;
        for (int i = 0; i < g.nx; ++i) {
            const int gi = g.ng + i;
            buf(j, i) = g.at(field, gj, gi);
        }
    }
    return arr;
}

}

PYBIND11_MODULE(_emu, m) {
    m.doc() = "C++ ядро эмулятора";

    py::class_<Params>(m, "Params")
        .def(py::init<int, int, double, double,
                       double, double, double,
                       double, double, double,
                       double, double, double, double,
                       double, double, double, double,
                       double,
                       double, double, double, double, double,
                       double, double, double>(),
             py::arg("nx"), py::arg("ny"), py::arg("Lx"), py::arg("Ly"),
             py::arg("rho"), py::arg("cp"), py::arg("lam"),
             py::arg("U"), py::arg("T_inf"), py::arg("Yo_inf"),
             py::arg("fuel_x0"), py::arg("fuel_x1"), py::arg("Ts"), py::arg("Yf_s"),
             py::arg("B"), py::arg("Ta"), py::arg("Qcp"), py::arg("s"),
             py::arg("a_p"),
             py::arg("ign_T"), py::arg("ign_t"), py::arg("ign_x0"), py::arg("ign_x1"), py::arg("ign_y1"),
             py::arg("t_end"), py::arg("cfl"), py::arg("fo"))
        .def_readonly("dx", &Params::dx)
        .def_readonly("dy", &Params::dy)
        .def_readonly("alpha", &Params::alpha)
        .def_readonly("D", &Params::D)
        .def_readonly("dt", &Params::dt);

    py::class_<Solver>(m, "Solver")
        .def(py::init<Params>())
        .def("step", &Solver::step)
        .def("run", &Solver::run, py::arg("t_end"), py::arg("max_steps") = -1)
        .def_property_readonly("t", &Solver::time)
        .def_property_readonly("n_steps", &Solver::steps)
        .def("fields", [](const Solver& s) {
            py::dict d;
            d["T"] = extract_field(s, Field::T);
            d["Yf"] = extract_field(s, Field::F);
            d["Yo"] = extract_field(s, Field::O);
            return d;
        })
        .def("diagnostics", [](const Solver& s) {
            Diagnostics d = s.diagnostics();
            py::dict out;
            out["Tmax"] = d.Tmax;
            out["Yf_min"] = d.Yf_min;
            out["Yo_min"] = d.Yo_min;
            out["w_max"] = d.w_max;
            out["has_nan"] = d.has_nan;
            return out;
        });
}
