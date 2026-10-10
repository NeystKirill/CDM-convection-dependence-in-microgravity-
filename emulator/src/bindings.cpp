#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "solver.hpp"

namespace py = pybind11;
using namespace ff1d;

namespace {

py::array_t<double> to_array(const std::vector<double>& v) {
    py::array_t<double> arr(v.size());
    auto buf = arr.mutable_unchecked<1>();
    for (size_t i = 0; i < v.size(); ++i) buf(i) = v[i];
    return arr;
}

}

PYBIND11_MODULE(_ffcore, m) {
    m.doc() = "C++ ядро уровня 1 (сферическое пламя)";

    py::class_<Params>(m, "Params")
        .def(py::init<int, double, double, double,
                       double, double, double,
                       double, double, double,
                       double, double, double,
                       bool, double, double,
                       double, double, double,
                       double, double, double,
                       double, double, double,
                       double, double, double, double, double,
                       double, double, int>(),
             py::arg("n"), py::arg("Rb"), py::arg("Rout"), py::arg("stretch"),
             py::arg("p"), py::arg("mdot"), py::arg("YF_inj"),
             py::arg("XO2_inf"), py::arg("T_inf"), py::arg("Tb"),
             py::arg("Tign"), py::arg("tign"), py::arg("ignition_kernel_Rb"),
             py::arg("rad_enabled"), py::arg("rad_mult"), py::arg("a_p_ref"),
             py::arg("lam_ref"), py::arg("lam_exponent"), py::arg("cp"),
             py::arg("Le_fuel"), py::arg("Le_o2"), py::arg("Le_product"),
             py::arg("Mo"), py::arg("Mn2"), py::arg("Mp"),
             py::arg("B"), py::arg("Ta"), py::arg("Qcp"), py::arg("s"), py::arg("Mf"),
             py::arg("cfl"), py::arg("t_end"), py::arg("n_iter"))
        .def_readonly("Yo_inf", &Params::Yo_inf);

    py::class_<Solver>(m, "Solver")
        .def(py::init<Params>())
        .def("step", &Solver::step)
        .def("run", &Solver::run, py::arg("t_end"), py::arg("max_steps") = -1)
        .def_property_readonly("t", &Solver::time)
        .def_property_readonly("n_steps", &Solver::steps)
        .def("state", [](const Solver& s) {
            py::dict d;
            d["r"] = to_array(s.grid().r_c);
            d["T"] = to_array(s.T());
            d["Yf"] = to_array(s.Yf());
            d["Yo"] = to_array(s.Yo());
            d["rho"] = to_array(s.density());
            return d;
        })
        .def("qoi", [](const Solver& s) {
            py::dict d;
            double chi = s.chi_rad();
            d["t"] = s.time();
            d["r_f"] = s.r_f();
            d["T_max"] = s.T_max();
            d["chi_rad"] = chi;
            d["eta"] = s.eta();
            if (s.tracker().t_ext.has_value()) {
                d["t_ext"] = s.tracker().t_ext.value();
            } else {
                d["t_ext"] = py::none();
            }
            d["regime"] = s.tracker().regime(chi);
            return d;
        });
}
