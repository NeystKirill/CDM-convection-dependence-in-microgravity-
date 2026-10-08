#include <pybind11/pybind11.h>
namespace py = pybind11;
PYBIND11_MODULE(_emu, m) { m.doc() = "C++ ядро эмулятора"; }
