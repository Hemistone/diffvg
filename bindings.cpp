// Partially migrated pybind11 bindings live here to shrink diffvg.cpp.
// The module entry point (PYBIND11_MODULE) remains in diffvg.cpp for now.

#include <pybind11/pybind11.h>

#include "vector.h"

namespace py = pybind11;

namespace diffvg_bindings {

void register_math(py::module_ &m) {
    py::class_<Vector2f>(m, "Vector2f")
        .def(py::init<float, float>())
        .def_readwrite("x", &Vector2f::x)
        .def_readwrite("y", &Vector2f::y);

    py::class_<Vector3f>(m, "Vector3f")
        .def(py::init<float, float, float>())
        .def_readwrite("x", &Vector3f::x)
        .def_readwrite("y", &Vector3f::y)
        .def_readwrite("z", &Vector3f::z);

    py::class_<Vector4f>(m, "Vector4f")
        .def(py::init<float, float, float, float>())
        .def_readwrite("x", &Vector4f::x)
        .def_readwrite("y", &Vector4f::y)
        .def_readwrite("z", &Vector4f::z)
        .def_readwrite("w", &Vector4f::w);
}

} // namespace diffvg_bindings
