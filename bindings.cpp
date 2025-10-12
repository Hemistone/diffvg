// Partially migrated pybind11 bindings live here to shrink diffvg.cpp.
// The module entry point (PYBIND11_MODULE) remains in diffvg.cpp for now.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "vector.h"
#include "ptr.h"
#include "filter.h"
#include "color.h"

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

void register_ptr(py::module_ &m) {
    py::class_<ptr<void>>(m, "void_ptr")
        .def(py::init<std::size_t>())
        .def("as_size_t", &ptr<void>::as_size_t);
    py::class_<ptr<float>>(m, "float_ptr")
        .def(py::init<std::size_t>());
    py::class_<ptr<int>>(m, "int_ptr")
        .def(py::init<std::size_t>());
}

void register_filter(py::module_ &m) {
    py::enum_<FilterType>(m, "FilterType")
        .value("box", FilterType::Box)
        .value("tent", FilterType::Tent)
        .value("parabolic", FilterType::RadialParabolic)
        .value("hann", FilterType::Hann);

    py::class_<Filter>(m, "Filter")
        .def(py::init<FilterType, float>());
}

void register_color(py::module_ &m) {
    py::enum_<ColorType>(m, "ColorType")
        .value("constant", ColorType::Constant)
        .value("linear_gradient", ColorType::LinearGradient)
        .value("radial_gradient", ColorType::RadialGradient);

    py::class_<Constant>(m, "Constant")
        .def(py::init<Vector4f>())
        .def("get_ptr", &Constant::get_ptr)
        .def_readonly("color", &Constant::color);

    py::class_<LinearGradient>(m, "LinearGradient")
        .def(py::init<Vector2f, Vector2f, int, ptr<float>, ptr<float>>())
        .def("get_ptr", &LinearGradient::get_ptr)
        .def("copy_to", &LinearGradient::copy_to)
        .def_readonly("begin", &LinearGradient::begin)
        .def_readonly("end", &LinearGradient::end)
        .def_readonly("num_stops", &LinearGradient::num_stops);

    py::class_<RadialGradient>(m, "RadialGradient")
        .def(py::init<Vector2f, Vector2f, int, ptr<float>, ptr<float>>())
        .def("get_ptr", &RadialGradient::get_ptr)
        .def("copy_to", &RadialGradient::copy_to)
        .def_readonly("center", &RadialGradient::center)
        .def_readonly("radius", &RadialGradient::radius)
        .def_readonly("num_stops", &RadialGradient::num_stops);
}

} // namespace diffvg_bindings
