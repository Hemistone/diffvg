#include "diffvg.h"
#include "render.h"
#include "scene.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Forward registration from bindings.cpp (incremental migration)
namespace diffvg_bindings {
    void register_math(py::module_ &m);
    void register_ptr(py::module_ &m);
    void register_color(py::module_ &m);
    void register_shapes(py::module_ &m);
    void register_shape_group(py::module_ &m);
    void register_filter(py::module_ &m);
    void register_scene(py::module_ &m);
}

PYBIND11_MODULE(diffvg, m) {
    m.doc() = "Differential Vector Graphics";

#ifdef COMPILE_WITH_CUDA
    constexpr bool diffvg_compiled_with_cuda = true;
#else
    constexpr bool diffvg_compiled_with_cuda = false;
#endif
    m.def("is_cuda_compiled", []() { return diffvg_compiled_with_cuda; },
          "Return True if diffvg was built with CUDA support.");

    // Register python bindings in smaller chunks to keep this TU manageable.
    diffvg_bindings::register_math(m);
    diffvg_bindings::register_ptr(m);
    diffvg_bindings::register_color(m);
    diffvg_bindings::register_shapes(m);
    diffvg_bindings::register_shape_group(m);
    diffvg_bindings::register_filter(m);
    diffvg_bindings::register_scene(m);

    m.def("render", &render, "");
}
#define DIFFVG_NO_CUDA_RUNTIME_INCLUDES 1
