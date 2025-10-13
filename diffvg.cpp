#include "public_api.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(diffvg, m) {
    m.doc() = "Differential Vector Graphics";

    // Register python bindings in smaller chunks to keep this TU manageable
    // and to keep the public surface centralized in public_api.h.
    diffvg_bindings::register_runtime(m);
    diffvg_bindings::register_math(m);
    diffvg_bindings::register_ptr(m);
    diffvg_bindings::register_color(m);
    diffvg_bindings::register_shapes(m);
    diffvg_bindings::register_shape_group(m);
    diffvg_bindings::register_filter(m);
    diffvg_bindings::register_scene(m);
}
#define DIFFVG_NO_CUDA_RUNTIME_INCLUDES 1
