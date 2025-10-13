#pragma once

// Central header describing the C++ surface that Python bindings are allowed to
// touch. This keeps the pybind11 translation unit decoupled from the rest of
// the renderer internals while making the exposed API explicit.

#include "ptr.h"
#include "vector.h"
#include "filter.h"
#include "color.h"
#include "shape.h"
#include "scene.h"
#include "render.h"

namespace pybind11 {
class module_;
} // namespace pybind11

namespace diffvg_bindings {
void register_math(pybind11::module_ &m);
void register_ptr(pybind11::module_ &m);
void register_color(pybind11::module_ &m);
void register_shapes(pybind11::module_ &m);
void register_shape_group(pybind11::module_ &m);
void register_filter(pybind11::module_ &m);
void register_scene(pybind11::module_ &m);
void register_runtime(pybind11::module_ &m);
} // namespace diffvg_bindings

