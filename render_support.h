#pragma once

#include <cassert>

#include "diffvg.h"
#include "color.h"
#include "edge_query.h"
#include "filter.h"
#include "scene.h"
#include "vector.h"

struct BoundaryData;
struct ClosestPointPathInfo;

DEVICE bool is_inside(const SceneData &scene_data,
                      int shape_group_id,
                      const Vector2f &pt,
                      EdgeQuery *edge_query);

DEVICE void accumulate_boundary_gradient(const Shape &shape,
                                         float contrib,
                                         float t,
                                         const Vector2f &normal,
                                         const BoundaryData &boundary_data,
                                         Shape &d_shape,
                                         const Matrix3x3f &shape_to_canvas,
                                         const Vector2f &local_boundary_pt,
                                         Matrix3x3f &d_shape_to_canvas);

DEVICE Vector4f sample_color(const ColorType &color_type,
                             void *color,
                             const Vector2f &pt);

DEVICE void d_sample_color(const ColorType &color_type,
                           void *color_ptr,
                           const Vector2f &pt,
                           const Vector4f &d_color,
                           void *d_color_ptr,
                           float *d_translation);

DEVICE Vector4f sample_color(const SceneData &scene,
                             const Vector4f *background_color,
                             const Vector2f &screen_pt,
                             const Vector4f *d_color,
                             EdgeQuery *edge_query,
                             Vector4f *d_background_color,
                             float *d_translation);

DEVICE Vector4f sample_color_prefiltered(const SceneData &scene,
                                         const Vector4f *background_color,
                                         const Vector2f &screen_pt,
                                         const Vector4f *d_color,
                                         Vector4f *d_background_color,
                                         float *d_translation);

DEVICE float smoothstep(float d);
DEVICE float d_smoothstep(float d, float d_ret);

DEVICE Vector4f gather_d_color(const Filter &filter,
                               const float *d_color_image,
                               const float *weight_image,
                               int width,
                               int height,
                               const Vector2f &pt);

DEVICE float sample_distance(const SceneData &scene,
                             const Vector2f &screen_pt,
                             float weight,
                             const float *d_dist,
                             float *d_translation);

// Device-capable implementations are provided in render_support.cpp; when
// compiling with CUDA, separable device linking is enabled so device symbols
// resolve across translation units.
