#include "render_support.h"

#include "atomic.h"
#include "cdf.h"
#include "compute_distance.h"
#include "cuda_utils.h"
#include "sample_boundary.h"
#include "matrix.h"
#include "parallel.h"
#include "pcg.h"
#include "winding_number.h"
#include "within_distance.h"

#include <cassert>
#include <cmath>


DEVICE
Vector4f sample_color(const ColorType &color_type,
                      void *color,
                      const Vector2f &pt) {
    switch (color_type) {
        case ColorType::Constant: {
            auto c = (const Constant*)color;
            assert(isfinite(c->color));
            return c->color;
        } case ColorType::LinearGradient: {
            auto c = (const LinearGradient*)color;
            // Project pt to (c->begin, c->end)
            auto beg = c->begin;
            auto end = c->end;
            auto t = dot(pt - beg, end - beg) / max(dot(end - beg, end - beg), 1e-3f);
            // Find the correponding stop:
            if (t < c->stop_offsets[0]) {
                return Vector4f{c->stop_colors[0],
                                c->stop_colors[1],
                                c->stop_colors[2],
                                c->stop_colors[3]};
            }
            for (int i = 0; i < c->num_stops - 1; i++) {
                auto offset_curr = c->stop_offsets[i];
                auto offset_next = c->stop_offsets[i + 1];
                assert(offset_next > offset_curr);
                if (t >= offset_curr && t < offset_next) {
                    auto color_curr = Vector4f{
                        c->stop_colors[4 * i + 0],
                        c->stop_colors[4 * i + 1],
                        c->stop_colors[4 * i + 2],
                        c->stop_colors[4 * i + 3]};
                    auto color_next = Vector4f{
                        c->stop_colors[4 * (i + 1) + 0],
                        c->stop_colors[4 * (i + 1) + 1],
                        c->stop_colors[4 * (i + 1) + 2],
                        c->stop_colors[4 * (i + 1) + 3]};
                    auto tt = (t - offset_curr) / (offset_next - offset_curr);
                    assert(isfinite(tt));
                    assert(isfinite(color_curr));
                    assert(isfinite(color_next));
                    return color_curr * (1 - tt) + color_next * tt;
                }
            }
            return Vector4f{c->stop_colors[4 * (c->num_stops - 1) + 0],
                            c->stop_colors[4 * (c->num_stops - 1) + 1],
                            c->stop_colors[4 * (c->num_stops - 1) + 2],
                            c->stop_colors[4 * (c->num_stops - 1) + 3]};
        } case ColorType::RadialGradient: {
            auto c = (const RadialGradient*)color;
            // Distance from pt to center
            auto offset = pt - c->center;
            auto normalized_offset = offset / c->radius;
            auto t = length(normalized_offset);
            // Find the correponding stop:
            if (t < c->stop_offsets[0]) {
                return Vector4f{c->stop_colors[0],
                                c->stop_colors[1],
                                c->stop_colors[2],
                                c->stop_colors[3]};
            }
            for (int i = 0; i < c->num_stops - 1; i++) {
                auto offset_curr = c->stop_offsets[i];
                auto offset_next = c->stop_offsets[i + 1];
                assert(offset_next > offset_curr);
                if (t >= offset_curr && t < offset_next) {
                    auto color_curr = Vector4f{
                        c->stop_colors[4 * i + 0],
                        c->stop_colors[4 * i + 1],
                        c->stop_colors[4 * i + 2],
                        c->stop_colors[4 * i + 3]};
                    auto color_next = Vector4f{
                        c->stop_colors[4 * (i + 1) + 0],
                        c->stop_colors[4 * (i + 1) + 1],
                        c->stop_colors[4 * (i + 1) + 2],
                        c->stop_colors[4 * (i + 1) + 3]};
                    auto tt = (t - offset_curr) / (offset_next - offset_curr);
                    assert(isfinite(tt));
                    assert(isfinite(color_curr));
                    assert(isfinite(color_next));
                    return color_curr * (1 - tt) + color_next * tt;
                }
            }
            return Vector4f{c->stop_colors[4 * (c->num_stops - 1) + 0],
                            c->stop_colors[4 * (c->num_stops - 1) + 1],
                            c->stop_colors[4 * (c->num_stops - 1) + 2],
                            c->stop_colors[4 * (c->num_stops - 1) + 3]};
        } default: {
            assert(false);
        }
    }
    return Vector4f{};
}

DEVICE
void d_sample_color(const ColorType &color_type,
                    void *color_ptr,
                    const Vector2f &pt,
                    const Vector4f &d_color,
                    void *d_color_ptr,
                    float *d_translation) {
    switch (color_type) {
        case ColorType::Constant: {
            auto d_c = (Constant*)d_color_ptr;
            atomic_add(&d_c->color[0], d_color);
            return;
        } case ColorType::LinearGradient: {
            auto c = (const LinearGradient*)color_ptr;
            auto d_c = (LinearGradient*)d_color_ptr;
            // Project pt to (c->begin, c->end)
            auto beg = c->begin;
            auto end = c->end;
            auto t = dot(pt - beg, end - beg) / max(dot(end - beg, end - beg), 1e-3f);
            // Find the correponding stop:
            if (t < c->stop_offsets[0]) {
                atomic_add(&d_c->stop_colors[0], d_color);
                return;
            }
            for (int i = 0; i < c->num_stops - 1; i++) {
                auto offset_curr = c->stop_offsets[i];
                auto offset_next = c->stop_offsets[i + 1];
                assert(offset_next > offset_curr);
                if (t >= offset_curr && t < offset_next) {
                    auto color_curr = Vector4f{
                        c->stop_colors[4 * i + 0],
                        c->stop_colors[4 * i + 1],
                        c->stop_colors[4 * i + 2],
                        c->stop_colors[4 * i + 3]};
                    auto color_next = Vector4f{
                        c->stop_colors[4 * (i + 1) + 0],
                        c->stop_colors[4 * (i + 1) + 1],
                        c->stop_colors[4 * (i + 1) + 2],
                        c->stop_colors[4 * (i + 1) + 3]};
                    auto tt = (t - offset_curr) / (offset_next - offset_curr);
                    // return color_curr * (1 - tt) + color_next * tt;
                    auto d_color_curr = d_color * (1 - tt);
                    auto d_color_next = d_color * tt;
                    auto d_tt = sum(d_color * (color_next - color_curr));
                    auto d_offset_next = -d_tt * tt / (offset_next - offset_curr);
                    auto d_offset_curr = d_tt * ((tt - 1.f) / (offset_next - offset_curr));
                    auto d_t = d_tt / (offset_next - offset_curr);
                    assert(isfinite(d_tt));
                    atomic_add(&d_c->stop_colors[4 * i], d_color_curr);
                    atomic_add(&d_c->stop_colors[4 * (i + 1)], d_color_next);
                    atomic_add(&d_c->stop_offsets[i], d_offset_curr);
                    atomic_add(&d_c->stop_offsets[i + 1], d_offset_next);
                    // auto t = dot(pt - beg, end - beg) / max(dot(end - beg, end - beg), 1e-6f);
                    // l = max(dot(end - beg, end - beg), 1e-3f)
                    // t = dot(pt - beg, end - beg) / l;
                    auto l = max(dot(end - beg, end - beg), 1e-3f);
                    auto d_beg = d_t * (-(pt - beg)-(end - beg)) / l;
                    auto d_end = d_t * (pt - beg) / l;
                    auto d_l = -d_t * t / l;
                    if (dot(end - beg, end - beg) > 1e-3f) {
                        d_beg += 2 * d_l * (beg - end);
                        d_end += 2 * d_l * (end - beg);
                    }
                    atomic_add(&d_c->begin[0], d_beg);
                    atomic_add(&d_c->end[0], d_end);
                    if (d_translation != nullptr) {
                        atomic_add(d_translation, (d_beg + d_end));
                    }
                    return;
                }
            }
            atomic_add(&d_c->stop_colors[4 * (c->num_stops - 1)], d_color);
            return;
        } case ColorType::RadialGradient: {
            auto c = (const RadialGradient*)color_ptr;
            auto d_c = (RadialGradient*)d_color_ptr;
            // Distance from pt to center
            auto offset = pt - c->center;
            auto normalized_offset = offset / c->radius;
            auto t = length(normalized_offset);
            // Find the correponding stop:
            if (t < c->stop_offsets[0]) {
                atomic_add(&d_c->stop_colors[0], d_color);
                return;
            }
            for (int i = 0; i < c->num_stops - 1; i++) {
                auto offset_curr = c->stop_offsets[i];
                auto offset_next = c->stop_offsets[i + 1];
                assert(offset_next > offset_curr);
                if (t >= offset_curr && t < offset_next) {
                    auto color_curr = Vector4f{
                        c->stop_colors[4 * i + 0],
                        c->stop_colors[4 * i + 1],
                        c->stop_colors[4 * i + 2],
                        c->stop_colors[4 * i + 3]};
                    auto color_next = Vector4f{
                        c->stop_colors[4 * (i + 1) + 0],
                        c->stop_colors[4 * (i + 1) + 1],
                        c->stop_colors[4 * (i + 1) + 2],
                        c->stop_colors[4 * (i + 1) + 3]};
                    auto tt = (t - offset_curr) / (offset_next - offset_curr);
                    assert(isfinite(tt));
                    // return color_curr * (1 - tt) + color_next * tt;
                    auto d_color_curr = d_color * (1 - tt);
                    auto d_color_next = d_color * tt;
                    auto d_tt = sum(d_color * (color_next - color_curr));
                    auto d_offset_next = -d_tt * tt / (offset_next - offset_curr);
                    auto d_offset_curr = d_tt * ((tt - 1.f) / (offset_next - offset_curr));
                    auto d_t = d_tt / (offset_next - offset_curr);
                    assert(isfinite(d_t));
                    atomic_add(&d_c->stop_colors[4 * i], d_color_curr);
                    atomic_add(&d_c->stop_colors[4 * (i + 1)], d_color_next);
                    atomic_add(&d_c->stop_offsets[i], d_offset_curr);
                    atomic_add(&d_c->stop_offsets[i + 1], d_offset_next);
                    // offset = pt - c->center
                    // normalized_offset = offset / c->radius
                    // t = length(normalized_offset)
                    auto d_normalized_offset = d_length(normalized_offset, d_t);
                    auto d_offset = d_normalized_offset / c->radius;
                    auto d_radius = -d_normalized_offset * offset / (c->radius * c->radius);
                    auto d_center = -d_offset;
                    atomic_add(&d_c->center[0], d_center);
                    atomic_add(&d_c->radius[0], d_radius);
                    if (d_translation != nullptr) {
                        atomic_add(d_translation, d_center);
                    }
                }
            }
            atomic_add(&d_c->stop_colors[4 * (c->num_stops - 1)], d_color);
            return;
        } default: {
            assert(false);
        }
    }
}

 

struct Fragment {
    Vector3f color;
    float alpha;
    int group_id;
    bool is_stroke;
};

struct PrefilterFragment {
    Vector3f color;
    float alpha;
    int group_id;
    bool is_stroke;
    int shape_id;
    float distance;
    Vector2f closest_pt;
    ClosestPointPathInfo path_info;
    bool within_distance;
};
// INSERT_AFTER_STRUCTS
DEVICE
Vector4f sample_color(const SceneData &scene,
                      const Vector4f *background_color,
                      const Vector2f &screen_pt,
                      const Vector4f *d_color,
                      EdgeQuery *edge_query,
                      Vector4f *d_background_color,
                      float *d_translation) {
    if (edge_query != nullptr) {
        edge_query->hit = false;
    }

    auto pt = screen_pt;
    pt.x *= scene.canvas_width;
    pt.y *= scene.canvas_height;
    constexpr auto max_hit_shapes = 256;
    constexpr auto max_bvh_stack_size = 64;
    Fragment fragments[max_hit_shapes];
    int bvh_stack[max_bvh_stack_size];
    auto stack_size = 0;
    auto num_fragments = 0;
    bvh_stack[stack_size++] = 2 * scene.num_shape_groups - 2;
    while (stack_size > 0) {
        const BVHNode &node = scene.bvh_nodes[bvh_stack[--stack_size]];
        if (node.child1 < 0) {
            auto group_id = node.child0;
            const ShapeGroup &shape_group = scene.shape_groups[group_id];
            if (shape_group.stroke_color != nullptr) {
                if (within_distance(scene, group_id, pt, edge_query)) {
                    auto color_alpha = sample_color(shape_group.stroke_color_type,
                                                    shape_group.stroke_color,
                                                    pt);
                    Fragment f;
                    f.color = Vector3f{color_alpha[0], color_alpha[1], color_alpha[2]};
                    f.alpha = color_alpha[3];
                    f.group_id = group_id;
                    f.is_stroke = true;
                    assert(num_fragments < max_hit_shapes);
                    fragments[num_fragments++] = f;
                }
            }
            if (shape_group.fill_color != nullptr) {
                if (is_inside(scene, group_id, pt, edge_query)) {
                    auto color_alpha = sample_color(shape_group.fill_color_type,
                                                    shape_group.fill_color,
                                                    pt);
                    Fragment f;
                    f.color = Vector3f{color_alpha[0], color_alpha[1], color_alpha[2]};
                    f.alpha = color_alpha[3];
                    f.group_id = group_id;
                    f.is_stroke = false;
                    assert(num_fragments < max_hit_shapes);
                    fragments[num_fragments++] = f;
                }
            }
        } else {
            assert(node.child0 >= 0 && node.child1 >= 0);
            const AABB &b0 = scene.bvh_nodes[node.child0].box;
            if (inside(b0, pt, scene.bvh_nodes[node.child0].max_radius)) {
                bvh_stack[stack_size++] = node.child0;
            }
            const AABB &b1 = scene.bvh_nodes[node.child1].box;
            if (inside(b1, pt, scene.bvh_nodes[node.child1].max_radius)) {
                bvh_stack[stack_size++] = node.child1;
            }
            assert(stack_size <= max_bvh_stack_size);
        }
    }
    if (num_fragments <= 0) {
        if (background_color != nullptr) {
            if (d_background_color != nullptr) {
                *d_background_color = *d_color;
            }
            return *background_color;
        }
        return Vector4f{0, 0, 0, 0};
    }
    for (int i = 1; i < num_fragments; i++) {
        auto j = i;
        auto temp = fragments[j];
        while (j > 0 && fragments[j - 1].group_id > temp.group_id) {
            fragments[j] = fragments[j - 1];
            j--;
        }
        fragments[j] = temp;
    }
    Vector3f accum_color[max_hit_shapes];
    float accum_alpha[max_hit_shapes];
    auto first_alpha = 0.f;
    auto first_color = Vector3f{0, 0, 0};
    if (background_color != nullptr) {
        first_alpha = background_color->w;
        first_color = Vector3f{background_color->x,
                               background_color->y,
                               background_color->z};
    }
    for (int i = 0; i < num_fragments; i++) {
        const Fragment &fragment = fragments[i];
        auto new_color = fragment.color;
        auto new_alpha = fragment.alpha;
        auto prev_alpha = i > 0 ? accum_alpha[i - 1] : first_alpha;
        auto prev_color = i > 0 ? accum_color[i - 1] : first_color;
        if (edge_query != nullptr) {
            if (new_alpha >= 1.f && edge_query->hit) {
                edge_query->hit = false;
            }
            if (edge_query->shape_group_id == fragment.group_id) {
                edge_query->hit = true;
            }
        }
        accum_color[i] = prev_color * (1 - new_alpha) + new_alpha * new_color;
        accum_alpha[i] = prev_alpha * (1 - new_alpha) + new_alpha;
    }
    auto final_color = accum_color[num_fragments - 1];
    auto final_alpha = accum_alpha[num_fragments - 1];
    if (final_alpha > 1e-6f) {
        final_color /= final_alpha;
    }
    assert(isfinite(final_color));
    assert(isfinite(final_alpha));
    if (d_color != nullptr) {
        auto d_final_color = Vector3f{(*d_color)[0], (*d_color)[1], (*d_color)[2]};
        auto d_final_alpha = (*d_color)[3];
        auto d_curr_color = d_final_color;
        auto d_curr_alpha = d_final_alpha;
        if (final_alpha > 1e-6f) {
            d_curr_color = d_final_color / final_alpha;
            d_curr_alpha -= sum(d_final_color * final_color) / final_alpha;
        }
        assert(isfinite(*d_color));
        assert(isfinite(d_curr_color));
        assert(isfinite(d_curr_alpha));
        for (int i = num_fragments - 1; i >= 0; i--) {
            auto prev_alpha = i > 0 ? accum_alpha[i - 1] : first_alpha;
            auto prev_color = i > 0 ? accum_color[i - 1] : first_color;
            auto d_prev_alpha = d_curr_alpha * (1.f - fragments[i].alpha);
            auto d_alpha_i = d_curr_alpha * (1.f - prev_alpha);
            d_alpha_i += sum(d_curr_color * (fragments[i].color - prev_color));
            auto d_prev_color = d_curr_color * (1 - fragments[i].alpha);
            auto d_color_i = d_curr_color * fragments[i].alpha;
            auto group_id = fragments[i].group_id;
            if (fragments[i].is_stroke) {
                d_sample_color(scene.shape_groups[group_id].stroke_color_type,
                               scene.shape_groups[group_id].stroke_color,
                               pt,
                               Vector4f{d_color_i[0], d_color_i[1], d_color_i[2], d_alpha_i},
                               scene.d_shape_groups[group_id].stroke_color,
                               d_translation);
            } else {
                d_sample_color(scene.shape_groups[group_id].fill_color_type,
                               scene.shape_groups[group_id].fill_color,
                               pt,
                               Vector4f{d_color_i[0], d_color_i[1], d_color_i[2], d_alpha_i},
                               scene.d_shape_groups[group_id].fill_color,
                               d_translation);
            }
            d_curr_color = d_prev_color;
            d_curr_alpha = d_prev_alpha;
        }
        if (d_background_color != nullptr) {
            d_background_color->x += d_curr_color.x;
            d_background_color->y += d_curr_color.y;
            d_background_color->z += d_curr_color.z;
            d_background_color->w += d_curr_alpha;
        }
    }
    return Vector4f{final_color[0], final_color[1], final_color[2], final_alpha};
}



DEVICE
bool is_inside(const SceneData &scene_data,
               int shape_group_id,
               const Vector2f &pt,
               EdgeQuery *edge_query) {
    const ShapeGroup &shape_group = scene_data.shape_groups[shape_group_id];
    // pt is in canvas space, transform it to shape's local space
    auto local_pt = xform_pt(shape_group.canvas_to_shape, pt);
    const auto &bvh_nodes = scene_data.shape_groups_bvh_nodes[shape_group_id];
    const AABB &bbox = bvh_nodes[2 * shape_group.num_shapes - 2].box;
    if (!inside(bbox, local_pt)) {
        return false;
    }
    auto winding_number = 0;
    // Traverse the shape group BVH
    constexpr auto max_bvh_stack_size = 64;
    int bvh_stack[max_bvh_stack_size];
    auto stack_size = 0;
    bvh_stack[stack_size++] = 2 * shape_group.num_shapes - 2;
    while (stack_size > 0) {
        const BVHNode &node = bvh_nodes[bvh_stack[--stack_size]];
        if (node.child1 < 0) {
            // leaf
            auto shape_id = node.child0;
            auto w = compute_winding_number(
                scene_data.shapes[shape_id], scene_data.path_bvhs[shape_id], local_pt);
            winding_number += w;
            if (edge_query != nullptr) {
                if (edge_query->shape_group_id == shape_group_id &&
                        edge_query->shape_id == shape_id) {
                    if ((shape_group.use_even_odd_rule && abs(w) % 2 == 1) ||
                        (!shape_group.use_even_odd_rule && w != 0)) {
                        edge_query->hit = true;
                    }
                }
            }
        } else {
            assert(node.child0 >= 0 && node.child1 >= 0);
            const AABB &b0 = bvh_nodes[node.child0].box;
            if (inside(b0, local_pt)) {
                bvh_stack[stack_size++] = node.child0;
            }
            const AABB &b1 = bvh_nodes[node.child1].box;
            if (inside(b1, local_pt)) {
                bvh_stack[stack_size++] = node.child1;
            }
            assert(stack_size <= max_bvh_stack_size);
        }
    }
    if (shape_group.use_even_odd_rule) {
        return abs(winding_number) % 2 == 1;
    } else {
        return winding_number != 0;
    }
}

DEVICE void accumulate_boundary_gradient(const Shape &shape,
                                         float contrib,
                                         float t,
                                         const Vector2f &normal,
                                         const BoundaryData &boundary_data,
                                         Shape &d_shape,
                                         const Matrix3x3f &shape_to_canvas,
                                         const Vector2f &local_boundary_pt,
                                         Matrix3x3f &d_shape_to_canvas) {
    assert(isfinite(contrib));
    assert(isfinite(normal));
    // According to Reynold transport theorem,
    // the Jacobian of the boundary integral is dot(velocity, normal),
    // where the velocity depends on the variable being differentiated with.
    if (boundary_data.is_stroke) {
        auto has_path_thickness = false;
        if (shape.type == ShapeType::Path) {
            const Path &path = *(const Path *)shape.ptr;
            has_path_thickness = path.thickness != nullptr;
        }
        // differentiate stroke width: velocity is the same as normal
        if (has_path_thickness) {
            Path *d_p = (Path*)d_shape.ptr;
            auto base_point_id = boundary_data.path.base_point_id;
            auto point_id = boundary_data.path.point_id;
            auto t = boundary_data.path.t;
            const Path &path = *(const Path *)shape.ptr;
            if (path.num_control_points[base_point_id] == 0) {
                // Straight line
                auto i0 = point_id;
                auto i1 = (point_id + 1) % path.num_points;
                // r = r0 + t * (r1 - r0)
                atomic_add(&d_p->thickness[i0], (1 - t) * contrib);
                atomic_add(&d_p->thickness[i1], (    t) * contrib);
            } else if (path.num_control_points[base_point_id] == 1) {
                // Quadratic Bezier curve
                auto i0 = point_id;
                auto i1 = point_id + 1;
                auto i2 = (point_id + 2) % path.num_points;
                // r = (1-t)^2r0 + 2(1-t)t r1 + t^2 r2
                atomic_add(&d_p->thickness[i0], square(1 - t) * contrib);
                atomic_add(&d_p->thickness[i1], (2*(1-t)*t) * contrib);
                atomic_add(&d_p->thickness[i2], (t*t) * contrib);
            } else if (path.num_control_points[base_point_id] == 2) {
                auto i0 = point_id;
                auto i1 = point_id + 1;
                auto i2 = point_id + 2;
                auto i3 = (point_id + 3) % path.num_points;
                // r = (1-t)^3r0 + 3*(1-t)^2tr1 + 3*(1-t)t^2r2 + t^3r3
                atomic_add(&d_p->thickness[i0], cubic(1 - t) * contrib);
                atomic_add(&d_p->thickness[i1], 3 * square(1 - t) * t * contrib);
                atomic_add(&d_p->thickness[i2], 3 * (1 - t) * t * t * contrib);
                atomic_add(&d_p->thickness[i3], t * t * t * contrib);
            } else {
                assert(false);
            }
        } else {
            atomic_add(&d_shape.stroke_width, contrib);
        }
    }
    switch (shape.type) {
        case ShapeType::Circle: {
            Circle *d_p = (Circle*)d_shape.ptr;
            // velocity for the center is (1, 0) for x and (0, 1) for y
            atomic_add(&d_p->center[0], normal * contrib);
            // velocity for the radius is the same as the normal
            atomic_add(&d_p->radius, contrib);
            break;
        } case ShapeType::Ellipse: {
            Ellipse *d_p = (Ellipse*)d_shape.ptr;
            // velocity for the center is (1, 0) for x and (0, 1) for y
            atomic_add(&d_p->center[0], normal * contrib);
            // velocity for the radius:
            // x = center.x + r.x * cos(2pi * t)
            // y = center.y + r.y * sin(2pi * t)
            // for r.x: (cos(2pi * t), 0)
            // for r.y: (0, sin(2pi * t))
            atomic_add(&d_p->radius.x, cos(2 * float(M_PI) * t) * normal.x * contrib);
            atomic_add(&d_p->radius.y, sin(2 * float(M_PI) * t) * normal.y * contrib);
            break;
        } case ShapeType::Path: {
            Path *d_p = (Path*)d_shape.ptr;
            auto base_point_id = boundary_data.path.base_point_id;
            auto point_id = boundary_data.path.point_id;
            auto t = boundary_data.path.t;
            const Path &path = *(const Path *)shape.ptr;
            if (path.num_control_points[base_point_id] == 0) {
                // Straight line
                auto i0 = point_id;
                auto i1 = (point_id + 1) % path.num_points;
                // pt = p0 + t * (p1 - p0)
                // velocity for p0.x: (1 - t,     0)
                //              p0.y: (    0, 1 - t)
                //              p1.x: (    t,     0)
                //              p1.y: (    0,     t)
                atomic_add(&d_p->points[2 * i0 + 0], (1 - t) * normal.x * contrib);
                atomic_add(&d_p->points[2 * i0 + 1], (1 - t) * normal.y * contrib);
                atomic_add(&d_p->points[2 * i1 + 0], (    t) * normal.x * contrib);
                atomic_add(&d_p->points[2 * i1 + 1], (    t) * normal.y * contrib);
            } else if (path.num_control_points[base_point_id] == 1) {
                // Quadratic Bezier curve
                auto i0 = point_id;
                auto i1 = point_id + 1;
                auto i2 = (point_id + 2) % path.num_points;
                // pt = (1-t)^2p0 + 2(1-t)t p1 + t^2 p2
                // velocity for p0.x: ((1-t)^2,       0)
                //              p0.y: (      0, (1-t)^2)
                //              p1.x: (2(1-t)t,       0)
                //              p1.y: (      0, 2(1-t)t)
                //              p1.x: (    t^2,       0)
                //              p1.y: (      0,     t^2)
                atomic_add(&d_p->points[2 * i0 + 0], square(1 - t) * normal.x * contrib);
                atomic_add(&d_p->points[2 * i0 + 1], square(1 - t) * normal.y * contrib);
                atomic_add(&d_p->points[2 * i1 + 0], (2*(1-t)*t) * normal.x * contrib);
                atomic_add(&d_p->points[2 * i1 + 1], (2*(1-t)*t) * normal.y * contrib);
                atomic_add(&d_p->points[2 * i2 + 0], (t*t) * normal.x * contrib);
                atomic_add(&d_p->points[2 * i2 + 1], (t*t) * normal.y * contrib);
            } else if (path.num_control_points[base_point_id] == 2) {
                auto i0 = point_id;
                auto i1 = point_id + 1;
                auto i2 = point_id + 2;
                auto i3 = (point_id + 3) % path.num_points;
                // pt = (1-t)^3p0 + 3*(1-t)^2tp1 + 3*(1-t)t^2p2 + t^3p3
                // velocity for p0.x: (   (1-t)^3,          0)
                //              p0.y: (         0,    (1-t)^3)
                //              p1.x: (3*(1-t)^2t,          0)
                //              p1.y: (         0, 3*(1-t)^2t)
                //              p2.x: (3*(1-t)t^2,          0)
                //              p2.y: (         0, 3*(1-t)t^2)
                //              p2.x: (       t^3,          0)
                //              p2.y: (         0,        t^3)
                atomic_add(&d_p->points[2 * i0 + 0], cubic(1 - t) * normal.x * contrib);
                atomic_add(&d_p->points[2 * i0 + 1], cubic(1 - t) * normal.y * contrib);
                atomic_add(&d_p->points[2 * i1 + 0], 3 * square(1 - t) * t * normal.x * contrib);
                atomic_add(&d_p->points[2 * i1 + 1], 3 * square(1 - t) * t * normal.y * contrib);
                atomic_add(&d_p->points[2 * i2 + 0], 3 * (1 - t) * t * t * normal.x * contrib);
                atomic_add(&d_p->points[2 * i2 + 1], 3 * (1 - t) * t * t * normal.y * contrib);
                atomic_add(&d_p->points[2 * i3 + 0], t * t * t * normal.x * contrib);
                atomic_add(&d_p->points[2 * i3 + 1], t * t * t * normal.y * contrib);
            } else {
                assert(false);
            }
            break;
        } case ShapeType::Rect: {
            Rect *d_p = (Rect*)d_shape.ptr;
            // The velocity depends on the position of the boundary
            if (normal == Vector2f{-1, 0}) {
                // left
                // velocity for p_min is (1, 0) for x and (0, 0) for y
                atomic_add(&d_p->p_min.x, -contrib);
            } else if (normal == Vector2f{1, 0}) {
                // right
                // velocity for p_max is (1, 0) for x and (0, 0) for y
                atomic_add(&d_p->p_max.x, contrib);
            } else if (normal == Vector2f{0, -1}) {
                // top
                // velocity for p_min is (0, 0) for x and (0, 1) for y
                atomic_add(&d_p->p_min.y, -contrib);
            } else if (normal == Vector2f{0, 1}) {
                // bottom
                // velocity for p_max is (0, 0) for x and (0, 1) for y
                atomic_add(&d_p->p_max.y, contrib);
            } else {
                // incorrect normal assignment?
                assert(false);
            }
            break;
        } default: {
            assert(false);
            break;
        }
    }
    // for shape_to_canvas we have the following relationship:
    // boundary_pt = xform_pt(shape_to_canvas, local_pt)
    // the velocity is the derivative of boundary_pt with respect to shape_to_canvas
    // we can use reverse-mode AD to compute the dot product of the velocity and the Jacobian
    // by passing the normal in d_xform_pt
    auto d_shape_to_canvas_ = Matrix3x3f();
    auto d_local_boundary_pt = Vector2f{0, 0};
    d_xform_pt(shape_to_canvas,
               local_boundary_pt,
               normal * contrib,
               d_shape_to_canvas_,
               d_local_boundary_pt);
    atomic_add(&d_shape_to_canvas(0, 0), d_shape_to_canvas_);
}

DEVICE
float sample_distance(const SceneData &scene,
                      const Vector2f &screen_pt,
                      float weight,
                      const float *d_dist,
                      float *d_translation) {
    // screen_pt is in screen space ([0, 1), [0, 1)),
    // need to transform to canvas space
    auto pt = screen_pt;
    pt.x *= scene.canvas_width;
    pt.y *= scene.canvas_height;
    // for each shape
    auto min_group_id = -1;
    auto min_distance = 0.f;
    auto min_shape_id = -1;
    auto closest_pt = Vector2f{0, 0};
    auto min_path_info = ClosestPointPathInfo{-1, -1, 0};
    for (int group_id = scene.num_shape_groups - 1; group_id >= 0; group_id--) {
        auto s = -1;
        auto p = Vector2f{0, 0};
        ClosestPointPathInfo local_path_info;
        auto d = infinity<float>();
        if (compute_distance(scene, group_id, pt, infinity<float>(), &s, &p, &local_path_info, &d)) {
            if (min_group_id == -1 || d < min_distance) {
                min_distance = d;
                min_group_id = group_id;
                min_shape_id = s;
                closest_pt = p;
                min_path_info = local_path_info;
            }
        }
    }
    if (min_group_id == -1) {
        return min_distance;
    }
    min_distance *= weight;
    auto inside = false;
    const ShapeGroup &shape_group = scene.shape_groups[min_group_id];
    if (shape_group.fill_color != nullptr) {
        inside = is_inside(scene,
                           min_group_id,
                           pt,
                           nullptr);
        if (inside) {
            min_distance = -min_distance;
        }
    }
    assert((min_group_id >= 0 && min_shape_id >= 0) || scene.num_shape_groups == 0);
    if (d_dist != nullptr) {
        auto d_abs_dist = inside ? -(*d_dist) : (*d_dist);
        const ShapeGroup &shape_group = scene.shape_groups[min_group_id];
        const Shape &shape = scene.shapes[min_shape_id];
        ShapeGroup &d_shape_group = scene.d_shape_groups[min_group_id];
        Shape &d_shape = scene.d_shapes[min_shape_id];
        d_compute_distance(shape_group.canvas_to_shape,
                           shape_group.shape_to_canvas,
                           shape,
                           pt,
                           closest_pt,
                           min_path_info,
                           d_abs_dist,
                           d_shape_group.shape_to_canvas,
                           d_shape,
                           d_translation);
    }
    return min_distance;
}

// Gather d_color from d_image inside the filter kernel, normalize by
// weight_image.
DEVICE
Vector4f gather_d_color(const Filter &filter,
                        const float *d_color_image,
                        const float *weight_image,
                        int width,
                        int height,
                        const Vector2f &pt) {
    auto x = int(pt.x);
    auto y = int(pt.y);
    auto radius = filter.radius;
    assert(radius > 0);
    auto ri = (int)ceil(radius);
    auto d_color = Vector4f{0, 0, 0, 0};
    for (int dy = -ri; dy <= ri; dy++) {
        for (int dx = -ri; dx <= ri; dx++) {
            auto xx = x + dx;
            auto yy = y + dy;
            if (xx >= 0 && xx < width && yy >= 0 && yy < height) {
                auto xc = xx + 0.5f;
                auto yc = yy + 0.5f;
                auto filter_weight =
                    compute_filter_weight(filter, xc - pt.x, yc - pt.y);
                // pixel = \sum weight * color / \sum weight
                auto weight_sum = weight_image[yy * width + xx];
                if (weight_sum > 0) {
                    d_color += (filter_weight / weight_sum) * Vector4f{
                        d_color_image[4 * (yy * width + xx) + 0],
                        d_color_image[4 * (yy * width + xx) + 1],
                        d_color_image[4 * (yy * width + xx) + 2],
                        d_color_image[4 * (yy * width + xx) + 3],
                    };
                }
            }
        }
    }
    return d_color;
}

DEVICE
float smoothstep(float d) {
    auto t = clamp((d + 1.f) / 2.f, 0.f, 1.f);
    return t * t * (3 - 2 * t);
}

DEVICE
float d_smoothstep(float d, float d_ret) {
    if (d < -1.f || d > 1.f) {
        return 0.f;
    }
    auto t = (d + 1.f) / 2.f;
    // ret = t * t * (3 - 2 * t)
    //     = 3 * t * t - 2 * t * t * t
    auto d_t = d_ret * (6 * t - 6 * t * t);
    return d_t / 2.f;
}

DEVICE
Vector4f sample_color_prefiltered(const SceneData &scene,
                                  const Vector4f *background_color,
                                  const Vector2f &screen_pt,
                                  const Vector4f *d_color,
                                  Vector4f *d_background_color,
                                  float *d_translation) {
    // screen_pt is in screen space ([0, 1), [0, 1)),
    // need to transform to canvas space
    auto pt = screen_pt;
    pt.x *= scene.canvas_width;
    pt.y *= scene.canvas_height;
    constexpr auto max_hit_shapes = 64;
    constexpr auto max_bvh_stack_size = 64;
    PrefilterFragment fragments[max_hit_shapes];
    int bvh_stack[max_bvh_stack_size];
    auto stack_size = 0;
    auto num_fragments = 0;
    bvh_stack[stack_size++] = 2 * scene.num_shape_groups - 2;
    while (stack_size > 0) {
        const BVHNode &node = scene.bvh_nodes[bvh_stack[--stack_size]];
        if (node.child1 < 0) {
            // leaf
            auto group_id = node.child0;
            const ShapeGroup &shape_group = scene.shape_groups[group_id];
            if (shape_group.stroke_color != nullptr) {
                auto min_shape_id = -1;
                auto closest_pt = Vector2f{0, 0};
                auto local_path_info = ClosestPointPathInfo{-1, -1, 0};
                auto d = infinity<float>();
                compute_distance(scene, group_id, pt, infinity<float>(),
                                 &min_shape_id, &closest_pt, &local_path_info, &d);
                assert(min_shape_id != -1);
                const auto &shape = scene.shapes[min_shape_id];
                auto w = smoothstep(fabs(d) + shape.stroke_width) -
                         smoothstep(fabs(d) - shape.stroke_width);
                if (w > 0) {
                    auto color_alpha = sample_color(shape_group.stroke_color_type,
                                                    shape_group.stroke_color,
                                                    pt);
                    color_alpha[3] *= w;

                    PrefilterFragment f;
                    f.color = Vector3f{color_alpha[0], color_alpha[1], color_alpha[2]};
                    f.alpha = color_alpha[3];
                    f.group_id = group_id;
                    f.shape_id = min_shape_id;
                    f.distance = d;
                    f.closest_pt = closest_pt;
                    f.is_stroke = true;
                    f.path_info = local_path_info;
                    f.within_distance = true;
                    assert(num_fragments < max_hit_shapes);
                    fragments[num_fragments++] = f;
                }
            }
            if (shape_group.fill_color != nullptr) {
                auto min_shape_id = -1;
                auto closest_pt = Vector2f{0, 0};
                auto local_path_info = ClosestPointPathInfo{-1, -1, 0};
                auto d = infinity<float>();
                auto found = compute_distance(scene,
                                              group_id,
                                              pt,
                                              1.f,
                                              &min_shape_id,
                                              &closest_pt,
                                              &local_path_info,
                                              &d);
                auto inside = is_inside(scene, group_id, pt, nullptr);
                if (found || inside) {
                    if (!inside) {
                        d = -d;
                    }
                    auto w = smoothstep(d);
                    if (w > 0) {
                        auto color_alpha = sample_color(shape_group.fill_color_type,
                                                        shape_group.fill_color,
                                                        pt);
                        color_alpha[3] *= w;

                        PrefilterFragment f;
                        f.color = Vector3f{color_alpha[0], color_alpha[1], color_alpha[2]};
                        f.alpha = color_alpha[3];
                        f.group_id = group_id;
                        f.shape_id = min_shape_id;
                        f.distance = d;
                        f.closest_pt = closest_pt;
                        f.is_stroke = false;
                        f.path_info = local_path_info;
                        f.within_distance = found;
                        assert(num_fragments < max_hit_shapes);
                        fragments[num_fragments++] = f;
                    }
                }
            }
        } else {
            assert(node.child0 >= 0 && node.child1 >= 0);
            const AABB &b0 = scene.bvh_nodes[node.child0].box;
            if (inside(b0, pt, scene.bvh_nodes[node.child0].max_radius)) {
                bvh_stack[stack_size++] = node.child0;
            }
            const AABB &b1 = scene.bvh_nodes[node.child1].box;
            if (inside(b1, pt, scene.bvh_nodes[node.child1].max_radius)) {
                bvh_stack[stack_size++] = node.child1;
            }
            assert(stack_size <= max_bvh_stack_size);
        }
    }
    if (num_fragments <= 0) {
        if (background_color != nullptr) {
            if (d_background_color != nullptr) {
                *d_background_color = *d_color;
            }
            return *background_color;
        }
        return Vector4f{0, 0, 0, 0};
    }
    // Sort the fragments from back to front (i.e. increasing order of group id)
    // https://github.com/frigaut/yorick-imutil/blob/master/insort.c#L37
    for (int i = 1; i < num_fragments; i++) {
        auto j = i;
        auto temp = fragments[j];
        while (j > 0 && fragments[j - 1].group_id > temp.group_id) {
            fragments[j] = fragments[j - 1];
            j--;
        }
        fragments[j] = temp;
    }
    // Blend the color
    Vector3f accum_color[max_hit_shapes];
    float accum_alpha[max_hit_shapes];
    auto first_alpha = 0.f;
    auto first_color = Vector3f{0, 0, 0};
    if (background_color != nullptr) {
        first_alpha = background_color->w;
        first_color = Vector3f{background_color->x,
                               background_color->y,
                               background_color->z};
    }
    for (int i = 0; i < num_fragments; i++) {
        const PrefilterFragment &fragment = fragments[i];
        auto new_color = fragment.color;
        auto new_alpha = fragment.alpha;
        auto prev_alpha = i > 0 ? accum_alpha[i - 1] : first_alpha;
        auto prev_color = i > 0 ? accum_color[i - 1] : first_color;
        // prev_color is alpha premultiplied, don't need to multiply with
        // prev_alpha
        accum_color[i] = prev_color * (1 - new_alpha) + new_alpha * new_color;
        accum_alpha[i] = prev_alpha * (1 - new_alpha) + new_alpha;
    }
    auto final_color = accum_color[num_fragments - 1];
    auto final_alpha = accum_alpha[num_fragments - 1];
    if (final_alpha > 1e-6f) {
        final_color /= final_alpha;
    }
    assert(isfinite(final_color));
    assert(isfinite(final_alpha));
    if (d_color != nullptr) {
        // Backward pass
        auto d_final_color = Vector3f{(*d_color)[0], (*d_color)[1], (*d_color)[2]};
        auto d_final_alpha = (*d_color)[3];
        auto d_curr_color = d_final_color;
        auto d_curr_alpha = d_final_alpha;
        if (final_alpha > 1e-6f) {
            // final_color = curr_color / final_alpha
            d_curr_color = d_final_color / final_alpha;
            d_curr_alpha -= sum(d_final_color * final_color) / final_alpha;
        }
        assert(isfinite(*d_color));
        assert(isfinite(d_curr_color));
        assert(isfinite(d_curr_alpha));
        for (int i = num_fragments - 1; i >= 0; i--) {
            // color[n] = prev_color * (1 - new_alpha) + new_alpha * new_color;
            // alpha[n] = prev_alpha * (1 - new_alpha) + new_alpha;
            auto prev_alpha = i > 0 ? accum_alpha[i - 1] : first_alpha;
            auto prev_color = i > 0 ? accum_color[i - 1] : first_color;
            auto d_prev_alpha = d_curr_alpha * (1.f - fragments[i].alpha);
            auto d_alpha_i = d_curr_alpha * (1.f - prev_alpha);
            d_alpha_i += sum(d_curr_color * (fragments[i].color - prev_color));
            auto d_prev_color = d_curr_color * (1 - fragments[i].alpha);
            auto d_color_i = d_curr_color * fragments[i].alpha;
            auto group_id = fragments[i].group_id;
            if (fragments[i].is_stroke) {
                const auto &shape = scene.shapes[fragments[i].shape_id];
                auto d = fragments[i].distance;
                auto abs_d_plus_width = fabs(d) + shape.stroke_width;
                auto abs_d_minus_width = fabs(d) - shape.stroke_width;
                auto w = smoothstep(abs_d_plus_width) -
                         smoothstep(abs_d_minus_width);
                if (w != 0) {
                    auto d_w = w > 0 ? (fragments[i].alpha / w) * d_alpha_i : 0.f;
                    d_alpha_i *= w;

                    // Backprop to color
                    d_sample_color(scene.shape_groups[group_id].stroke_color_type,
                                   scene.shape_groups[group_id].stroke_color,
                                   pt,
                                   Vector4f{d_color_i[0], d_color_i[1], d_color_i[2], d_alpha_i},
                                   scene.d_shape_groups[group_id].stroke_color,
                                   d_translation);

                    auto d_abs_d_plus_width = d_smoothstep(abs_d_plus_width, d_w);
                    auto d_abs_d_minus_width = -d_smoothstep(abs_d_minus_width, d_w);

                    auto d_d = d_abs_d_plus_width + d_abs_d_minus_width;
                    if (d < 0) {
                        d_d = -d_d;
                    }
                    auto d_stroke_width = d_abs_d_plus_width - d_abs_d_minus_width;

                    const auto &shape_group = scene.shape_groups[group_id];
                    ShapeGroup &d_shape_group = scene.d_shape_groups[group_id];
                    Shape &d_shape = scene.d_shapes[fragments[i].shape_id];
                    if (fabs(d_d) > 1e-10f) {
                        d_compute_distance(shape_group.canvas_to_shape,
                                           shape_group.shape_to_canvas,
                                           shape,
                                           pt,
                                           fragments[i].closest_pt,
                                           fragments[i].path_info,
                                           d_d,
                                           d_shape_group.shape_to_canvas,
                                           d_shape,
                                           d_translation);
                    }
                    atomic_add(&d_shape.stroke_width, d_stroke_width);
                }
            } else {
                const auto &shape = scene.shapes[fragments[i].shape_id];
                auto d = fragments[i].distance;
                auto w = smoothstep(d);
                if (w != 0) {
                    // color_alpha[3] = color_alpha[3] * w;
                    auto d_w = w > 0 ? (fragments[i].alpha / w) * d_alpha_i : 0.f;
                    d_alpha_i *= w;

                    d_sample_color(scene.shape_groups[group_id].fill_color_type,
                                   scene.shape_groups[group_id].fill_color,
                                   pt,
                                   Vector4f{d_color_i[0], d_color_i[1], d_color_i[2], d_alpha_i},
                                   scene.d_shape_groups[group_id].fill_color,
                                   d_translation);

                    // w = smoothstep(d)
                    auto d_d = d_smoothstep(d, d_w);
                    if (d < 0) {
                        d_d = -d_d;
                    }

                    const auto &shape_group = scene.shape_groups[group_id];
                    ShapeGroup &d_shape_group = scene.d_shape_groups[group_id];
                    Shape &d_shape = scene.d_shapes[fragments[i].shape_id];
                    if (fabs(d_d) > 1e-10f && fragments[i].within_distance) {
                        d_compute_distance(shape_group.canvas_to_shape,
                                           shape_group.shape_to_canvas,
                                           shape,
                                           pt,
                                           fragments[i].closest_pt,
                                           fragments[i].path_info,
                                           d_d,
                                           d_shape_group.shape_to_canvas,
                                           d_shape,
                                           d_translation);
                    }
                }
            }
            d_curr_color = d_prev_color;
            d_curr_alpha = d_prev_alpha;
        }
        if (d_background_color != nullptr) {
            d_background_color->x += d_curr_color.x;
            d_background_color->y += d_curr_color.y;
            d_background_color->z += d_curr_color.z;
            d_background_color->w += d_curr_alpha;
        }
    }
    return Vector4f{final_color[0], final_color[1], final_color[2], final_alpha};
}

// We use a "mega kernel" for rendering
