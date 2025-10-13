#include "render.h"
#include "render_support.h"

#include "diffvg.h"
#include "aabb.h"
#include "shape.h"
#include "sample_boundary.h"
#include "atomic.h"
#include "cdf.h"
#include "compute_distance.h"
#include "cuda_utils.h"
#include "edge_query.h"
#include "filter.h"
#include "matrix.h"
#include "parallel.h"
#include "pcg.h"
#include "scene.h"
#include "vector.h"
#include "winding_number.h"
#include "within_distance.h"

#include <cassert>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <cstdint>
#include <cstdlib>


static inline bool diffvg_disable_gpu_sort() {
    const char *env = std::getenv("DIFFVG_DISABLE_GPU_SORT");
    if (!env) return false;
    // Treat any non-empty, non-"0" value as true
    return env[0] != '\0' && !(env[0] == '0' && env[1] == '\0');
}

#ifdef COMPILE_WITH_CUDA
// Implemented in scene.cpp (compiled as a CUDA TU).
#ifdef __cplusplus
extern "C" void diffvg_gpu_sort_by_key_uint_uint(uint32_t* keys, int* vals, size_t n);
#else
extern void diffvg_gpu_sort_by_key_uint_uint(uint32_t* keys, int* vals, size_t n);
#endif
#endif

struct Command {
    int shape_group_id;
    int shape_id;
    int point_id; // Only used by path
};

struct weight_kernel {
    DEVICE void operator()(int idx) {
        auto rng_state = init_pcg32(idx, seed);
        // height * width * num_samples_y * num_samples_x
        auto sx = idx % num_samples_x;
        auto sy = (idx / num_samples_x) % num_samples_y;
        auto x = (idx / (num_samples_x * num_samples_y)) % width;
        auto y = (idx / (num_samples_x * num_samples_y * width));
        assert(y < height);
        auto rx = next_pcg32_float(&rng_state);
        auto ry = next_pcg32_float(&rng_state);
        if (use_prefiltering) {
            rx = ry = 0.5f;
        }
        auto pt = Vector2f{x + ((float)sx + rx) / num_samples_x,
                           y + ((float)sy + ry) / num_samples_y};
        auto radius = scene.filter->radius;
        assert(radius >= 0);
        auto ri = (int)ceil(radius);
        for (int dy = -ri; dy <= ri; dy++) {
            for (int dx = -ri; dx <= ri; dx++) {
                auto xx = x + dx;
                auto yy = y + dy;
                if (xx >= 0 && xx < width && yy >= 0 && yy < height) {
                    auto xc = xx + 0.5f;
                    auto yc = yy + 0.5f;
                    auto filter_weight = compute_filter_weight(*scene.filter,
                                                               xc - pt.x,
                                                               yc - pt.y);
                    atomic_add(weight_image[yy * width + xx], filter_weight);
                }
            }
        }
    }

    SceneData scene;
    float *weight_image;
    int width;
    int height;
    int num_samples_x;
    int num_samples_y;
    uint64_t seed;
    bool use_prefiltering;
};


struct render_kernel {
    DEVICE void operator()(int idx) {
        // height * width * num_samples_y * num_samples_x
        auto pt = Vector2f{0, 0};
        auto x = 0;
        auto y = 0;
        if (eval_positions == nullptr) {
            auto rng_state = init_pcg32(idx, seed);
            auto sx = idx % num_samples_x;
            auto sy = (idx / num_samples_x) % num_samples_y;
            x = (idx / (num_samples_x * num_samples_y)) % width;
            y = (idx / (num_samples_x * num_samples_y * width));
            assert(x < width && y < height);
            auto rx = next_pcg32_float(&rng_state);
            auto ry = next_pcg32_float(&rng_state);
            if (use_prefiltering) {
                rx = ry = 0.5f;
            }
            pt = Vector2f{x + ((float)sx + rx) / num_samples_x,
                          y + ((float)sy + ry) / num_samples_y};
        } else {
            pt = Vector2f{eval_positions[2 * idx],
                          eval_positions[2 * idx + 1]};
            x = int(pt.x);
            y = int(pt.y);
        }

        // normalize pt to [0, 1]
        auto npt = pt;
        npt.x /= width;
        npt.y /= height;
        auto num_samples = num_samples_x * num_samples_y;
        if (render_image != nullptr || d_render_image != nullptr) {
            Vector4f d_color = Vector4f{0, 0, 0, 0};
            if (d_render_image != nullptr) {
                // Gather d_color from d_render_image inside the filter kernel
                // normalize using weight_image
                d_color = gather_d_color(*scene.filter,
                                         d_render_image,
                                         weight_image,
                                         width,
                                         height,
                                         pt);
            }
            auto color = Vector4f{0, 0, 0, 0};
            if (use_prefiltering) {
                color = sample_color_prefiltered(scene,
                    background_image != nullptr ? (const Vector4f*)&background_image[4 * ((y * width) + x)] : nullptr,
                    npt,
                    d_render_image != nullptr ? &d_color : nullptr,
                    d_background_image != nullptr ? (Vector4f*)&d_background_image[4 * ((y * width) + x)] : nullptr,
                    d_translation != nullptr ? &d_translation[2 * (y * width + x)] : nullptr);
            } else {
                color = sample_color(scene,
                    background_image != nullptr ? (const Vector4f*)&background_image[4 * ((y * width) + x)] : nullptr,
                    npt,
                    d_render_image != nullptr ? &d_color : nullptr,
                    nullptr,
                    d_background_image != nullptr ? (Vector4f*)&d_background_image[4 * ((y * width) + x)] : nullptr,
                    d_translation != nullptr ? &d_translation[2 * (y * width + x)] : nullptr);
            }
            assert(isfinite(color));
            // Splat color onto render_image
            auto radius = scene.filter->radius;
            assert(radius >= 0);
            auto ri = (int)ceil(radius);
            for (int dy = -ri; dy <= ri; dy++) {
                for (int dx = -ri; dx <= ri; dx++) {
                    auto xx = x + dx;
                    auto yy = y + dy;
                    if (xx >= 0 && xx < width && yy >= 0 && yy < height &&
                            weight_image[yy * width + xx] > 0) {
                        auto weight_sum = weight_image[yy * width + xx];
                        auto xc = xx + 0.5f;
                        auto yc = yy + 0.5f;
                        auto filter_weight = compute_filter_weight(*scene.filter,
                                                                   xc - pt.x,
                                                                   yc - pt.y);
                        auto weighted_color = filter_weight * color / weight_sum;
                        if (render_image != nullptr) {
                            atomic_add(render_image[4 * (yy * width + xx) + 0],
                                       weighted_color[0]);
                            atomic_add(render_image[4 * (yy * width + xx) + 1],
                                       weighted_color[1]);
                            atomic_add(render_image[4 * (yy * width + xx) + 2],
                                       weighted_color[2]);
                            atomic_add(render_image[4 * (yy * width + xx) + 3],
                                       weighted_color[3]);
                        }
                        if (d_render_image != nullptr) {
                            // Backprop to filter_weight
                            // pixel = \sum weight * color / \sum weight
                            auto d_pixel = Vector4f{
                                d_render_image[4 * (yy * width + xx) + 0],
                                d_render_image[4 * (yy * width + xx) + 1],
                                d_render_image[4 * (yy * width + xx) + 2],
                                d_render_image[4 * (yy * width + xx) + 3],
                            };
                            auto d_weight =
                                (dot(d_pixel, color) * weight_sum -
                                 filter_weight * dot(d_pixel, color) * (weight_sum - filter_weight)) /
                                square(weight_sum);
                            d_compute_filter_weight(*scene.filter,
                                                    xc - pt.x,
                                                    yc - pt.y,
                                                    d_weight,
                                                    scene.d_filter);
                        }
                    }
                }
            }
        }
        if (sdf_image != nullptr || d_sdf_image != nullptr) {
            float d_dist = 0.f;
            if (d_sdf_image != nullptr) {
                if (eval_positions == nullptr) {
                    d_dist = d_sdf_image[y * width + x];
                } else {
                    d_dist = d_sdf_image[idx];
                }
            }
            auto weight = eval_positions == nullptr ? 1.f / num_samples : 1.f;
            auto dist = sample_distance(scene, npt, weight,
                d_sdf_image != nullptr ? &d_dist : nullptr, 
                d_translation != nullptr ? &d_translation[2 * (y * width + x)] : nullptr);
            if (sdf_image != nullptr) {
                if (eval_positions == nullptr) {
                    atomic_add(sdf_image[y * width + x], dist);
                } else {
                    atomic_add(sdf_image[idx], dist);
                }
            }
        }
    }

    SceneData scene;
    float *background_image;
    float *render_image;
    float *weight_image;
    float *sdf_image;
    float *d_background_image;
    float *d_render_image;
    float *d_sdf_image;
    float *d_translation;
    int width;
    int height;
    int num_samples_x;
    int num_samples_y;
    uint64_t seed;
    bool use_prefiltering;
    float *eval_positions;
};

struct BoundarySample {
    Vector2f pt;
    Vector2f local_pt;
    Vector2f normal;
    int shape_group_id;
    int shape_id;
    float t;
    BoundaryData data;
    float pdf;
};

struct sample_boundary_kernel {
    DEVICE void operator()(int idx) {
        boundary_samples[idx].pt = Vector2f{0, 0};
        boundary_samples[idx].shape_id = -1;
        boundary_ids[idx] = idx;
        morton_codes[idx] = 0;

        auto rng_state = init_pcg32(idx, seed);
        auto u = next_pcg32_float(&rng_state);
        // Sample a shape
        auto sample_id = sample(scene.sample_shapes_cdf,
                                scene.num_total_shapes,
                                u);
        assert(sample_id >= 0 && sample_id < scene.num_total_shapes);
        auto shape_id = scene.sample_shape_id[sample_id];
        assert(shape_id >= 0 && shape_id < scene.num_shapes);
        auto shape_group_id = scene.sample_group_id[sample_id];
        assert(shape_group_id >= 0 && shape_group_id < scene.num_shape_groups);
        auto shape_pmf = scene.sample_shapes_pmf[shape_id];
        if (shape_pmf <= 0) {
            return;
        }
        // Sample a point on the boundary of the shape
        auto boundary_pdf = 0.f;
        auto normal = Vector2f{0, 0};
        auto t = next_pcg32_float(&rng_state);
        BoundaryData boundary_data;
        const ShapeGroup &shape_group = scene.shape_groups[shape_group_id];
        auto local_boundary_pt = sample_boundary(
            scene, shape_group_id, shape_id,
            t, normal, boundary_pdf, boundary_data);
        if (boundary_pdf <= 0) {
            return;
        }

        // local_boundary_pt & normal are in shape's local space,
        // transform them to canvas space
        auto boundary_pt = xform_pt(shape_group.shape_to_canvas, local_boundary_pt);
        normal = xform_normal(shape_group.canvas_to_shape, normal);
        // Normalize boundary_pt to [0, 1)
        boundary_pt.x /= scene.canvas_width;
        boundary_pt.y /= scene.canvas_height;

        boundary_samples[idx].pt = boundary_pt;
        boundary_samples[idx].local_pt = local_boundary_pt;
        boundary_samples[idx].normal = normal;
        boundary_samples[idx].shape_group_id = shape_group_id;
        boundary_samples[idx].shape_id = shape_id;
        boundary_samples[idx].t = t;
        boundary_samples[idx].data = boundary_data;
        boundary_samples[idx].pdf = shape_pmf * boundary_pdf;
        TVector2<uint32_t> p_i{boundary_pt.x * 1023, boundary_pt.y * 1023};
        morton_codes[idx] = (expand_bits(p_i.x) << 1u) |
                            (expand_bits(p_i.y) << 0u);
    }

    SceneData scene;
    uint64_t seed;
    BoundarySample *boundary_samples;
    int *boundary_ids;
    uint32_t *morton_codes;
};

struct render_edge_kernel {
    DEVICE void operator()(int idx) {
        auto bid = boundary_ids[idx];
        if (boundary_samples[bid].shape_id == -1) {
            return;
        }
        auto boundary_pt = boundary_samples[bid].pt;
        auto local_boundary_pt = boundary_samples[bid].local_pt;
        auto normal = boundary_samples[bid].normal;
        auto shape_group_id = boundary_samples[bid].shape_group_id;
        auto shape_id = boundary_samples[bid].shape_id;
        auto t = boundary_samples[bid].t;
        auto boundary_data = boundary_samples[bid].data;
        auto pdf = boundary_samples[bid].pdf;

        const ShapeGroup &shape_group = scene.shape_groups[shape_group_id];

        auto bx = int(boundary_pt.x * width);
        auto by = int(boundary_pt.y * height);
        if (bx < 0 || bx >= width || by < 0 || by >= height) {
            return;
        }

        // Sample the two sides of the boundary
        auto inside_query = EdgeQuery{shape_group_id, shape_id, false};
        auto outside_query = EdgeQuery{shape_group_id, shape_id, false};
        auto color_inside = sample_color(scene,
            background_image != nullptr ? (const Vector4f *)&background_image[4 * ((by * width) + bx)] : nullptr,
            boundary_pt - 1e-4f * normal,
            nullptr,
            &inside_query,
            nullptr,
            nullptr);
        auto color_outside = sample_color(scene,
            background_image != nullptr ? (const Vector4f *)&background_image[4 * ((by * width) + bx)] : nullptr,
            boundary_pt + 1e-4f * normal,
            nullptr,
            &outside_query,
            nullptr,
            nullptr);
        if (!inside_query.hit && !outside_query.hit) {
            // occluded
            return;
        }
        if (!inside_query.hit) {
            normal = -normal;
            swap_(inside_query, outside_query);
            swap_(color_inside, color_outside);
        }
        // Boundary point in screen space
        auto sboundary_pt = boundary_pt;
        sboundary_pt.x *= width;
        sboundary_pt.y *= height;
        auto d_color = gather_d_color(*scene.filter,
                                      d_render_image,
                                      weight_image,
                                      width,
                                      height,
                                      sboundary_pt);
        // Normalization factor
        d_color /= float(scene.canvas_width * scene.canvas_height);
        
        assert(isfinite(d_color));
        assert(isfinite(pdf) && pdf > 0);
        auto contrib = dot(color_inside - color_outside, d_color) / pdf;
        ShapeGroup &d_shape_group = scene.d_shape_groups[shape_group_id];
        accumulate_boundary_gradient(scene.shapes[shape_id],
            contrib, t, normal, boundary_data, scene.d_shapes[shape_id],
            shape_group.shape_to_canvas, local_boundary_pt, d_shape_group.shape_to_canvas);
        // Don't need to backprop to filter weights:
        // \int f'(x) g(x) dx doesn't contain discontinuities
        // if f is continuous, even if g is discontinuous
        if (d_translation != nullptr) {
            // According to Reynold transport theorem,
            // the Jacobian of the boundary integral is dot(velocity, normal)
            // The velocity of the object translating x is (1, 0)
            // The velocity of the object translating y is (0, 1)
            atomic_add(&d_translation[2 * (by * width + bx) + 0], normal.x * contrib);
            atomic_add(&d_translation[2 * (by * width + bx) + 1], normal.y * contrib);
        }
    }

    SceneData scene;
    const float *background_image;
    const BoundarySample *boundary_samples;
    const int *boundary_ids;
    float *weight_image;
    float *d_render_image;
    float *d_translation;
    int width;
    int height;
    int num_samples_x;
    int num_samples_y;
};

void render(std::shared_ptr<Scene> scene,
            ptr<float> background_image,
            ptr<float> render_image,
            ptr<float> render_sdf,
            int width,
            int height,
            int num_samples_x,
            int num_samples_y,
            uint64_t seed,
            ptr<float> d_background_image,
            ptr<float> d_render_image,
            ptr<float> d_render_sdf,
            ptr<float> d_translation,
            bool use_prefiltering,
            ptr<float> eval_positions,
            int num_eval_positions) {
#ifdef __NVCC__
    int old_device_id = -1;
    if (scene->use_gpu) {
        checkCuda(cudaGetDevice(&old_device_id));
        if (scene->gpu_index != -1) {
            checkCuda(cudaSetDevice(scene->gpu_index));
        }
    }
#endif
    parallel_init();

    float *weight_image = nullptr;
    // Allocate and zero the weight image
    if (scene->use_gpu) {
#ifdef __CUDACC__
        if (eval_positions.get() == nullptr) {
            checkCuda(cudaMallocManaged(&weight_image, width * height * sizeof(float)));
            cudaMemset(weight_image, 0, width * height * sizeof(float));
        }
#else
        assert(false);
#endif
    } else {
        if (eval_positions.get() == nullptr) {
            weight_image = (float*)malloc(width * height * sizeof(float));
            memset(weight_image, 0, width * height * sizeof(float));
        }
    }

    if (render_image.get() != nullptr || d_render_image.get() != nullptr ||
        render_sdf.get() != nullptr || d_render_sdf.get() != nullptr) {
        if (weight_image != nullptr) {
            parallel_for(weight_kernel{
                get_scene_data(*scene.get()),
                weight_image,
                width,
                height,
                num_samples_x,
                num_samples_y,
                seed
            }, width * height * num_samples_x * num_samples_y, scene->use_gpu);
        }

        auto num_samples = eval_positions.get() == nullptr ?
            width * height * num_samples_x * num_samples_y : num_eval_positions;
        parallel_for(render_kernel{
            get_scene_data(*scene.get()),
            background_image.get(),
            render_image.get(),
            weight_image,
            render_sdf.get(),
            d_background_image.get(),
            d_render_image.get(),
            d_render_sdf.get(),
            d_translation.get(),
            width,
            height,
            num_samples_x,
            num_samples_y,
            seed,
            use_prefiltering,
            eval_positions.get()
        }, num_samples, scene->use_gpu);
    }

    // Boundary sampling
    if (!use_prefiltering && d_render_image.get() != nullptr) {
        auto num_samples = width * height * num_samples_x * num_samples_y;
        BoundarySample *boundary_samples = nullptr;
        int *boundary_ids = nullptr; // for sorting
        uint32_t *morton_codes = nullptr; // for sorting
        // Allocate boundary samples
        if (scene->use_gpu) {
#ifdef __CUDACC__
            checkCuda(cudaMallocManaged(&boundary_samples,
                num_samples * sizeof(BoundarySample)));
            checkCuda(cudaMallocManaged(&boundary_ids,
                num_samples * sizeof(int)));
            checkCuda(cudaMallocManaged(&morton_codes,
                num_samples * sizeof(uint32_t)));
#else
            assert(false);
    #endif
        } else {
            boundary_samples = (BoundarySample*)malloc(
                num_samples * sizeof(BoundarySample));
            boundary_ids = (int*)malloc(
                num_samples * sizeof(int));
            morton_codes = (uint32_t*)malloc(
                num_samples * sizeof(uint32_t));
        }
        
        // Edge sampling
        // We sort the boundary samples for better thread coherency
        parallel_for(sample_boundary_kernel{
            get_scene_data(*scene.get()),
            seed,
            boundary_samples,
            boundary_ids,
            morton_codes
        }, num_samples, scene->use_gpu);
        if (scene->use_gpu) {
#ifdef COMPILE_WITH_CUDA
            if (!diffvg_disable_gpu_sort()) {
                diffvg_gpu_sort_by_key_uint_uint(morton_codes, boundary_ids, num_samples);
            }
#endif
        } else {
            // Don't need to sort for CPU, we are not using SIMD hardware anyway.
            // thrust::sort_by_key(thrust::host, morton_codes, morton_codes + num_samples, boundary_ids);
        }
        parallel_for(render_edge_kernel{
            get_scene_data(*scene.get()),
            background_image.get(),
            boundary_samples,
            boundary_ids,
            weight_image,
            d_render_image.get(),
            d_translation.get(),
            width,
            height,
            num_samples_x,
            num_samples_y
        }, num_samples, scene->use_gpu);
        if (scene->use_gpu) {
#ifdef __CUDACC__
            checkCuda(cudaFree(boundary_samples));
            checkCuda(cudaFree(boundary_ids));
            checkCuda(cudaFree(morton_codes));
#else
            assert(false);
#endif
        } else {
            free(boundary_samples);
            free(boundary_ids);
            free(morton_codes);
        }
    }

    // Clean up weight image
    if (scene->use_gpu) {
#ifdef __CUDACC__
        checkCuda(cudaFree(weight_image));
#else
        assert(false);
#endif
    } else {
        free(weight_image);
    }

    if (scene->use_gpu) {
        cuda_synchronize();
    }

    parallel_cleanup();
#ifdef __NVCC__
    if (old_device_id != -1) {
        checkCuda(cudaSetDevice(old_device_id));
    }
#endif
}
