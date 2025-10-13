#pragma once

#include <memory>

#include "ptr.h"

class Scene;

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
            int num_eval_positions);

