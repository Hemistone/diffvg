from __future__ import annotations

from typing import Optional

import torch

from .backend import current_api
from .device import get_device
from .output import OutputType

print_timing = False


def set_print_timing(enabled: bool) -> None:
    global print_timing
    print_timing = bool(enabled)


class RenderFunction:
    @staticmethod
    def serialize_scene(
        canvas_width,
        canvas_height,
        shapes,
        shape_groups,
        filter=None,
        output_type=OutputType.color,
        use_prefiltering: bool = False,
        eval_positions: torch.Tensor = torch.tensor([]),
        *,
        keep_on_device: Optional[bool] = None,
        device: Optional[torch.device | str] = None,
    ):
        api = current_api()
        if keep_on_device is None:
            keep_on_device = bool(getattr(api, "prefer_device_serialization", False))
        if keep_on_device and device is None:
            device = get_device()
        return api.serialize_scene(
            canvas_width,
            canvas_height,
            shapes,
            shape_groups,
            filter=filter,
            output_type=output_type,
            use_prefiltering=use_prefiltering,
            eval_positions=eval_positions,
            keep_on_device=keep_on_device,
            device=device,
        )

    @staticmethod
    def apply(
        width: int,
        height: int,
        num_samples_x: int,
        num_samples_y: int,
        seed: int,
        background_image,
        *scene_args,
    ):
        return current_api().apply(
            width,
            height,
            num_samples_x,
            num_samples_y,
            seed,
            background_image,
            *scene_args,
        )

    @staticmethod
    def render_grad(
        grad_img: torch.Tensor,
        width: int,
        height: int,
        num_samples_x: int,
        num_samples_y: int,
        seed: int,
        background_image,
        *scene_args,
    ):
        return current_api().render_grad(
            grad_img,
            width,
            height,
            num_samples_x,
            num_samples_y,
            seed,
            background_image,
            *scene_args,
        )


__all__ = ["RenderFunction", "OutputType", "set_print_timing", "print_timing"]
