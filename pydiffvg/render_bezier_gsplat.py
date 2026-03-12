from __future__ import annotations

from typing import Optional

import torch

from .backend import BezierGsplatConfig, get_backend_config
from .openstroke import CompiledOpenStrokeScene, OpenStrokeUnsupported, compile_scene, render_compiled_scene
from .render_pytorch import OutputType


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
    keep_on_device: bool = True,
    device: Optional[torch.device | str] = None,
):
    del filter, keep_on_device
    compiled = compile_scene(
        canvas_width,
        canvas_height,
        shapes,
        shape_groups,
        output_type=output_type,
        use_prefiltering=use_prefiltering,
        eval_positions=eval_positions,
        device=device,
    )
    return (compiled,)


def _get_config() -> BezierGsplatConfig:
    config = get_backend_config("bezier_gsplat")
    if isinstance(config, BezierGsplatConfig):
        return config
    return BezierGsplatConfig()


def _resolve_scene(args: tuple[object, ...]) -> CompiledOpenStrokeScene:
    if len(args) != 1 or not isinstance(args[0], CompiledOpenStrokeScene):
        raise OpenStrokeUnsupported(
            "bezier_gsplat expects a compiled open-stroke scene; call serialize_scene() again after topology changes"
        )
    return args[0]


def apply(
    width: int,
    height: int,
    num_samples_x: int,
    num_samples_y: int,
    seed: int,
    background_image: Optional[torch.Tensor],
    *args: object,
) -> torch.Tensor:
    del num_samples_x, num_samples_y, seed
    scene = _resolve_scene(tuple(args))
    return render_compiled_scene(
        scene,
        width=width,
        height=height,
        config=_get_config(),
        background_image=background_image,
    )


def render_grad(
    grad_img: torch.Tensor,
    width: int,
    height: int,
    num_samples_x: int,
    num_samples_y: int,
    seed: int,
    background_image: Optional[torch.Tensor],
    *args: object,
):
    del grad_img, width, height, num_samples_x, num_samples_y, seed, background_image, args
    raise RuntimeError("bezier_gsplat relies on standard autograd through apply(); render_grad is not used in the maintained path")


__all__ = ["serialize_scene", "apply", "render_grad"]
