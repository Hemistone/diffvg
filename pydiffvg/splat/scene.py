from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import diffvg
import torch

from ..backend import SplatConfig, get_backend_config
from ..device import get_device
from ..render_pytorch import OutputType
from .types import (
    PaintPayload,
    NonPathShapePayload,
    PathPayload,
    ScenePayload,
    ShapeGroupPayload,
    RenderRequest,
)


def _parse_paint(args: Iterable[object], start_idx: int) -> Tuple[PaintPayload, int]:
    idx = start_idx
    payload_type = args[idx]
    idx += 1
    if payload_type is None:
        return PaintPayload(None, ()), idx
    if payload_type == diffvg.ColorType.constant:
        color = args[idx]
        idx += 1
        return PaintPayload(payload_type, (color,)), idx
    if payload_type == diffvg.ColorType.linear_gradient:
        begin = args[idx]
        idx += 1
        end = args[idx]
        idx += 1
        offsets = args[idx]
        idx += 1
        stop_colors = args[idx]
        idx += 1
        return PaintPayload(payload_type, (begin, end, offsets, stop_colors)), idx
    if payload_type == diffvg.ColorType.radial_gradient:
        center = args[idx]
        idx += 1
        radius = args[idx]
        idx += 1
        offsets = args[idx]
        idx += 1
        stop_colors = args[idx]
        idx += 1
        return PaintPayload(payload_type, (center, radius, offsets, stop_colors)), idx
    raise ValueError(f"Unsupported paint payload type: {payload_type}")


def _deserialize_scene(args: Iterable[object]) -> ScenePayload:
    idx = 0
    canvas_width = int(args[idx]); idx += 1
    canvas_height = int(args[idx]); idx += 1
    num_shapes = int(args[idx]); idx += 1
    num_shape_groups = int(args[idx]); idx += 1
    output_type = args[idx]; idx += 1
    use_prefiltering = bool(args[idx]); idx += 1
    eval_positions = args[idx]; idx += 1

    paths: List[PathPayload] = []
    non_path_shapes: List[NonPathShapePayload] = []

    for shape_id in range(num_shapes):
        shape_type = args[idx]; idx += 1
        if shape_type == diffvg.ShapeType.path:
            num_control_points = args[idx]; idx += 1
            points = args[idx]; idx += 1
            thickness = args[idx]; idx += 1
            is_closed = bool(args[idx]); idx += 1
            use_distance_approx = bool(args[idx]); idx += 1
            stroke_width = args[idx]; idx += 1
            paths.append(
                PathPayload(
                    shape_id=shape_id,
                    num_control_points=num_control_points,
                    points=points,
                    thickness=thickness,
                    is_closed=is_closed,
                    use_distance_approx=use_distance_approx,
                    stroke_width=stroke_width,
                )
            )
            continue
        if shape_type == diffvg.ShapeType.circle:
            radius = args[idx]; idx += 1
            center = args[idx]; idx += 1
            shape_tensors = (radius, center)
        elif shape_type == diffvg.ShapeType.ellipse:
            radius = args[idx]; idx += 1
            center = args[idx]; idx += 1
            shape_tensors = (radius, center)
        elif shape_type == diffvg.ShapeType.rect:
            p_min = args[idx]; idx += 1
            p_max = args[idx]; idx += 1
            shape_tensors = (p_min, p_max)
        else:
            raise ValueError(f"Unsupported shape type in splat backend: {shape_type}")
        stroke_width = args[idx]; idx += 1
        non_path_shapes.append(
            NonPathShapePayload(
                shape_id=shape_id,
                shape_type=shape_type,
                tensors=shape_tensors,
                stroke_width=stroke_width,
            )
        )

    shape_groups: List[ShapeGroupPayload] = []
    for _ in range(num_shape_groups):
        shape_ids = args[idx]; idx += 1
        fill_payload, idx = _parse_paint(args, idx)
        stroke_payload, idx = _parse_paint(args, idx)
        use_even_odd_rule = bool(args[idx]); idx += 1
        shape_to_canvas = args[idx]; idx += 1
        shape_groups.append(
            ShapeGroupPayload(
                shape_ids=shape_ids,
                fill=fill_payload,
                stroke=stroke_payload,
                use_even_odd_rule=use_even_odd_rule,
                shape_to_canvas=shape_to_canvas,
            )
        )

    filter_type = args[idx]; idx += 1
    filter_radius = args[idx]

    return ScenePayload(
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        output_type=output_type,
        use_prefiltering=use_prefiltering,
        eval_positions=eval_positions,
        paths=paths,
        non_path_shapes=non_path_shapes,
        shape_groups=shape_groups,
        filter_type=filter_type,
        filter_radius=filter_radius,
    )


def _prepare_render_request(
    width: int,
    height: int,
    num_samples_x: int,
    num_samples_y: int,
    seed: int,
    background_image: Optional[torch.Tensor],
    args: Iterable[object],
) -> RenderRequest:
    config = get_backend_config("splat") or SplatConfig()
    scene = _deserialize_scene(args)
    device = get_device()
    generator: Optional[torch.Generator]
    if seed is None:
        generator = None
    else:
        if device.type == "cuda":
            generator = torch.Generator(device=device)
        else:
            generator = torch.Generator()
        generator.manual_seed(int(seed) & 0xFFFFFFFFFFFFFFFF)
    return RenderRequest(
        width=width,
        height=height,
        num_samples_x=num_samples_x,
        num_samples_y=num_samples_y,
        seed=seed,
        background_image=background_image,
        config=config,
        scene=scene,
        generator=generator,
    )


__all__ = [
    "_parse_paint",
    "_deserialize_scene",
    "_prepare_render_request",
]

