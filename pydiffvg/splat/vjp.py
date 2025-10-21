from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import diffvg
import torch

from .types import GradSlot, ScenePayload, _SplatUnsupported


def _align_grad_devices(
    inputs: Iterable[object], grads: Iterable[Optional[torch.Tensor]]
) -> Tuple[Optional[torch.Tensor], ...]:
    aligned: List[Optional[torch.Tensor]] = []
    for inp, grad in zip(inputs, grads):
        if grad is None or not isinstance(grad, torch.Tensor):
            aligned.append(grad)
            continue
        if isinstance(inp, torch.Tensor):
            target_device = inp.device
            target_dtype = inp.dtype
            if grad.device != target_device or grad.dtype != target_dtype:
                grad = grad.to(device=target_device, dtype=target_dtype)
        aligned.append(grad)
    return tuple(aligned)


def _enable_gradient_args(args: Iterable[object]) -> Tuple[Tuple[object, ...], List[GradSlot]]:
    args_list = list(args)
    grad_slots: List[GradSlot] = []
    idx = 0

    if len(args_list) < 7:
        raise _SplatUnsupported("serialized scene args truncated")
    idx += 1  # canvas_width
    idx += 1  # canvas_height
    num_shapes = int(args_list[idx])
    idx += 1
    num_shape_groups = int(args_list[idx])
    idx += 1
    idx += 1  # output_type
    idx += 1  # use_prefiltering
    idx += 1  # eval_positions

    for _shape_id in range(num_shapes):
        shape_type = args_list[idx]
        idx += 1
        if shape_type != diffvg.ShapeType.path:
            raise _SplatUnsupported("non-path shapes are not handled yet")
        idx += 1  # num_control_points
        points_idx = idx
        points = args_list[idx]
        if not isinstance(points, torch.Tensor):
            raise _SplatUnsupported("expected tensor points for path")
        if not points.requires_grad:
            points.requires_grad_(True)
        args_list[points_idx] = points
        grad_slots.append(GradSlot(arg_index=points_idx, tensor=points))
        idx += 1
        thickness = args_list[idx]
        idx += 1
        if thickness is not None:
            raise _SplatUnsupported("per-point thickness is not supported yet")
        idx += 1  # is_closed
        idx += 1  # use_distance_approx
        stroke_idx = idx
        stroke_width = args_list[idx]
        if isinstance(stroke_width, torch.Tensor):
            if not stroke_width.requires_grad:
                stroke_width.requires_grad_(True)
            args_list[stroke_idx] = stroke_width
            grad_slots.append(GradSlot(arg_index=stroke_idx, tensor=stroke_width))
        idx += 1

    for _group_id in range(num_shape_groups):
        idx += 1  # shape_ids
        fill_color_type = args_list[idx]
        idx += 1
        if fill_color_type is not None:
            if fill_color_type != diffvg.ColorType.constant:
                raise _SplatUnsupported("only constant fill colors are supported")
            color_idx = idx
            fill_color = args_list[idx]
            if not isinstance(fill_color, torch.Tensor):
                raise _SplatUnsupported("fill color tensor expected")
            if not fill_color.requires_grad:
                fill_color.requires_grad_(True)
            args_list[color_idx] = fill_color
            grad_slots.append(GradSlot(arg_index=color_idx, tensor=fill_color))
            idx += 1
        stroke_color_type = args_list[idx]
        idx += 1
        if stroke_color_type != diffvg.ColorType.constant:
            if stroke_color_type is not None:
                raise _SplatUnsupported("only constant stroke colors supported")
            continue
        color_idx = idx
        stroke_color = args_list[idx]
        if not isinstance(stroke_color, torch.Tensor):
            raise _SplatUnsupported("stroke color tensor expected")
        if not stroke_color.requires_grad:
            stroke_color.requires_grad_(True)
        args_list[color_idx] = stroke_color
        grad_slots.append(GradSlot(arg_index=color_idx, tensor=stroke_color))
        idx += 1
        idx += 1  # use_even_odd_rule
        idx += 1  # shape_to_canvas

    idx += 1  # filter_type
    idx += 1  # filter_radius

    if idx != len(args_list):
        raise _SplatUnsupported("unexpected trailing data in serialized scene")

    return tuple(args_list), grad_slots


def _cpu_args(args: Iterable[object]) -> Tuple[object, ...]:
    cpu_args: List[object] = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            cpu_args.append(arg.to(device="cpu").contiguous())
        else:
            cpu_args.append(arg)
    return tuple(cpu_args)


def _scene_requires_grad(scene: ScenePayload) -> bool:
    for p in scene.paths:
        if isinstance(p.points, torch.Tensor) and p.points.requires_grad:
            return True
        if isinstance(p.stroke_width, torch.Tensor) and p.stroke_width.requires_grad:
            return True
    for g in scene.shape_groups:
        for paint in (g.fill, g.stroke):
            if paint.color_type is None:
                continue
            for t in paint.params:
                if isinstance(t, torch.Tensor) and t.requires_grad:
                    return True
    return False


__all__ = [
    "_align_grad_devices",
    "_enable_gradient_args",
    "_cpu_args",
    "_scene_requires_grad",
]
