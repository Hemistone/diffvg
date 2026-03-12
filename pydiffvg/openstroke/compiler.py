from __future__ import annotations

from typing import List, Sequence

import torch

from ..device import get_device
from ..shape import Path, Polygon
from .compiled import CompiledOpenStrokeScene, OpenStrokeUnsupported, StrokeStyleRef


class _OutputTypeCompat:
    color = 1


def _output_color_constant():
    try:
        from ..render_pytorch import OutputType
        return OutputType.color
    except Exception:
        return _OutputTypeCompat.color


def _is_identity_transform(transform: torch.Tensor) -> bool:
    identity = torch.eye(3, device=transform.device, dtype=transform.dtype)
    return bool(torch.allclose(transform, identity, atol=1e-6, rtol=0.0))


def _stroke_rgba(group) -> None:
    color = getattr(group, "stroke_color", None)
    if not isinstance(color, torch.Tensor) or color.numel() != 4:
        raise OpenStrokeUnsupported("only constant RGBA stroke colors are supported")


def _path_point_refs(shape) -> torch.Tensor:
    if not isinstance(shape.points, torch.Tensor) or shape.points.ndim != 2 or shape.points.shape[1] != 2:
        raise OpenStrokeUnsupported("path points must be a finite Nx2 tensor")
    if not torch.isfinite(shape.points).all():
        raise OpenStrokeUnsupported("path points must be finite")
    return shape.points


def _scalar_width(shape) -> None:
    width = getattr(shape, "stroke_width", None)
    if not isinstance(width, torch.Tensor) or width.numel() != 1:
        raise OpenStrokeUnsupported("only scalar stroke widths are supported")


_LINE_TO_CUBIC = torch.tensor(
    [
        [1.0, 0.0, 0.0, 0.0],
        [2.0 / 3.0, 1.0 / 3.0, 0.0, 0.0],
        [1.0 / 3.0, 2.0 / 3.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ],
    dtype=torch.float32,
)

_QUADRATIC_TO_CUBIC = torch.tensor(
    [
        [1.0, 0.0, 0.0, 0.0],
        [1.0 / 3.0, 2.0 / 3.0, 0.0, 0.0],
        [0.0, 2.0 / 3.0, 1.0 / 3.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ],
    dtype=torch.float32,
)

_CUBIC_IDENTITY = torch.eye(4, dtype=torch.float32)


def _segment_descriptors(shape, point_offset: int) -> List[tuple[list[int], torch.Tensor]]:
    if isinstance(shape, Polygon):
        if shape.is_closed:
            raise OpenStrokeUnsupported("closed polygons are not supported")
        count = int(shape.points.shape[0])
        if count < 2:
            return []
        segments: List[tuple[list[int], torch.Tensor]] = []
        for index in range(count - 1):
            segments.append(
                ([point_offset + index, point_offset + index + 1, point_offset + index + 1, point_offset + index + 1], _LINE_TO_CUBIC)
            )
        return segments

    if not isinstance(shape, Path):
        raise OpenStrokeUnsupported("only open Path/Polygon shapes are supported")
    if shape.is_closed:
        raise OpenStrokeUnsupported("closed paths are not supported")
    if getattr(shape, "use_distance_approx", False):
        raise OpenStrokeUnsupported("distance approximation paths are not supported")

    num_control_points = shape.num_control_points.detach().to(dtype=torch.int64, device="cpu")
    segments = []
    local_start = 0
    for order in num_control_points.tolist():
        if order == 0:
            src = [point_offset + local_start, point_offset + local_start + 1, point_offset + local_start + 1, point_offset + local_start + 1]
            weight = _LINE_TO_CUBIC
            local_start += 1
        elif order == 1:
            src = [point_offset + local_start, point_offset + local_start + 1, point_offset + local_start + 2, point_offset + local_start + 2]
            weight = _QUADRATIC_TO_CUBIC
            local_start += 2
        elif order == 2:
            src = [point_offset + local_start, point_offset + local_start + 1, point_offset + local_start + 2, point_offset + local_start + 3]
            weight = _CUBIC_IDENTITY
            local_start += 3
        else:
            raise OpenStrokeUnsupported(f"unsupported control point count {order}; expected 0, 1, or 2")
        segments.append((src, weight))
    return segments


def compile_scene(
    canvas_width: int,
    canvas_height: int,
    shapes: Sequence[object],
    shape_groups: Sequence[object],
    *,
    output_type=None,
    use_prefiltering: bool = False,
    eval_positions: torch.Tensor = torch.tensor([]),
    device: torch.device | str | None = None,
) -> CompiledOpenStrokeScene:
    target_device = torch.device(device) if device is not None else get_device()
    output_color = _output_color_constant()
    if output_type is not None and output_type != output_color:
        raise OpenStrokeUnsupported("only OutputType.color is supported")
    if use_prefiltering:
        raise OpenStrokeUnsupported("prefiltering is not supported")
    if eval_positions.numel() != 0:
        raise OpenStrokeUnsupported("SDF/eval_positions is not supported")

    point_refs: List[torch.Tensor] = []
    style_refs: List[StrokeStyleRef] = []
    control_source_indices: List[torch.Tensor] = []
    control_source_weights: List[torch.Tensor] = []
    segment_masks: List[torch.Tensor] = []
    style_indices: List[int] = []
    chunk_orders: List[int] = []

    point_offset = 0
    chunk_order = 0
    for group in shape_groups:
        if getattr(group, "fill_color", None) is not None:
            raise OpenStrokeUnsupported("filled shapes are not supported")
        if not _is_identity_transform(group.shape_to_canvas):
            raise OpenStrokeUnsupported("shape_to_canvas transforms are not supported")
        _stroke_rgba(group)

        for shape_id in group.shape_ids.detach().to(dtype=torch.int64, device="cpu").tolist():
            shape = shapes[shape_id]
            if not isinstance(shape, (Path, Polygon)):
                raise OpenStrokeUnsupported("only open Path/Polygon shapes are supported")
            points = _path_point_refs(shape)
            _scalar_width(shape)
            style_index = len(style_refs)
            style_refs.append(StrokeStyleRef(shape=shape, group=group))
            point_refs.append(points)
            segments = _segment_descriptors(shape, point_offset)
            point_offset += int(points.shape[0])
            if not segments:
                continue
            for start in range(0, len(segments), 3):
                chunk = segments[start:start + 3]
                chunk_indices = torch.zeros((3, 4), dtype=torch.int64, device=target_device)
                chunk_weights = torch.zeros((3, 4, 4), dtype=torch.float32, device=target_device)
                chunk_mask = torch.zeros(3, dtype=torch.bool, device=target_device)
                for seg_index, (indices, weights) in enumerate(chunk):
                    chunk_indices[seg_index] = torch.tensor(indices, dtype=torch.int64, device=target_device)
                    chunk_weights[seg_index] = weights.to(device=target_device)
                    chunk_mask[seg_index] = True
                control_source_indices.append(chunk_indices)
                control_source_weights.append(chunk_weights)
                segment_masks.append(chunk_mask)
                style_indices.append(style_index)
                chunk_orders.append(chunk_order)
                chunk_order += 1

    if not control_source_indices:
        raise OpenStrokeUnsupported("scene does not contain supported open-stroke paths")

    return CompiledOpenStrokeScene(
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        output_type=output_type,
        use_prefiltering=use_prefiltering,
        eval_positions=eval_positions.to(device=target_device),
        point_refs=tuple(point_refs),
        style_refs=tuple(style_refs),
        control_source_indices=torch.stack(control_source_indices, dim=0).contiguous(),
        control_source_weights=torch.stack(control_source_weights, dim=0).contiguous(),
        segment_mask=torch.stack(segment_masks, dim=0).contiguous(),
        style_index=torch.tensor(style_indices, dtype=torch.int64, device=target_device).contiguous(),
        chunk_order=torch.tensor(chunk_orders, dtype=torch.int64, device=target_device).contiguous(),
    )
