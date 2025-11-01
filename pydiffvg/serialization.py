from __future__ import annotations

from typing import Optional

import torch
import diffvg

def serialize_scene(
    canvas_width,
    canvas_height,
    shapes,
    shape_groups,
    filter=None,
    output_type=None,
    use_prefiltering=False,
    eval_positions=torch.tensor([]),
    *,
    keep_on_device: bool = False,
    device: Optional[torch.device | str] = None,
):
    """Serialize Python-side scene objects into a flat args list for diffvg.

    Notes
    - This mirrors the previous RenderFunction.serialize_scene behavior.
    - Imports for pydiffvg submodules are local to avoid circular imports at module load.
    - The `keep_on_device` flag is intended for backends that operate entirely in PyTorch; the
      baseline diffvg renderer forces tensors back to CPU memory.
    """
    # Lazy imports to avoid circular dependencies at import time
    from .device import get_device
    from .pixel_filter import PixelFilter
    from .shape import Circle, Ellipse, Path, Polygon, Rect
    from .color import LinearGradient, RadialGradient, Paint

    target_device: torch.device
    if keep_on_device:
        target_device = torch.device(device) if device is not None else get_device()
    else:
        target_device = torch.device("cpu")

    def _move_tensor(tensor: torch.Tensor, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        if dtype is not None:
            tensor = tensor.to(dtype)
        if keep_on_device:
            moved = tensor.to(device=target_device)
        else:
            moved = tensor.cpu()
        return moved.contiguous()

    if filter is None:
        filter = PixelFilter(type=diffvg.FilterType.box, radius=torch.tensor(0.5))

    num_shapes = len(shapes)
    num_shape_groups = len(shape_groups)
    args = []
    args.append(canvas_width)
    args.append(canvas_height)
    args.append(num_shapes)
    args.append(num_shape_groups)
    args.append(output_type)
    args.append(use_prefiltering)
    eval_device = target_device if keep_on_device else get_device()
    args.append(eval_positions.to(eval_device))

    for shape in shapes:
        use_thickness = False
        if isinstance(shape, Circle):
            assert shape.center.is_contiguous()
            args.append(diffvg.ShapeType.circle)
            args.append(_move_tensor(shape.radius))
            args.append(_move_tensor(shape.center))
        elif isinstance(shape, Ellipse):
            assert shape.radius.is_contiguous()
            assert shape.center.is_contiguous()
            args.append(diffvg.ShapeType.ellipse)
            args.append(_move_tensor(shape.radius))
            args.append(_move_tensor(shape.center))
        elif isinstance(shape, Path):
            assert shape.num_control_points.is_contiguous()
            assert shape.points.is_contiguous()
            assert shape.points.shape[1] == 2
            assert torch.isfinite(shape.points).all()
            args.append(diffvg.ShapeType.path)
            args.append(_move_tensor(shape.num_control_points, dtype=torch.int32))
            args.append(_move_tensor(shape.points))
            if len(shape.stroke_width.shape) > 0 and shape.stroke_width.shape[0] > 1:
                assert torch.isfinite(shape.stroke_width).all()
                use_thickness = True
                args.append(_move_tensor(shape.stroke_width))
            else:
                args.append(None)
            args.append(shape.is_closed)
            args.append(shape.use_distance_approx)
        elif isinstance(shape, Polygon):
            assert shape.points.is_contiguous()
            assert shape.points.shape[1] == 2
            args.append(diffvg.ShapeType.path)
            if shape.is_closed:
                args.append(_move_tensor(torch.zeros(shape.points.shape[0], dtype=torch.int32)))
            else:
                args.append(_move_tensor(torch.zeros(shape.points.shape[0] - 1, dtype=torch.int32)))
            args.append(_move_tensor(shape.points))
            args.append(None)
            args.append(shape.is_closed)
            args.append(False)  # use_distance_approx
        elif isinstance(shape, Rect):
            assert shape.p_min.is_contiguous()
            assert shape.p_max.is_contiguous()
            args.append(diffvg.ShapeType.rect)
            args.append(_move_tensor(shape.p_min))
            args.append(_move_tensor(shape.p_max))
        else:
            assert False

        if use_thickness:
            args.append(_move_tensor(torch.tensor(0.0)))
        else:
            args.append(_move_tensor(shape.stroke_width))

    for group_index, shape_group in enumerate(shape_groups):
        assert shape_group.shape_ids.is_contiguous()
        shape_indices = _move_tensor(shape_group.shape_ids, dtype=torch.int32)
        shape_indices_host = shape_group.shape_ids.to(torch.int64).cpu().tolist()
        args.append(shape_indices)

        # Fill color
        if shape_group.fill_color is None:
            args.append(None)
        elif isinstance(shape_group.fill_color, torch.Tensor):
            assert shape_group.fill_color.is_contiguous()
            args.append(diffvg.ColorType.constant)
            args.append(_move_tensor(shape_group.fill_color))
        elif isinstance(shape_group.fill_color, Paint):
            tensor = shape_group.fill_color.to_tensor()
            assert tensor.is_contiguous()
            args.append(diffvg.ColorType.constant)
            args.append(_move_tensor(tensor))
        elif isinstance(shape_group.fill_color, LinearGradient):
            assert shape_group.fill_color.begin.is_contiguous()
            assert shape_group.fill_color.end.is_contiguous()
            assert shape_group.fill_color.offsets.is_contiguous()
            assert shape_group.fill_color.stop_colors.is_contiguous()
            args.append(diffvg.ColorType.linear_gradient)
            args.append(_move_tensor(shape_group.fill_color.begin))
            args.append(_move_tensor(shape_group.fill_color.end))
            args.append(_move_tensor(shape_group.fill_color.offsets))
            args.append(_move_tensor(shape_group.fill_color.stop_colors))
        elif isinstance(shape_group.fill_color, RadialGradient):
            assert shape_group.fill_color.center.is_contiguous()
            assert shape_group.fill_color.radius.is_contiguous()
            assert shape_group.fill_color.offsets.is_contiguous()
            assert shape_group.fill_color.stop_colors.is_contiguous()
            args.append(diffvg.ColorType.radial_gradient)
            args.append(_move_tensor(shape_group.fill_color.center))
            args.append(_move_tensor(shape_group.fill_color.radius))
            args.append(_move_tensor(shape_group.fill_color.offsets))
            args.append(_move_tensor(shape_group.fill_color.stop_colors))

        if shape_group.fill_color is not None:
            for shape_idx in shape_indices_host:
                shape = shapes[shape_idx]
                is_open_path = (
                    isinstance(shape, Path) or isinstance(shape, Polygon)
                ) and not shape.is_closed
                if is_open_path:
                    group_label = shape_group.id or f"group[{group_index}]"
                    shape_label = getattr(shape, "id", "") or f"shape[{shape_idx}]"
                    raise ValueError(
                        f"ShapeGroup {group_label} applies a fill color to open path {shape_label}. "
                        "Close the path (`is_closed=True`) or drop the fill before rendering."
                    )

        # Stroke color
        if shape_group.stroke_color is None:
            args.append(None)
        elif isinstance(shape_group.stroke_color, torch.Tensor):
            assert shape_group.stroke_color.is_contiguous()
            args.append(diffvg.ColorType.constant)
            args.append(_move_tensor(shape_group.stroke_color))
        elif isinstance(shape_group.stroke_color, Paint):
            tensor = shape_group.stroke_color.to_tensor()
            assert tensor.is_contiguous()
            args.append(diffvg.ColorType.constant)
            args.append(_move_tensor(tensor))
        elif isinstance(shape_group.stroke_color, LinearGradient):
            assert shape_group.stroke_color.begin.is_contiguous()
            assert shape_group.stroke_color.end.is_contiguous()
            assert shape_group.stroke_color.offsets.is_contiguous()
            assert shape_group.stroke_color.stop_colors.is_contiguous()
            assert torch.isfinite(shape_group.stroke_color.stop_colors).all()
            args.append(diffvg.ColorType.linear_gradient)
            args.append(_move_tensor(shape_group.stroke_color.begin))
            args.append(_move_tensor(shape_group.stroke_color.end))
            args.append(_move_tensor(shape_group.stroke_color.offsets))
            args.append(_move_tensor(shape_group.stroke_color.stop_colors))
        elif isinstance(shape_group.stroke_color, RadialGradient):
            assert shape_group.stroke_color.center.is_contiguous()
            assert shape_group.stroke_color.radius.is_contiguous()
            assert shape_group.stroke_color.offsets.is_contiguous()
            assert shape_group.stroke_color.stop_colors.is_contiguous()
            assert torch.isfinite(shape_group.stroke_color.stop_colors).all()
            args.append(diffvg.ColorType.radial_gradient)
            args.append(_move_tensor(shape_group.stroke_color.center))
            args.append(_move_tensor(shape_group.stroke_color.radius))
            args.append(_move_tensor(shape_group.stroke_color.offsets))
            args.append(_move_tensor(shape_group.stroke_color.stop_colors))

        args.append(shape_group.use_even_odd_rule)
        args.append(_move_tensor(shape_group.shape_to_canvas.contiguous()))

    args.append(filter.type)
    args.append(_move_tensor(filter.radius))
    return args


__all__ = ["serialize_scene"]
