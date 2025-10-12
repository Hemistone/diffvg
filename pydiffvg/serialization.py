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
):
    """Serialize Python-side scene objects into a flat args list for diffvg.

    Notes
    - This mirrors the previous RenderFunction.serialize_scene behavior.
    - Imports for pydiffvg submodules are local to avoid circular imports at module load.
    """
    # Lazy imports to avoid circular dependencies at import time
    from .device import get_device
    from .pixel_filter import PixelFilter
    from .shape import Circle, Ellipse, Path, Polygon, Rect
    from .color import LinearGradient, RadialGradient

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
    args.append(eval_positions.to(get_device()))

    for shape in shapes:
        use_thickness = False
        if isinstance(shape, Circle):
            assert shape.center.is_contiguous()
            args.append(diffvg.ShapeType.circle)
            args.append(shape.radius.cpu())
            args.append(shape.center.cpu())
        elif isinstance(shape, Ellipse):
            assert shape.radius.is_contiguous()
            assert shape.center.is_contiguous()
            args.append(diffvg.ShapeType.ellipse)
            args.append(shape.radius.cpu())
            args.append(shape.center.cpu())
        elif isinstance(shape, Path):
            assert shape.num_control_points.is_contiguous()
            assert shape.points.is_contiguous()
            assert shape.points.shape[1] == 2
            assert torch.isfinite(shape.points).all()
            args.append(diffvg.ShapeType.path)
            args.append(shape.num_control_points.to(torch.int32).cpu())
            args.append(shape.points.cpu())
            if len(shape.stroke_width.shape) > 0 and shape.stroke_width.shape[0] > 1:
                assert torch.isfinite(shape.stroke_width).all()
                use_thickness = True
                args.append(shape.stroke_width.cpu())
            else:
                args.append(None)
            args.append(shape.is_closed)
            args.append(shape.use_distance_approx)
        elif isinstance(shape, Polygon):
            assert shape.points.is_contiguous()
            assert shape.points.shape[1] == 2
            args.append(diffvg.ShapeType.path)
            if shape.is_closed:
                args.append(torch.zeros(shape.points.shape[0], dtype=torch.int32))
            else:
                args.append(torch.zeros(shape.points.shape[0] - 1, dtype=torch.int32))
            args.append(shape.points.cpu())
            args.append(None)
            args.append(shape.is_closed)
            args.append(False)  # use_distance_approx
        elif isinstance(shape, Rect):
            assert shape.p_min.is_contiguous()
            assert shape.p_max.is_contiguous()
            args.append(diffvg.ShapeType.rect)
            args.append(shape.p_min.cpu())
            args.append(shape.p_max.cpu())
        else:
            assert False

        if use_thickness:
            args.append(torch.tensor(0.0))
        else:
            args.append(shape.stroke_width.cpu())

    for shape_group in shape_groups:
        assert shape_group.shape_ids.is_contiguous()
        args.append(shape_group.shape_ids.to(torch.int32).cpu())

        # Fill color
        if shape_group.fill_color is None:
            args.append(None)
        elif isinstance(shape_group.fill_color, torch.Tensor):
            assert shape_group.fill_color.is_contiguous()
            args.append(diffvg.ColorType.constant)
            args.append(shape_group.fill_color.cpu())
        elif isinstance(shape_group.fill_color, LinearGradient):
            assert shape_group.fill_color.begin.is_contiguous()
            assert shape_group.fill_color.end.is_contiguous()
            assert shape_group.fill_color.offsets.is_contiguous()
            assert shape_group.fill_color.stop_colors.is_contiguous()
            args.append(diffvg.ColorType.linear_gradient)
            args.append(shape_group.fill_color.begin.cpu())
            args.append(shape_group.fill_color.end.cpu())
            args.append(shape_group.fill_color.offsets.cpu())
            args.append(shape_group.fill_color.stop_colors.cpu())
        elif isinstance(shape_group.fill_color, RadialGradient):
            assert shape_group.fill_color.center.is_contiguous()
            assert shape_group.fill_color.radius.is_contiguous()
            assert shape_group.fill_color.offsets.is_contiguous()
            assert shape_group.fill_color.stop_colors.is_contiguous()
            args.append(diffvg.ColorType.radial_gradient)
            args.append(shape_group.fill_color.center.cpu())
            args.append(shape_group.fill_color.radius.cpu())
            args.append(shape_group.fill_color.offsets.cpu())
            args.append(shape_group.fill_color.stop_colors.cpu())

        # Warn on non-closed paths under fill is handled in caller environment; kept parity

        # Stroke color
        if shape_group.stroke_color is None:
            args.append(None)
        elif isinstance(shape_group.stroke_color, torch.Tensor):
            assert shape_group.stroke_color.is_contiguous()
            args.append(diffvg.ColorType.constant)
            args.append(shape_group.stroke_color.cpu())
        elif isinstance(shape_group.stroke_color, LinearGradient):
            assert shape_group.stroke_color.begin.is_contiguous()
            assert shape_group.stroke_color.end.is_contiguous()
            assert shape_group.stroke_color.offsets.is_contiguous()
            assert shape_group.stroke_color.stop_colors.is_contiguous()
            assert torch.isfinite(shape_group.stroke_color.stop_colors).all()
            args.append(diffvg.ColorType.linear_gradient)
            args.append(shape_group.stroke_color.begin.cpu())
            args.append(shape_group.stroke_color.end.cpu())
            args.append(shape_group.stroke_color.offsets.cpu())
            args.append(shape_group.stroke_color.stop_colors.cpu())
        elif isinstance(shape_group.stroke_color, RadialGradient):
            assert shape_group.stroke_color.center.is_contiguous()
            assert shape_group.stroke_color.radius.is_contiguous()
            assert shape_group.stroke_color.offsets.is_contiguous()
            assert shape_group.stroke_color.stop_colors.is_contiguous()
            assert torch.isfinite(shape_group.stroke_color.stop_colors).all()
            args.append(diffvg.ColorType.radial_gradient)
            args.append(shape_group.stroke_color.center.cpu())
            args.append(shape_group.stroke_color.radius.cpu())
            args.append(shape_group.stroke_color.offsets.cpu())
            args.append(shape_group.stroke_color.stop_colors.cpu())

        args.append(shape_group.use_even_odd_rule)
        args.append(shape_group.shape_to_canvas.contiguous().cpu())

    args.append(filter.type)
    args.append(filter.radius.cpu())
    return args


__all__ = ["serialize_scene"]
