"""Conversion utilities bridging vectorizer data structures and diffvg primitives."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import torch

from .api import Path as VectorPath, PenSpec, Segment, VectorDoc, VectorLayer
from .. import color, shape

ShapeLike = shape.Path | shape.Circle | shape.Ellipse | shape.Polygon | shape.Rect


def vector_doc_to_scene(doc: VectorDoc) -> Tuple[List[ShapeLike], List[shape.ShapeGroup]]:
    """Convert a :class:`VectorDoc` into diffvg shape and shape group lists."""

    shapes: List[ShapeLike] = []
    shape_groups: List[shape.ShapeGroup] = []

    for layer_index, layer in enumerate(doc.layers):
        for path_index, vec_path in enumerate(layer.paths):
            diffvg_path = _path_to_diffvg(vec_path, layer_index, path_index)
            if diffvg_path is None:
                continue
            shape_id = len(shapes)
            shapes.append(diffvg_path)

            stroke_color = _color_to_paint(vec_path.pen.stroke_color)
            fill_color = (
                _color_to_paint(vec_path.pen.fill_color)
                if vec_path.pen.fill_color is not None
                else None
            )
            if fill_color is not None and not vec_path.closed:
                fill_color = None
            group = shape.ShapeGroup(
                shape_ids=torch.tensor([shape_id], dtype=torch.int64),
                fill_color=fill_color,
                stroke_color=stroke_color,
                id=f"layer[{layer_index}]:path[{path_index}]",
            )
            shape_groups.append(group)

    return shapes, shape_groups


def scene_to_vector_doc(
    shapes: Sequence[ShapeLike],
    shape_groups: Sequence[shape.ShapeGroup],
    *,
    canvas_size: Tuple[int, int],
) -> VectorDoc:
    """Create a :class:`VectorDoc` from diffvg primitives."""

    shape_to_group: Dict[int, shape.ShapeGroup] = {}
    for group in shape_groups:
        for shape_idx in group.shape_ids.detach().cpu().tolist():
            shape_to_group[int(shape_idx)] = group

    vector_paths: List[VectorPath] = []
    for shape_index, diffvg_shape in enumerate(shapes):
        if not isinstance(diffvg_shape, shape.Path):
            continue
        pen = _pen_from_shape(diffvg_shape, shape_to_group.get(shape_index))
        segments = _segments_from_shape(diffvg_shape)
        vector_paths.append(
            VectorPath(
                segments=tuple(segments),
                closed=bool(diffvg_shape.is_closed),
                pen=pen,
            )
        )

    layers: Tuple[VectorLayer, ...]
    if vector_paths:
        layers = (VectorLayer(paths=tuple(vector_paths)),)
    else:
        layers = tuple()
    return VectorDoc(canvas_size=canvas_size, layers=layers)


def _path_to_diffvg(
    vec_path: VectorPath,
    layer_index: int,
    path_index: int,
) -> shape.Path | None:
    if not vec_path.segments:
        return None

    control_counts: List[int] = []
    points: List[Tuple[float, float]] = []

    for segment in vec_path.segments:
        kind = segment.kind.lower()
        seg_points = [_to_xy(point) for point in segment.points]
        if not seg_points:
            continue
        if not points:
            points.append(seg_points[0])
        elif not _points_close(points[-1], seg_points[0]):
            points.append(seg_points[0])

        if kind == "line":
            _expect_length(seg_points, 2, segment)
            control_counts.append(0)
            points.append(seg_points[1])
        elif kind == "quad":
            _expect_length(seg_points, 3, segment)
            control_counts.append(1)
            points.extend(seg_points[1:])
        elif kind == "cubic":
            _expect_length(seg_points, 4, segment)
            control_counts.append(2)
            points.extend(seg_points[1:])
        else:  # pragma: no cover - defensive guard for unsupported segments
            raise ValueError(f"Unsupported segment kind: {segment.kind}")

    if len(control_counts) == 0 or len(points) < 2:
        return None

    num_control_points = torch.tensor(control_counts, dtype=torch.int32)
    point_tensor = torch.tensor(points, dtype=torch.float32)
    stroke_width = torch.tensor(float(vec_path.pen.stroke_width), dtype=torch.float32)
    return shape.Path(
        num_control_points=num_control_points,
        points=point_tensor,
        is_closed=bool(vec_path.closed),
        stroke_width=stroke_width,
        id=f"layer[{layer_index}]:path[{path_index}]",
    )


def _segments_from_shape(diffvg_path: shape.Path) -> List[Segment]:
    counts = diffvg_path.num_control_points.detach().cpu().tolist()
    points = [tuple(float(v) for v in xy) for xy in diffvg_path.points.detach().cpu().tolist()]

    segments: List[Segment] = []
    point_idx = 0
    for count in counts:
        if count == 0:
            start = points[point_idx]
            end = points[point_idx + 1]
            segments.append(Segment(kind="line", points=(start, end)))
            point_idx += 1
        elif count == 1:
            start = points[point_idx]
            control = points[point_idx + 1]
            end = points[point_idx + 2]
            segments.append(Segment(kind="quad", points=(start, control, end)))
            point_idx += 2
        elif count == 2:
            start = points[point_idx]
            control1 = points[point_idx + 1]
            control2 = points[point_idx + 2]
            end = points[point_idx + 3]
            segments.append(
                Segment(kind="cubic", points=(start, control1, control2, end))
            )
            point_idx += 3
        else:  # pragma: no cover - unsupported diffvg path segment
            raise ValueError(f"Unsupported num_control_points value: {count}")
    return segments


def _pen_from_shape(
    diffvg_path: shape.Path,
    group: shape.ShapeGroup | None,
) -> PenSpec:
    stroke_width = float(diffvg_path.stroke_width.detach().cpu().reshape(-1)[0].item())

    stroke_color = (0.0, 0.0, 0.0, 1.0)
    fill_color = None
    if group is not None:
        stroke_color = _extract_color(group.stroke_color) or stroke_color
        fill_color = _extract_color(group.fill_color)

    return PenSpec(stroke_width=stroke_width, stroke_color=stroke_color, fill_color=fill_color)


def _color_to_paint(rgba: Iterable[float] | None) -> color.Paint | None:
    if rgba is None:
        return None
    return color.Paint(tuple(float(channel) for channel in rgba))


def _extract_color(value) -> Tuple[float, float, float, float] | None:
    if value is None:
        return None
    if isinstance(value, color.Paint):
        tensor = value.to_tensor()
    elif isinstance(value, torch.Tensor):
        tensor = value
    else:
        return None
    comps = [float(c) for c in tensor.detach().cpu().tolist()]
    if len(comps) == 3:
        comps.append(1.0)
    if len(comps) != 4:
        raise ValueError("Expected RGBA color with four components")
    return tuple(comps)  # type: ignore[return-value]


def _to_xy(point: Sequence[float]) -> Tuple[float, float]:
    x, y = point
    return float(x), float(y)


def _expect_length(points: Sequence[Tuple[float, float]], expected: int, segment: Segment) -> None:
    if len(points) != expected:
        raise ValueError(
            f"Segment of kind '{segment.kind}' expected {expected} control points,"
            f" got {len(points)}"
        )


def _points_close(a: Tuple[float, float], b: Tuple[float, float], *, eps: float = 1e-4) -> bool:
    return abs(a[0] - b[0]) <= eps and abs(a[1] - b[1]) <= eps


__all__ = ["vector_doc_to_scene", "scene_to_vector_doc"]
