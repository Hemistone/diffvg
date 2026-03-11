"""Geometry helpers for plotter-oriented SVG and scene analysis."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch

from ..shape import Circle, Ellipse, Path as DiffvgPath, Polygon, Rect


@dataclass(frozen=True)
class PathSegment:
    kind: str
    points: tuple[complex, ...]


@dataclass(frozen=True)
class StrokePolyline:
    points: tuple[complex, ...]
    closed: bool


_EPS = 1e-6


def _point_to_complex(point: torch.Tensor) -> complex:
    point_cpu = point.detach().cpu()
    return complex(float(point_cpu[0]), float(point_cpu[1]))


def _tensor_clone(value: torch.Tensor) -> torch.Tensor:
    return value.detach().clone()


def _apply_transform(points: Sequence[complex], transform: torch.Tensor | None) -> tuple[complex, ...]:
    if transform is None:
        return tuple(points)
    tf = transform.detach().cpu().to(dtype=torch.float32)
    out: list[complex] = []
    for point in points:
        vec = torch.tensor([point.real, point.imag, 1.0], dtype=tf.dtype)
        mapped = tf @ vec
        mapped_xy = mapped[:2] / mapped[2].clamp_min(1e-8)
        out.append(complex(float(mapped_xy[0]), float(mapped_xy[1])))
    return tuple(out)


def polyline_length(points: Sequence[complex]) -> float:
    return sum(abs(b - a) for a, b in zip(points, points[1:]))


def _lerp(a: complex, b: complex, t: float) -> complex:
    return a + (b - a) * t


def _quad_point(p0: complex, p1: complex, p2: complex, t: float) -> complex:
    a = _lerp(p0, p1, t)
    b = _lerp(p1, p2, t)
    return _lerp(a, b, t)


def _cubic_point(p0: complex, p1: complex, p2: complex, p3: complex, t: float) -> complex:
    a = _lerp(p0, p1, t)
    b = _lerp(p1, p2, t)
    c = _lerp(p2, p3, t)
    d = _lerp(a, b, t)
    e = _lerp(b, c, t)
    return _lerp(d, e, t)


def _approx_quad_length(p0: complex, p1: complex, p2: complex, steps: int = 24) -> float:
    pts = [_quad_point(p0, p1, p2, idx / steps) for idx in range(steps + 1)]
    return polyline_length(pts)


def _approx_cubic_length(p0: complex, p1: complex, p2: complex, p3: complex, steps: int = 32) -> float:
    pts = [_cubic_point(p0, p1, p2, p3, idx / steps) for idx in range(steps + 1)]
    return polyline_length(pts)


def _sample_quad_points(p0: complex, p1: complex, p2: complex, steps: int = 24) -> tuple[complex, ...]:
    return tuple(_quad_point(p0, p1, p2, idx / steps) for idx in range(steps + 1))


def _sample_cubic_points(
    p0: complex,
    p1: complex,
    p2: complex,
    p3: complex,
    steps: int = 32,
) -> tuple[complex, ...]:
    return tuple(_cubic_point(p0, p1, p2, p3, idx / steps) for idx in range(steps + 1))


def is_identity_transform(transform: torch.Tensor | None, *, atol: float = 1e-6) -> bool:
    if transform is None:
        return True
    tf = transform.detach().cpu().to(dtype=torch.float32)
    eye = torch.eye(3, dtype=tf.dtype)
    return bool(torch.allclose(tf, eye, atol=atol, rtol=0.0))


def path_segments(path: DiffvgPath) -> list[PathSegment]:
    points = path.points.detach()
    num_points = int(points.shape[0])
    if num_points == 0:
        return []
    point_id = 1
    current = _point_to_complex(points[0])
    segments: list[PathSegment] = []
    counts = path.num_control_points.detach().cpu().tolist()

    def point_at(index: int) -> complex:
        if path.is_closed:
            index = index % num_points
        return _point_to_complex(points[index])

    for cp_count in counts:
        cp = int(cp_count)
        if cp == 0:
            end = point_at(point_id)
            segments.append(PathSegment("line", (current, end)))
            current = end
            point_id += 1
        elif cp == 1:
            control = point_at(point_id)
            end = point_at(point_id + 1)
            segments.append(PathSegment("quad", (current, control, end)))
            current = end
            point_id += 2
        elif cp == 2:
            control1 = point_at(point_id)
            control2 = point_at(point_id + 1)
            end = point_at(point_id + 2)
            segments.append(PathSegment("cubic", (current, control1, control2, end)))
            current = end
            point_id += 3
        else:
            raise ValueError(f"Unsupported control-point count '{cp}' for plotter geometry")
    return segments


def segment_length(segment: PathSegment) -> float:
    if segment.kind == "line":
        return abs(segment.points[1] - segment.points[0])
    if segment.kind == "quad":
        return _approx_quad_length(*segment.points)
    if segment.kind == "cubic":
        return _approx_cubic_length(*segment.points)
    raise ValueError(f"Unsupported segment kind '{segment.kind}'")


def path_length(path: DiffvgPath) -> float:
    return sum(segment_length(segment) for segment in path_segments(path))


def path_endpoints(path: DiffvgPath) -> tuple[complex, complex]:
    segments = path_segments(path)
    if not segments:
        return 0j, 0j
    return segments[0].points[0], segments[-1].points[-1]


def _normalized(vector: complex) -> complex:
    norm = abs(vector)
    if norm <= _EPS:
        return 0j
    return vector / norm


def path_tangent(path: DiffvgPath, *, head: bool) -> complex:
    segments = path_segments(path)
    if not segments:
        return 0j
    segment = segments[0] if head else segments[-1]
    pts = segment.points
    if segment.kind == "line":
        tangent = pts[1] - pts[0]
    elif segment.kind == "quad":
        tangent = (pts[1] - pts[0]) if head else (pts[2] - pts[1])
    elif segment.kind == "cubic":
        tangent = (pts[1] - pts[0]) if head else (pts[3] - pts[2])
    else:
        tangent = 0j
    if abs(tangent) <= _EPS:
        tangent = pts[-1] - pts[0]
    return _normalized(tangent)


def reverse_path(path: DiffvgPath) -> DiffvgPath:
    reversed_segments: list[PathSegment] = []
    for segment in reversed(path_segments(path)):
        if segment.kind == "line":
            reversed_segments.append(PathSegment("line", (segment.points[1], segment.points[0])))
        elif segment.kind == "quad":
            reversed_segments.append(PathSegment("quad", (segment.points[2], segment.points[1], segment.points[0])))
        elif segment.kind == "cubic":
            reversed_segments.append(
                PathSegment("cubic", (segment.points[3], segment.points[2], segment.points[1], segment.points[0])))
        else:
            raise ValueError(f"Unsupported segment kind '{segment.kind}'")
    return build_path_from_segments(
        reversed_segments,
        stroke_width=_tensor_clone(path.stroke_width),
        is_closed=bool(path.is_closed),
        path_id=path.id,
        use_distance_approx=bool(path.use_distance_approx),
        device=path.points.device,
        dtype=path.points.dtype,
    )


def build_path_from_segments(
    segments: Sequence[PathSegment],
    *,
    stroke_width: torch.Tensor,
    is_closed: bool,
    path_id: str = "",
    use_distance_approx: bool = False,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> DiffvgPath:
    if not segments:
        raise ValueError("Cannot build a diffvg path from zero segments")
    if device is None:
        device = stroke_width.device
    if dtype is None:
        dtype = stroke_width.dtype if stroke_width.is_floating_point() else torch.float32

    points: list[tuple[float, float]] = [(segments[0].points[0].real, segments[0].points[0].imag)]
    counts: list[int] = []
    for segment in segments:
        if segment.kind == "line":
            counts.append(0)
            points.append((segment.points[1].real, segment.points[1].imag))
        elif segment.kind == "quad":
            counts.append(1)
            points.append((segment.points[1].real, segment.points[1].imag))
            points.append((segment.points[2].real, segment.points[2].imag))
        elif segment.kind == "cubic":
            counts.append(2)
            points.append((segment.points[1].real, segment.points[1].imag))
            points.append((segment.points[2].real, segment.points[2].imag))
            points.append((segment.points[3].real, segment.points[3].imag))
        else:
            raise ValueError(f"Unsupported segment kind '{segment.kind}'")

    point_tensor = torch.tensor(points, dtype=dtype, device=device)
    count_tensor = torch.tensor(counts, dtype=torch.int32, device=device)
    return DiffvgPath(
        num_control_points=count_tensor,
        points=point_tensor,
        is_closed=bool(is_closed),
        stroke_width=_tensor_clone(stroke_width).to(device=device),
        id=path_id,
        use_distance_approx=bool(use_distance_approx),
    )


def concatenate_paths(
    paths: Sequence[DiffvgPath],
    *,
    path_id: str,
    use_distance_approx: bool = False,
) -> DiffvgPath:
    if not paths:
        raise ValueError("Cannot concatenate zero paths")
    device = paths[0].points.device
    dtype = paths[0].points.dtype
    segments: list[PathSegment] = []
    prev_end: complex | None = None
    weighted_width = 0.0
    total_length = 0.0
    for path in paths:
        if path.is_closed:
            raise ValueError("concatenate_paths only supports open paths")
        path_segments_list = path_segments(path)
        if not path_segments_list:
            continue
        current_start = path_segments_list[0].points[0]
        if prev_end is not None and abs(current_start - prev_end) > _EPS:
            segments.append(PathSegment("line", (prev_end, current_start)))
        segments.extend(path_segments_list)
        path_len = path_length(path)
        weighted_width += path_len * float(path.stroke_width.detach().cpu().item())
        total_length += path_len
        prev_end = path_segments_list[-1].points[-1]
    if not segments:
        raise ValueError("concatenate_paths produced no segments")
    width_value = weighted_width / max(total_length, _EPS)
    width_tensor = torch.tensor(width_value, dtype=paths[0].stroke_width.dtype, device=device)
    return build_path_from_segments(
        segments,
        stroke_width=width_tensor,
        is_closed=False,
        path_id=path_id,
        use_distance_approx=use_distance_approx,
        device=device,
        dtype=dtype,
    )


def sample_path_polyline(path: DiffvgPath) -> StrokePolyline:
    out: list[complex] = []
    for segment in path_segments(path):
        if segment.kind == "line":
            pts = segment.points
        elif segment.kind == "quad":
            pts = _sample_quad_points(*segment.points)
        elif segment.kind == "cubic":
            pts = _sample_cubic_points(*segment.points)
        else:
            raise ValueError(f"Unsupported segment kind '{segment.kind}'")
        if out:
            out.extend(pts[1:])
        else:
            out.extend(pts)
    return StrokePolyline(points=tuple(out), closed=bool(path.is_closed))


def sample_shape_polyline(shape, transform: torch.Tensor | None = None) -> StrokePolyline | None:
    if isinstance(shape, DiffvgPath):
        sampled = sample_path_polyline(shape)
        return StrokePolyline(points=_apply_transform(sampled.points, transform), closed=sampled.closed)
    if isinstance(shape, Polygon):
        points = [_point_to_complex(point) for point in shape.points]
        if shape.is_closed and points and points[0] != points[-1]:
            points.append(points[0])
        return StrokePolyline(points=_apply_transform(points, transform), closed=bool(shape.is_closed))
    if isinstance(shape, Rect):
        p_min = _point_to_complex(shape.p_min)
        p_max = _point_to_complex(shape.p_max)
        points = [
            complex(p_min.real, p_min.imag),
            complex(p_max.real, p_min.imag),
            complex(p_max.real, p_max.imag),
            complex(p_min.real, p_max.imag),
            complex(p_min.real, p_min.imag),
        ]
        return StrokePolyline(points=_apply_transform(points, transform), closed=True)
    if isinstance(shape, Circle):
        center = _point_to_complex(shape.center)
        radius = float(shape.radius.detach().cpu().item())
        steps = 64
        points = [
            center + complex(radius * math.cos(2.0 * math.pi * idx / steps), radius * math.sin(2.0 * math.pi * idx / steps))
            for idx in range(steps)
        ]
        points.append(points[0])
        return StrokePolyline(points=_apply_transform(points, transform), closed=True)
    if isinstance(shape, Ellipse):
        center = _point_to_complex(shape.center)
        radius = shape.radius.detach().cpu()
        rx = float(radius[0])
        ry = float(radius[1])
        steps = 64
        points = [
            center + complex(rx * math.cos(2.0 * math.pi * idx / steps), ry * math.sin(2.0 * math.pi * idx / steps))
            for idx in range(steps)
        ]
        points.append(points[0])
        return StrokePolyline(points=_apply_transform(points, transform), closed=True)
    return None


__all__ = [
    "PathSegment",
    "StrokePolyline",
    "build_path_from_segments",
    "concatenate_paths",
    "is_identity_transform",
    "path_endpoints",
    "path_length",
    "path_segments",
    "path_tangent",
    "polyline_length",
    "reverse_path",
    "sample_path_polyline",
    "sample_shape_polyline",
    "segment_length",
]
