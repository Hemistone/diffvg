"""Stroke-only SVG parsing utilities for pydiffvg.

The maintained runtime only needs to read SVGs that are compatible with the
stroke-first product surface:

- open paths / polylines / lines
- constant RGBA stroke colors
- scalar stroke widths
- optional group transforms
- no fills, gradients, filters, or non-path primitives
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable
import warnings
import xml.etree.ElementTree as etree

import matplotlib.colors
import numpy as np
import torch

from .shape import Polygon, ShapeGroup, from_svg_path


def remove_namespaces(s: str) -> str:
    return re.sub(r"{.*}", "", s)


def parse_int(s: str) -> int:
    return int(round(parse_length(s)))


def parse_length(s: str) -> float:
    if not isinstance(s, str):
        raise TypeError("SVG length must be a string")
    match = re.match(r"^\s*([+-]?\d*\.?\d+)", s)
    if not match:
        raise ValueError(f"Could not parse SVG length '{s}'")
    return float(match.group(1))


def parse_color(s: str, defs: dict | None = None):
    del defs
    if s is None:
        return None
    if isinstance(s, torch.Tensor):
        return s
    text = str(s).strip()
    if text == "" or text.lower() == "none":
        return None
    if text.lower().startswith("url("):
        raise ValueError("stroke-first SVG parser does not support paint servers or gradients")
    rgb_match = re.match(r"^(rgba?|RGBA?)\((.+)\)$", text)
    if rgb_match:
        parts = [part.strip() for part in rgb_match.group(2).split(",") if part.strip()]
        if len(parts) not in (3, 4):
            raise ValueError(f"Unsupported SVG color '{text}'")
        rgb = [float(part) / 255.0 for part in parts[:3]]
        alpha = 1.0 if len(parts) == 3 else float(parts[3])
        return torch.tensor([rgb[0], rgb[1], rgb[2], alpha], dtype=torch.float32)
    try:
        rgba = matplotlib.colors.to_rgba(text)
    except ValueError as exc:
        raise ValueError(f"Unsupported SVG color '{text}'") from exc
    return torch.tensor(rgba, dtype=torch.float32)


def _parse_transform_substr(transform_substr: str) -> np.ndarray:
    type_str, value_str = transform_substr.split("(", 1)
    value_str = value_str.replace(",", " ")
    values = list(map(float, filter(None, value_str.split())))

    transform = np.identity(3, dtype=np.float32)
    if "matrix" in type_str:
        transform[0:2, 0:3] = np.array([values[0:6:2], values[1:6:2]], dtype=np.float32)
    elif "translate" in type_str:
        transform[0, 2] = values[0]
        if len(values) > 1:
            transform[1, 2] = values[1]
    elif "scale" in type_str:
        x_scale = values[0]
        y_scale = values[1] if len(values) > 1 else x_scale
        transform[0, 0] = x_scale
        transform[1, 1] = y_scale
    elif "rotate" in type_str:
        angle = values[0] * np.pi / 180.0
        offset = values[1:3] if len(values) == 3 else (0.0, 0.0)
        tf_offset = np.identity(3, dtype=np.float32)
        tf_offset[0:2, 2:3] = np.array([[offset[0]], [offset[1]]], dtype=np.float32)
        tf_rotate = np.identity(3, dtype=np.float32)
        tf_rotate[0:2, 0:2] = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
            dtype=np.float32,
        )
        tf_offset_neg = np.identity(3, dtype=np.float32)
        tf_offset_neg[0:2, 2:3] = np.array([[-offset[0]], [-offset[1]]], dtype=np.float32)
        transform = tf_offset.dot(tf_rotate).dot(tf_offset_neg)
    elif "skewX" in type_str:
        transform[0, 1] = np.tan(values[0] * np.pi / 180.0)
    elif "skewY" in type_str:
        transform[1, 0] = np.tan(values[0] * np.pi / 180.0)
    else:
        warnings.warn(f"Unknown SVG transform type '{type_str}'; ignoring it")
    return transform


def parse_transform(transform_str: str | None):
    if not transform_str:
        return torch.eye(3, dtype=torch.float32)
    if not isinstance(transform_str, str):
        raise TypeError("Must provide a string to parse")
    total_transform = np.identity(3, dtype=np.float32)
    for substr in transform_str.split(")")[:-1]:
        total_transform = total_transform.dot(_parse_transform_substr(substr))
    return torch.from_numpy(total_transform).to(dtype=torch.float32)


def _parse_style_string(text: str | None) -> dict[str, str]:
    if not text:
        return {}
    out: dict[str, str] = {}
    for item in text.split(";"):
        if ":" not in item:
            continue
        key, value = item.split(":", 1)
        out[key.strip()] = value.strip()
    return out


def _compose_transform(transform: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    pts = torch.cat((points, torch.ones((points.shape[0], 1), dtype=torch.float32)), dim=1)
    pts = pts @ transform.t()
    pts = pts / pts[:, 2:3]
    return pts[:, :2].contiguous()


@dataclass(frozen=True)
class _ParseState:
    transform: torch.Tensor
    stroke_color: torch.Tensor | None
    stroke_width: float | None


def _default_state() -> _ParseState:
    return _ParseState(transform=torch.eye(3, dtype=torch.float32), stroke_color=None, stroke_width=None)


def _resolved_attribs(node: etree.Element) -> dict[str, str]:
    attribs = dict(node.attrib)
    style = _parse_style_string(attribs.get("style"))
    for key, value in style.items():
        attribs.setdefault(key, value)
    return attribs


def _resolve_state(node: etree.Element, parent: _ParseState) -> _ParseState:
    attribs = _resolved_attribs(node)
    transform = parent.transform
    if "transform" in attribs:
        transform = parent.transform @ parse_transform(attribs["transform"])

    fill = attribs.get("fill")
    if fill is not None and fill.strip().lower() != "none":
        raise ValueError("stroke-first SVG parser does not support filled shapes")
    if "fill-rule" in attribs:
        raise ValueError("stroke-first SVG parser does not support fill rules")

    stroke_color = parent.stroke_color.detach().clone() if isinstance(parent.stroke_color, torch.Tensor) else None
    if "stroke" in attribs:
        stroke_color = parse_color(attribs["stroke"])

    stroke_width = parent.stroke_width
    if "stroke-width" in attribs:
        stroke_width = parse_length(attribs["stroke-width"])

    if stroke_color is not None:
        opacity = 1.0
        if "opacity" in attribs:
            opacity *= float(attribs["opacity"])
        if "stroke-opacity" in attribs:
            opacity *= float(attribs["stroke-opacity"])
        stroke_color = stroke_color.reshape(4).clone()
        stroke_color[3] = stroke_color[3] * float(max(0.0, min(1.0, opacity)))

    return _ParseState(transform=transform, stroke_color=stroke_color, stroke_width=stroke_width)


def _group_for_shape(shape_index: int, stroke_color: torch.Tensor) -> ShapeGroup:
    return ShapeGroup(
        shape_ids=torch.tensor([shape_index], dtype=torch.int32),
        fill_color=None,
        use_even_odd_rule=False,
        stroke_color=stroke_color.reshape(4).clone(),
        shape_to_canvas=torch.eye(3, dtype=torch.float32),
        id="",
    )


def _append_path_shapes(
    shapes: list[object],
    shape_groups: list[ShapeGroup],
    d: str,
    state: _ParseState,
) -> None:
    if state.stroke_color is None:
        return
    if state.stroke_width is None:
        raise ValueError("stroke-first SVG parser requires explicit stroke-width on visible strokes")
    parsed = from_svg_path(d, state.transform, force_close=False)
    for path in parsed:
        if bool(path.is_closed):
            raise ValueError("stroke-first SVG parser does not support closed paths")
        path.stroke_width = torch.tensor(float(state.stroke_width) / 2.0, dtype=torch.float32)
        shape_index = len(shapes)
        shapes.append(path)
        shape_groups.append(_group_for_shape(shape_index, state.stroke_color))


def _append_polyline_shape(
    shapes: list[object],
    shape_groups: list[ShapeGroup],
    points: Iterable[tuple[float, float]],
    state: _ParseState,
    *,
    close: bool = False,
) -> None:
    if state.stroke_color is None:
        return
    if state.stroke_width is None:
        raise ValueError("stroke-first SVG parser requires explicit stroke-width on visible strokes")
    pts = torch.tensor(list(points), dtype=torch.float32).view(-1, 2)
    if pts.shape[0] < 2:
        return
    if close:
        raise ValueError("stroke-first SVG parser does not support closed polygons")
    pts = _compose_transform(state.transform, pts)
    shape = Polygon(points=pts, is_closed=False, stroke_width=torch.tensor(float(state.stroke_width) / 2.0))
    shape_index = len(shapes)
    shapes.append(shape)
    shape_groups.append(_group_for_shape(shape_index, state.stroke_color))


def _parse_points_attr(text: str) -> list[tuple[float, float]]:
    cleaned = text.replace(",", " ")
    values = [float(item) for item in cleaned.split() if item.strip()]
    if len(values) % 2 != 0:
        raise ValueError("polyline/polygon points must contain an even number of coordinates")
    return [(values[i], values[i + 1]) for i in range(0, len(values), 2)]


def _parse_shape(node: etree.Element, state: _ParseState, shapes: list[object], shape_groups: list[ShapeGroup]) -> None:
    tag = remove_namespaces(node.tag)
    if tag == "path":
        d = node.attrib.get("d")
        if not d:
            return
        _append_path_shapes(shapes, shape_groups, d, state)
        return
    if tag == "polyline":
        points = _parse_points_attr(node.attrib.get("points", ""))
        _append_polyline_shape(shapes, shape_groups, points, state, close=False)
        return
    if tag == "line":
        x1 = float(node.attrib.get("x1", "0"))
        y1 = float(node.attrib.get("y1", "0"))
        x2 = float(node.attrib.get("x2", "0"))
        y2 = float(node.attrib.get("y2", "0"))
        _append_polyline_shape(shapes, shape_groups, [(x1, y1), (x2, y2)], state, close=False)
        return
    if tag == "polygon":
        raise ValueError("stroke-first SVG parser does not support polygon elements; convert them to open paths")
    if tag in {"circle", "ellipse", "rect"}:
        raise ValueError(f"stroke-first SVG parser does not support '{tag}' elements")


def _walk(node: etree.Element, parent_state: _ParseState, shapes: list[object], shape_groups: list[ShapeGroup]) -> None:
    tag = remove_namespaces(node.tag)
    if tag in {"defs", "style", "filter"}:
        return
    if tag in {"linearGradient", "radialGradient"}:
        raise ValueError(f"stroke-first SVG parser does not support '{tag}' sections")
    state = _resolve_state(node, parent_state)
    if tag == "g" or tag == "svg":
        for child in node:
            _walk(child, state, shapes, shape_groups)
        return
    _parse_shape(node, state, shapes, shape_groups)


def parse_scene(node: etree.Element):
    canvas_width = -1
    canvas_height = -1
    if "viewBox" in node.attrib:
        values = node.attrib["viewBox"].replace(",", " ").split()
        if len(values) != 4:
            raise ValueError("viewBox must contain four values")
        canvas_width = parse_int(values[2])
        canvas_height = parse_int(values[3])
    else:
        if "width" not in node.attrib or "height" not in node.attrib:
            raise ValueError("stroke-first SVG parser requires explicit width/height or viewBox")
        canvas_width = parse_int(node.attrib["width"])
        canvas_height = parse_int(node.attrib["height"])

    shapes: list[object] = []
    shape_groups: list[ShapeGroup] = []
    _walk(node, _default_state(), shapes, shape_groups)
    return canvas_width, canvas_height, shapes, shape_groups


def svg_to_scene(filename):
    tree = etree.parse(filename)
    return parse_scene(tree.getroot())


__all__ = [
    "svg_to_scene",
    "parse_scene",
    "parse_transform",
    "parse_color",
    "parse_length",
    "parse_int",
]
