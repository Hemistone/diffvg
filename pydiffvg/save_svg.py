"""Write stroke-first pydiffvg scenes to an SVG file.

The maintained runtime only supports open-stroke scenes, so SVG export follows
that same contract and rejects filled or gradient-based content explicitly.
"""

from __future__ import annotations

from typing import List
import xml.etree.ElementTree as etree
from xml.dom import minidom

import torch

import pydiffvg


def prettify(elem) -> str:
    """Return a pretty-printed XML string for the Element."""
    rough_string = etree.tostring(elem, "utf-8")
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def _format_rgb(color: torch.Tensor) -> str:
    rgba = color.detach().to(device="cpu", dtype=torch.float32).reshape(4)
    if not torch.isfinite(rgba).all():
        raise ValueError("save_svg only supports finite RGBA stroke colors")
    rgba = rgba.clamp(0.0, 1.0)
    return f"rgb({int(255 * rgba[0])}, {int(255 * rgba[1])}, {int(255 * rgba[2])})"


def _finite_scalar(value: torch.Tensor, *, what: str) -> float:
    scalar = value.detach().to(device="cpu", dtype=torch.float32).reshape(())
    if not torch.isfinite(scalar):
        raise ValueError(f"save_svg only supports finite {what}")
    return float(scalar.item())


def _finite_points(points: torch.Tensor, *, what: str) -> torch.Tensor:
    data = points.detach().to(device="cpu", dtype=torch.float32)
    if not torch.isfinite(data).all():
        raise ValueError(f"save_svg only supports finite {what}")
    return data


def _path_d(shape: object) -> str:
    if isinstance(shape, pydiffvg.Polygon):
        if bool(shape.is_closed):
            raise ValueError("save_svg only supports open polygons")
        points = _finite_points(shape.points, what="polygon points")
        if points.shape[0] < 2:
            raise ValueError("open polygon must contain at least two points")
        path_str = f"M {points[0, 0].item()} {points[0, 1].item()}"
        for index in range(1, points.shape[0]):
            path_str += f" L {points[index, 0].item()} {points[index, 1].item()}"
        return path_str

    if not isinstance(shape, pydiffvg.Path):
        raise ValueError("save_svg only supports Path and Polygon shapes")
    if bool(shape.is_closed):
        raise ValueError("save_svg only supports open paths")

    num_segments = int(shape.num_control_points.shape[0])
    num_control_points = shape.num_control_points.detach().to(device="cpu", dtype=torch.int64).tolist()
    points = _finite_points(shape.points, what="path points")
    if points.shape[0] == 0:
        raise ValueError("open path must contain at least one point")
    num_points = int(points.shape[0])
    path_str = f"M {points[0, 0].item()} {points[0, 1].item()}"
    point_id = 1
    for segment_index in range(num_segments):
        order = num_control_points[segment_index]
        if order == 0:
            p = point_id % num_points
            path_str += f" L {points[p, 0].item()} {points[p, 1].item()}"
            point_id += 1
        elif order == 1:
            p1 = (point_id + 1) % num_points
            path_str += (
                f" Q {points[point_id, 0].item()} {points[point_id, 1].item()}"
                f" {points[p1, 0].item()} {points[p1, 1].item()}"
            )
            point_id += 2
        elif order == 2:
            p2 = (point_id + 2) % num_points
            path_str += (
                f" C {points[point_id, 0].item()} {points[point_id, 1].item()}"
                f" {points[point_id + 1, 0].item()} {points[point_id + 1, 1].item()}"
                f" {points[p2, 0].item()} {points[p2, 1].item()}"
            )
            point_id += 3
        else:
            raise ValueError("save_svg only supports line, quadratic, and cubic segments")
    return path_str


def save_svg(
    filename: str,
    width: int,
    height: int,
    shapes: List[object],
    shape_groups: List[object],
    use_gamma: bool = False,
    background_rgb: tuple[float, float, float] | None = None,
) -> None:
    root = etree.Element("svg")
    root.set("version", "1.1")
    root.set("xmlns", "http://www.w3.org/2000/svg")
    root.set("width", str(width))
    root.set("height", str(height))
    defs = etree.SubElement(root, "defs")
    g = etree.SubElement(root, "g")

    # Stroke-first SVG export intentionally omits background geometry even when
    # a viewer background color is requested, because plotter-oriented scenes
    # must remain stroke-only on round-trip.

    if use_gamma:
        flt = etree.SubElement(defs, "filter")
        flt.set("id", "gamma")
        flt.set("x", "0")
        flt.set("y", "0")
        flt.set("width", "100%")
        flt.set("height", "100%")
        gamma = etree.SubElement(flt, "feComponentTransfer")
        gamma.set("color-interpolation-filters", "sRGB")
        for channel in ("R", "G", "B", "A"):
            func = etree.SubElement(gamma, f"feFunc{channel}")
            func.set("type", "gamma")
            func.set("amplitude", "1")
            func.set("exponent", str(1 / 2.2))
        g.set("style", "filter:url(#gamma)")

    for group_index, shape_group in enumerate(shape_groups):
        if getattr(shape_group, "fill_color", None) is not None:
            raise ValueError("save_svg only supports stroke-only scenes")
        stroke_color = getattr(shape_group, "stroke_color", None)
        if not isinstance(stroke_color, torch.Tensor) or stroke_color.numel() != 4:
            raise ValueError("save_svg only supports constant RGBA stroke colors")
        shape_ids = shape_group.shape_ids.detach().to(dtype=torch.int64, device="cpu")
        if int(shape_ids.numel()) != 1:
            raise ValueError("save_svg expects one shape per ShapeGroup in stroke-first mode")
        shape = shapes[int(shape_ids.item())]
        stroke_width = getattr(shape, "stroke_width", None)
        if not isinstance(stroke_width, torch.Tensor) or stroke_width.numel() != 1:
            raise ValueError("save_svg only supports scalar stroke widths")
        stroke_width_value = _finite_scalar(stroke_width, what="stroke widths")
        if stroke_width_value < 0.0:
            raise ValueError("save_svg only supports non-negative stroke widths")
        stroke_alpha = _finite_scalar(stroke_color.reshape(4)[3], what="stroke alpha")
        stroke_alpha = min(max(stroke_alpha, 0.0), 1.0)

        shape_node = etree.SubElement(g, "path")
        shape_node.set("d", _path_d(shape))
        shape_node.set("fill", "none")
        shape_node.set("stroke", _format_rgb(stroke_color))
        shape_node.set("stroke-opacity", str(stroke_alpha))
        shape_node.set("stroke-width", str(2.0 * stroke_width_value))
        shape_node.set("stroke-linecap", "round")
        shape_node.set("stroke-linejoin", "round")
        shape_node.set("id", f"shape_{group_index}")

    with open(filename, "w", encoding="utf-8") as handle:
        handle.write(prettify(root))


__all__ = ["save_svg"]
