"""Plotter-oriented metrics for diffvg scenes and SVG files."""

from __future__ import annotations

import json
import shlex
import shutil
import statistics
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path as FsPath
from typing import Iterable, Optional

import torch

from ..parse_svg import svg_to_scene
from ..shape import ShapeGroup
from .geometry import polyline_length, sample_shape_polyline


@dataclass(frozen=True)
class PlotterStroke:
    shape_index: int
    group_index: int
    length: float
    width: float
    closed: bool
    color_key: str
    start: complex
    end: complex
    shape_type: str


@dataclass(frozen=True)
class PlotterMetrics:
    stroke_count: int
    closed_count: int
    color_count: int
    total_pen_down: float
    total_pen_up: float
    travel_ratio: float
    mean_stroke_length: float
    median_stroke_length: float
    min_stroke_length: float
    max_stroke_length: float
    mean_stroke_width: float
    median_stroke_width: float
    pct_lt_5: float
    pct_lt_10: float
    pct_lt_20: float
    pct_lt_50: float


@dataclass(frozen=True)
class PlotterMetricsRow:
    source: str
    mode: str
    metrics: PlotterMetrics


def _cloneable_color_key(color) -> str:
    if color is None:
        return "none"
    if isinstance(color, torch.Tensor):
        values = color.detach().cpu().flatten().tolist()
        return "rgba(" + ",".join(f"{value:.6f}" for value in values) + ")"
    return type(color).__name__


def _stroke_width(shape) -> float:
    if hasattr(shape, "stroke_width") and isinstance(shape.stroke_width, torch.Tensor):
        return 2.0 * float(shape.stroke_width.detach().cpu().item())
    return 0.0


def _shape_name(shape) -> str:
    return type(shape).__name__


def iter_scene_strokes(width: int, height: int, shapes: list[object], shape_groups: list[ShapeGroup]) -> list[PlotterStroke]:
    _ = (width, height)
    strokes: list[PlotterStroke] = []
    for group_index, group in enumerate(shape_groups):
        if getattr(group, "stroke_color", None) is None:
            continue
        transform = getattr(group, "shape_to_canvas", None)
        color_key = _cloneable_color_key(group.stroke_color)
        for shape_index_tensor in group.shape_ids.detach().cpu().tolist():
            shape_index = int(shape_index_tensor)
            if shape_index < 0 or shape_index >= len(shapes):
                continue
            shape = shapes[shape_index]
            sampled = sample_shape_polyline(shape, transform)
            if sampled is None or len(sampled.points) < 2:
                continue
            strokes.append(
                PlotterStroke(
                    shape_index=shape_index,
                    group_index=group_index,
                    length=polyline_length(sampled.points),
                    width=_stroke_width(shape),
                    closed=bool(sampled.closed),
                    color_key=color_key,
                    start=sampled.points[0],
                    end=sampled.points[-1],
                    shape_type=_shape_name(shape),
                )
            )
    return strokes


def summarize_strokes(strokes: Iterable[PlotterStroke]) -> PlotterMetrics:
    stroke_list = list(strokes)
    if not stroke_list:
        return PlotterMetrics(
            stroke_count=0,
            closed_count=0,
            color_count=0,
            total_pen_down=0.0,
            total_pen_up=0.0,
            travel_ratio=0.0,
            mean_stroke_length=0.0,
            median_stroke_length=0.0,
            min_stroke_length=0.0,
            max_stroke_length=0.0,
            mean_stroke_width=0.0,
            median_stroke_width=0.0,
            pct_lt_5=0.0,
            pct_lt_10=0.0,
            pct_lt_20=0.0,
            pct_lt_50=0.0,
        )

    lengths = [stroke.length for stroke in stroke_list]
    widths = [stroke.width for stroke in stroke_list]
    pen_up = 0.0
    for current, next_stroke in zip(stroke_list, stroke_list[1:]):
        pen_up += abs(next_stroke.start - current.end)

    def pct_lt(threshold: float) -> float:
        return 100.0 * sum(1 for length in lengths if length < threshold) / len(lengths)

    total_pen_down = sum(lengths)
    return PlotterMetrics(
        stroke_count=len(stroke_list),
        closed_count=sum(1 for stroke in stroke_list if stroke.closed),
        color_count=len({stroke.color_key for stroke in stroke_list}),
        total_pen_down=total_pen_down,
        total_pen_up=pen_up,
        travel_ratio=(pen_up / total_pen_down) if total_pen_down > 0.0 else 0.0,
        mean_stroke_length=statistics.mean(lengths),
        median_stroke_length=statistics.median(lengths),
        min_stroke_length=min(lengths),
        max_stroke_length=max(lengths),
        mean_stroke_width=statistics.mean(widths),
        median_stroke_width=statistics.median(widths),
        pct_lt_5=pct_lt(5.0),
        pct_lt_10=pct_lt(10.0),
        pct_lt_20=pct_lt(20.0),
        pct_lt_50=pct_lt(50.0),
    )


def analyze_scene(width: int, height: int, shapes: list[object], shape_groups: list[ShapeGroup]) -> PlotterMetrics:
    return summarize_strokes(iter_scene_strokes(width, height, shapes, shape_groups))


def analyze_svg(svg_path: str | FsPath) -> PlotterMetrics:
    scene = svg_to_scene(str(svg_path))
    return analyze_scene(*scene)


def run_vpype_pipeline(svg_path: str | FsPath, pipeline: str, output_dir: str | FsPath) -> FsPath:
    if shutil.which("vpype") is None:
        raise RuntimeError("vpype is not installed or not on PATH. Install it to use vpype comparison mode.")
    svg_path = FsPath(svg_path).resolve()
    output_dir = FsPath(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / svg_path.name
    cmd = ["vpype", "read", str(svg_path), *shlex.split(pipeline), "write", str(output_path)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return output_path


def analyze_svgs(
    svg_paths: Iterable[str | FsPath],
    *,
    vpype_pipeline: str | None = None,
    vpype_output_dir: str | FsPath | None = None,
) -> list[PlotterMetricsRow]:
    rows: list[PlotterMetricsRow] = []
    temp_dir: Optional[tempfile.TemporaryDirectory[str]] = None
    vpype_root: Optional[FsPath] = None
    if vpype_pipeline:
        if vpype_output_dir is None:
            temp_dir = tempfile.TemporaryDirectory(prefix="svg-plotter-metrics-", dir="/tmp")
            vpype_root = FsPath(temp_dir.name)
        else:
            vpype_root = FsPath(vpype_output_dir).resolve()
            vpype_root.mkdir(parents=True, exist_ok=True)
    try:
        for item in svg_paths:
            path = FsPath(item).resolve()
            rows.append(PlotterMetricsRow(source=str(path), mode="raw", metrics=analyze_svg(path)))
            if vpype_pipeline and vpype_root is not None:
                optimized = run_vpype_pipeline(path, vpype_pipeline, vpype_root)
                rows.append(PlotterMetricsRow(source=str(path), mode="vpype", metrics=analyze_svg(optimized)))
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()
    return rows


def format_metrics_table(rows: Iterable[PlotterMetricsRow], *, base_dir: FsPath | None = None) -> str:
    row_list = list(rows)
    headers = [
        "file",
        "mode",
        "strokes",
        "closed",
        "colors",
        "pen_down",
        "pen_up",
        "travel/down",
        "median_len",
        "mean_len",
        "%<10",
        "%<20",
        "%<50",
        "median_w",
    ]
    values: list[list[str]] = []
    for row in row_list:
        try:
            source = str(FsPath(row.source).resolve().relative_to(base_dir or FsPath.cwd()))
        except ValueError:
            source = row.source
        metrics = row.metrics
        values.append(
            [
                source,
                row.mode,
                str(metrics.stroke_count),
                str(metrics.closed_count),
                str(metrics.color_count),
                f"{metrics.total_pen_down:.1f}",
                f"{metrics.total_pen_up:.1f}",
                f"{metrics.travel_ratio:.3f}",
                f"{metrics.median_stroke_length:.1f}",
                f"{metrics.mean_stroke_length:.1f}",
                f"{metrics.pct_lt_10:.2f}",
                f"{metrics.pct_lt_20:.2f}",
                f"{metrics.pct_lt_50:.2f}",
                f"{metrics.median_stroke_width:.2f}",
            ]
        )
    widths = [len(header) for header in headers]
    for row in values:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))
    lines = [
        "  ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)),
        "  ".join("-" * widths[idx] for idx in range(len(headers))),
    ]
    for row in values:
        lines.append("  ".join(value.ljust(widths[idx]) for idx, value in enumerate(row)))
    return "\n".join(lines)


def metrics_rows_to_json(rows: Iterable[PlotterMetricsRow]) -> str:
    payload = [
        {
            "source": row.source,
            "mode": row.mode,
            "metrics": asdict(row.metrics),
        }
        for row in rows
    ]
    return json.dumps(payload, indent=2) + "\n"


__all__ = [
    "PlotterMetrics",
    "PlotterMetricsRow",
    "PlotterStroke",
    "analyze_scene",
    "analyze_svg",
    "analyze_svgs",
    "format_metrics_table",
    "iter_scene_strokes",
    "metrics_rows_to_json",
    "run_vpype_pipeline",
    "summarize_strokes",
]
