"""Conservative plotter-aware cleanup for diffvg scenes."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path as FsPath
from typing import Optional

import torch

from ..parse_svg import svg_to_scene
from ..save_svg import save_svg
from ..shape import Circle, Ellipse, Path as DiffvgPath, Polygon, Rect, ShapeGroup
from .geometry import (
    concatenate_paths,
    is_identity_transform,
    path_endpoints,
    path_length,
    path_tangent,
    reverse_path,
)
from .metrics import PlotterMetrics, analyze_scene


@dataclass(frozen=True)
class PlotterCleanupConfig:
    min_stroke_length_px: float = 10.0
    merge_distance_px: float = 3.0
    merge_angle_deg: float = 18.0
    width_tolerance_px: float = 0.25
    reorder: bool = True


@dataclass(frozen=True)
class PlotterCleanupStats:
    eligible_groups: int
    skipped_groups: int
    pruned_paths: int
    stitched_joins: int
    reordered_buckets: int
    reoriented_paths: int


@dataclass(frozen=True)
class PlotterCleanupResult:
    width: int
    height: int
    shapes: list[object]
    shape_groups: list[ShapeGroup]
    before_metrics: PlotterMetrics
    after_metrics: PlotterMetrics
    stats: PlotterCleanupStats

    def to_report(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "before_metrics": asdict(self.before_metrics),
            "after_metrics": asdict(self.after_metrics),
            "stats": asdict(self.stats),
        }


@dataclass
class _CandidateStroke:
    path: DiffvgPath
    group: ShapeGroup
    color_key: str
    original_index: int


@dataclass
class _CleanupState:
    eligible_groups: int = 0
    skipped_groups: int = 0
    pruned_paths: int = 0
    stitched_joins: int = 0
    reordered_buckets: int = 0
    reoriented_paths: int = 0


def _clone_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.detach().clone()


def _clone_color(color):
    if isinstance(color, torch.Tensor):
        return _clone_tensor(color)
    return color


def _clone_shape(shape):
    if isinstance(shape, DiffvgPath):
        return DiffvgPath(
            num_control_points=_clone_tensor(shape.num_control_points),
            points=_clone_tensor(shape.points),
            is_closed=bool(shape.is_closed),
            stroke_width=_clone_tensor(shape.stroke_width),
            id=shape.id,
            use_distance_approx=bool(shape.use_distance_approx),
        )
    if isinstance(shape, Polygon):
        return Polygon(
            points=_clone_tensor(shape.points),
            is_closed=bool(shape.is_closed),
            stroke_width=_clone_tensor(shape.stroke_width),
            id=shape.id,
        )
    if isinstance(shape, Circle):
        return Circle(
            radius=_clone_tensor(shape.radius),
            center=_clone_tensor(shape.center),
            stroke_width=_clone_tensor(shape.stroke_width),
            id=shape.id,
        )
    if isinstance(shape, Ellipse):
        return Ellipse(
            radius=_clone_tensor(shape.radius),
            center=_clone_tensor(shape.center),
            stroke_width=_clone_tensor(shape.stroke_width),
            id=shape.id,
        )
    if isinstance(shape, Rect):
        return Rect(
            p_min=_clone_tensor(shape.p_min),
            p_max=_clone_tensor(shape.p_max),
            stroke_width=_clone_tensor(shape.stroke_width),
            id=shape.id,
        )
    raise TypeError(f"Unsupported shape type '{type(shape).__name__}'")


def _clone_group(group: ShapeGroup, shape_ids: list[int]) -> ShapeGroup:
    return ShapeGroup(
        shape_ids=torch.tensor(shape_ids, dtype=torch.int32, device=group.shape_ids.device),
        fill_color=_clone_color(group.fill_color),
        use_even_odd_rule=bool(group.use_even_odd_rule),
        stroke_color=_clone_color(group.stroke_color),
        shape_to_canvas=_clone_tensor(group.shape_to_canvas),
        id=group.id,
    )


def _color_key(group: ShapeGroup) -> str:
    color = group.stroke_color
    if isinstance(color, torch.Tensor):
        values = color.detach().cpu().flatten().tolist()
        return "rgba(" + ",".join(f"{value:.6f}" for value in values) + ")"
    return type(color).__name__


def _svg_width(path: DiffvgPath) -> float:
    return 2.0 * float(path.stroke_width.detach().cpu().item())


def _is_candidate(group: ShapeGroup, shape) -> bool:
    if len(group.shape_ids.detach().cpu().tolist()) != 1:
        return False
    if not isinstance(shape, DiffvgPath):
        return False
    if bool(shape.is_closed):
        return False
    if group.fill_color is not None:
        return False
    if not isinstance(group.stroke_color, torch.Tensor):
        return False
    if group.stroke_color.ndim != 1 or int(group.stroke_color.numel()) != 4:
        return False
    if not is_identity_transform(group.shape_to_canvas):
        return False
    return True


def _append_group_copy(out_shapes: list[object], out_groups: list[ShapeGroup], shapes: list[object], group: ShapeGroup) -> None:
    new_ids: list[int] = []
    for shape_index in group.shape_ids.detach().cpu().tolist():
        shape = _clone_shape(shapes[int(shape_index)])
        new_ids.append(len(out_shapes))
        out_shapes.append(shape)
    out_groups.append(_clone_group(group, new_ids))


def _candidate_length(item: _CandidateStroke) -> float:
    return path_length(item.path)


def _best_start_orientation(items: list[_CandidateStroke]) -> tuple[_CandidateStroke, bool]:
    if len(items) == 1:
        return items[0], False
    best_idx = max(range(len(items)), key=lambda idx: (_candidate_length(items[idx]), -items[idx].original_index))
    item = items[best_idx]
    normal_path = item.path
    reversed_path = reverse_path(item.path)

    def nearest_from(path: DiffvgPath) -> float:
        _, tail = path_endpoints(path)
        best = math.inf
        for idx, other in enumerate(items):
            if idx == best_idx:
                continue
            other_start, other_end = path_endpoints(other.path)
            best = min(best, abs(tail - other_start), abs(tail - other_end))
        return best

    return item, nearest_from(reversed_path) + 1e-6 < nearest_from(normal_path)


def _reorder_bucket(items: list[_CandidateStroke], state: _CleanupState, config: PlotterCleanupConfig) -> list[_CandidateStroke]:
    if not config.reorder or len(items) <= 1:
        return items

    state.reordered_buckets += 1
    remaining = list(items)
    seed, reverse_seed = _best_start_orientation(remaining)
    if reverse_seed:
        seed = _CandidateStroke(path=reverse_path(seed.path), group=seed.group, color_key=seed.color_key, original_index=seed.original_index)
        state.reoriented_paths += 1
    ordered = [seed]
    remaining = [item for item in remaining if item.original_index != seed.original_index]

    while remaining:
        _, tail = path_endpoints(ordered[-1].path)
        best_idx = 0
        best_reverse = False
        best_distance = math.inf
        for idx, item in enumerate(remaining):
            start, _ = path_endpoints(item.path)
            rev_item = reverse_path(item.path)
            rev_start, _ = path_endpoints(rev_item)
            normal_dist = abs(tail - start)
            reverse_dist = abs(tail - rev_start)
            if reverse_dist + 1e-6 < normal_dist:
                dist = reverse_dist
                reverse = True
            else:
                dist = normal_dist
                reverse = False
            if dist + 1e-6 < best_distance or (abs(dist - best_distance) <= 1e-6 and item.original_index < remaining[best_idx].original_index):
                best_idx = idx
                best_reverse = reverse
                best_distance = dist
        next_item = remaining.pop(best_idx)
        if best_reverse:
            next_item = _CandidateStroke(
                path=reverse_path(next_item.path),
                group=next_item.group,
                color_key=next_item.color_key,
                original_index=next_item.original_index,
            )
            state.reoriented_paths += 1
        ordered.append(next_item)
    return ordered


def _tangent_angle_deg(a: complex, b: complex) -> float:
    if abs(a) <= 1e-6 or abs(b) <= 1e-6:
        return 180.0
    dot = max(-1.0, min(1.0, (a.real * b.real + a.imag * b.imag)))
    return math.degrees(math.acos(dot))


def _can_stitch(current: _CandidateStroke, next_item: _CandidateStroke, config: PlotterCleanupConfig) -> bool:
    current_width = _svg_width(current.path)
    next_width = _svg_width(next_item.path)
    if abs(current_width - next_width) > config.width_tolerance_px:
        return False
    _, current_end = path_endpoints(current.path)
    next_start, _ = path_endpoints(next_item.path)
    if abs(next_start - current_end) > config.merge_distance_px:
        return False
    tail_tangent = path_tangent(current.path, head=False)
    head_tangent = path_tangent(next_item.path, head=True)
    return _tangent_angle_deg(tail_tangent, head_tangent) <= config.merge_angle_deg


def _merged_candidate_path(current: _CandidateStroke, next_item: _CandidateStroke) -> DiffvgPath:
    merged_id = current.path.id or next_item.path.id or current.group.id or next_item.group.id or "plotter_path"
    return concatenate_paths(
        [current.path, next_item.path],
        path_id=merged_id,
        use_distance_approx=bool(current.path.use_distance_approx or next_item.path.use_distance_approx),
    )


def _stitch_bucket(items: list[_CandidateStroke], state: _CleanupState, config: PlotterCleanupConfig) -> list[_CandidateStroke]:
    if not items:
        return []
    stitched: list[_CandidateStroke] = [items[0]]
    for item in items[1:]:
        current = stitched[-1]
        if _can_stitch(current, item, config):
            stitched[-1] = _CandidateStroke(
                path=_merged_candidate_path(current, item),
                group=current.group,
                color_key=current.color_key,
                original_index=current.original_index,
            )
            state.stitched_joins += 1
        else:
            stitched.append(item)
    return stitched


def _cleanup_bucket(items: list[_CandidateStroke], state: _CleanupState, config: PlotterCleanupConfig) -> list[_CandidateStroke]:
    filtered: list[_CandidateStroke] = []
    for item in items:
        if _candidate_length(item) < config.min_stroke_length_px:
            state.pruned_paths += 1
            continue
        filtered.append(item)
    if len(filtered) <= 1:
        return filtered
    reordered = _reorder_bucket(filtered, state, config)
    return _stitch_bucket(reordered, state, config)


def _flush_window(items: list[_CandidateStroke], out_shapes: list[object], out_groups: list[ShapeGroup], state: _CleanupState, config: PlotterCleanupConfig) -> None:
    if not items:
        return
    bucket_order: list[str] = []
    buckets: dict[str, list[_CandidateStroke]] = {}
    for item in items:
        bucket = item.color_key
        if bucket not in buckets:
            buckets[bucket] = []
            bucket_order.append(bucket)
        buckets[bucket].append(item)
    for bucket in bucket_order:
        for item in _cleanup_bucket(buckets[bucket], state, config):
            shape_index = len(out_shapes)
            out_shapes.append(item.path)
            out_groups.append(_clone_group(item.group, [shape_index]))


def cleanup_scene(
    width: int,
    height: int,
    shapes: list[object],
    shape_groups: list[ShapeGroup],
    config: PlotterCleanupConfig | None = None,
) -> PlotterCleanupResult:
    cfg = config or PlotterCleanupConfig()
    before_metrics = analyze_scene(width, height, shapes, shape_groups)
    out_shapes: list[object] = []
    out_groups: list[ShapeGroup] = []
    state = _CleanupState()
    window: list[_CandidateStroke] = []

    for group_index, group in enumerate(shape_groups):
        shape_ids = group.shape_ids.detach().cpu().tolist()
        if len(shape_ids) != 1:
            _flush_window(window, out_shapes, out_groups, state, cfg)
            window = []
            _append_group_copy(out_shapes, out_groups, shapes, group)
            state.skipped_groups += 1
            continue
        shape = shapes[int(shape_ids[0])]
        if not _is_candidate(group, shape):
            _flush_window(window, out_shapes, out_groups, state, cfg)
            window = []
            _append_group_copy(out_shapes, out_groups, shapes, group)
            state.skipped_groups += 1
            continue
        state.eligible_groups += 1
        window.append(
            _CandidateStroke(
                path=_clone_shape(shape),
                group=_clone_group(group, [0]),
                color_key=_color_key(group),
                original_index=group_index,
            )
        )
    _flush_window(window, out_shapes, out_groups, state, cfg)

    after_metrics = analyze_scene(width, height, out_shapes, out_groups)
    return PlotterCleanupResult(
        width=width,
        height=height,
        shapes=out_shapes,
        shape_groups=out_groups,
        before_metrics=before_metrics,
        after_metrics=after_metrics,
        stats=PlotterCleanupStats(
            eligible_groups=state.eligible_groups,
            skipped_groups=state.skipped_groups,
            pruned_paths=state.pruned_paths,
            stitched_joins=state.stitched_joins,
            reordered_buckets=state.reordered_buckets,
            reoriented_paths=state.reoriented_paths,
        ),
    )


def cleanup_svg(
    input_svg: str | FsPath,
    output_svg: str | FsPath,
    *,
    config: PlotterCleanupConfig | None = None,
    background_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> PlotterCleanupResult:
    width, height, shapes, shape_groups = svg_to_scene(str(input_svg))
    result = cleanup_scene(width, height, shapes, shape_groups, config=config)
    save_svg(
        str(output_svg),
        result.width,
        result.height,
        result.shapes,
        result.shape_groups,
        use_gamma=False,
        background_rgb=background_rgb,
    )
    return result


__all__ = [
    "PlotterCleanupConfig",
    "PlotterCleanupResult",
    "PlotterCleanupStats",
    "cleanup_scene",
    "cleanup_svg",
]
