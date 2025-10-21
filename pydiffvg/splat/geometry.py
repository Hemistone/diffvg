from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import diffvg
import torch

from .types import (
    FillSpec,
    PathSpec,
    ScenePayload,
    SegmentData,
    _SplatUnsupported,
)


def _build_segments(points: torch.Tensor, control_counts: List[int]) -> List[SegmentData]:
    segments: List[SegmentData] = []
    idx = 0
    current = points[idx]
    idx += 1
    for cp in control_counts:
        controls: List[torch.Tensor] = []
        for _ in range(cp):
            controls.append(points[idx])
            idx += 1
        end = points[idx]
        idx += 1
        segments.append(SegmentData(start=current, controls=tuple(controls), end=end))
        current = end
    return segments


def _gather_specs(
    scene: ScenePayload, device: torch.device, dtype: torch.dtype
) -> Tuple[List[PathSpec], List[FillSpec]]:
    if scene.non_path_shapes:
        raise _SplatUnsupported("non-path shapes are not handled yet")

    identity = torch.eye(3, device=device, dtype=dtype)
    stroke_colors: Dict[int, torch.Tensor] = {}
    fill_colors: Dict[int, torch.Tensor] = {}
    for group in scene.shape_groups:
        transform = group.shape_to_canvas.to(device=device, dtype=dtype)
        if not torch.allclose(transform, identity, atol=1e-6):
            raise _SplatUnsupported("shape_to_canvas transforms are not supported yet")

        if group.fill.color_type is not None:
            if group.fill.color_type != diffvg.ColorType.constant:
                raise _SplatUnsupported("only constant fill colors are supported")
            color = group.fill.params[0].to(device=device, dtype=dtype)
            if color.shape[-1] != 4:
                raise _SplatUnsupported("fill colors must include RGBA channels")
            for sid in group.shape_ids.to(torch.int64).tolist():
                fill_colors[sid] = color

        stroke = group.stroke
        if stroke.color_type is not None:
            if stroke.color_type != diffvg.ColorType.constant:
                raise _SplatUnsupported("only constant stroke colors are supported")
            color = stroke.params[0].to(device=device, dtype=dtype)
            if color.shape[-1] != 4:
                raise _SplatUnsupported("stroke colors must include RGBA channels")
            for sid in group.shape_ids.to(torch.int64).tolist():
                stroke_colors[sid] = color

    if not scene.paths:
        raise _SplatUnsupported("scene contains no path primitives")

    stroke_specs: List[PathSpec] = []
    fill_specs: List[FillSpec] = []
    for payload in scene.paths:
        if payload.thickness is not None:
            raise _SplatUnsupported("per-point thickness is not supported yet")
        if payload.use_distance_approx:
            raise _SplatUnsupported("distance approximation paths are not supported")

        points = payload.points.to(device=device, dtype=dtype)
        num_control_points = payload.num_control_points.to(torch.int64).tolist()
        segments = _build_segments(points, num_control_points)

        if payload.shape_id in stroke_colors:
            if payload.is_closed:
                raise _SplatUnsupported("strokes on closed paths are not supported yet")
            stroke_width = payload.stroke_width.to(device=device, dtype=dtype)
            if stroke_width.numel() != 1:
                raise _SplatUnsupported("per-segment stroke width is not supported yet")
            color_rgba = stroke_colors[payload.shape_id]
            stroke_specs.append(
                PathSpec(
                    shape_id=payload.shape_id,
                    segments=segments,
                    stroke_width=stroke_width.reshape(1),
                    color_rgb=color_rgba[:3],
                    opacity=color_rgba[3:4],
                )
            )

        if payload.is_closed:
            if payload.shape_id not in fill_colors:
                continue
            color_rgba = fill_colors[payload.shape_id]
            fill_specs.append(
                FillSpec(
                    shape_id=payload.shape_id,
                    segments=segments,
                    color_rgb=color_rgba[:3],
                    opacity=color_rgba[3:4],
                )
            )
        else:
            if payload.shape_id in fill_colors:
                raise _SplatUnsupported("fill colors require closed paths")

    if not stroke_specs and not fill_specs:
        raise _SplatUnsupported("scene does not contain supported path primitives")

    return stroke_specs, fill_specs


def _segment_samples(
    num_samples: int,
    device: torch.device,
    dtype: torch.dtype,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    if num_samples <= 0:
        return torch.empty(0, device=device, dtype=dtype)
    bins = torch.arange(num_samples, device=device, dtype=dtype)
    if generator is None:
        offsets = torch.full((num_samples,), 0.5, device=device, dtype=dtype)
    else:
        offsets = torch.rand(num_samples, device=device, dtype=dtype, generator=generator)
    return (bins + offsets) / float(num_samples)


def _evaluate_segment(segment: SegmentData, t_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    start = segment.start
    controls = segment.controls
    end = segment.end
    t = t_values.unsqueeze(-1)
    omt = 1.0 - t

    if len(controls) == 0:
        pos = omt * start + t * end
        tangent = end - start
        tangent = tangent.unsqueeze(0).expand_as(pos)
    elif len(controls) == 1:
        ctrl = controls[0]
        pos = omt * omt * start + 2.0 * omt * t * ctrl + t * t * end
        tangent = 2.0 * omt * (ctrl - start) + 2.0 * t * (end - ctrl)
    elif len(controls) == 2:
        ctrl1, ctrl2 = controls
        omt2 = omt * omt
        t2 = t * t
        pos = (
            omt2 * omt * start
            + 3.0 * omt2 * t * ctrl1
            + 3.0 * omt * t2 * ctrl2
            + t2 * t * end
        )
        tangent = (
            3.0 * omt2 * (ctrl1 - start)
            + 6.0 * omt * t * (ctrl2 - ctrl1)
            + 3.0 * t2 * (end - ctrl2)
        )
    else:
        raise _SplatUnsupported("unsupported Bézier degree (expected line/quadratic/cubic)")
    return pos, tangent


def _segment_arclength(segment: SegmentData, device: torch.device, dtype: torch.dtype, samples: int = 8) -> torch.Tensor:
    samples = max(int(samples), 1)
    t = torch.linspace(0.0, 1.0, steps=samples + 1, device=device, dtype=dtype)
    pos, _ = _evaluate_segment(segment, t)
    diffs = pos[1:] - pos[:-1]
    dist = torch.linalg.norm(diffs, dim=1)
    return dist.sum()


__all__ = [
    "_build_segments",
    "_gather_specs",
    "_segment_samples",
    "_evaluate_segment",
    "_segment_arclength",
]
