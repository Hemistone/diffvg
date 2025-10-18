from __future__ import annotations

import warnings
from dataclasses import dataclass
import os
import sys
import atexit
from typing import Dict, Iterable, List, Optional, Tuple
from types import SimpleNamespace

import diffvg
import torch

from .backend import DepthPolicy, SplatConfig, get_backend_config
from .device import get_device
from .render_pytorch import OutputType, BaselineRenderFunction as _BaselineRF
from .serialization import serialize_scene as _serialize_scene


@dataclass(frozen=True)
class PaintPayload:
    color_type: Optional[diffvg.ColorType]
    params: Tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class PathPayload:
    shape_id: int
    num_control_points: torch.Tensor
    points: torch.Tensor
    thickness: Optional[torch.Tensor]
    is_closed: bool
    use_distance_approx: bool
    stroke_width: torch.Tensor


@dataclass(frozen=True)
class NonPathShapePayload:
    shape_id: int
    shape_type: diffvg.ShapeType
    tensors: Tuple[torch.Tensor, ...]
    stroke_width: torch.Tensor


@dataclass(frozen=True)
class ShapeGroupPayload:
    shape_ids: torch.Tensor
    fill: PaintPayload
    stroke: PaintPayload
    use_even_odd_rule: bool
    shape_to_canvas: torch.Tensor


@dataclass(frozen=True)
class ScenePayload:
    canvas_width: int
    canvas_height: int
    output_type: Optional[int]
    use_prefiltering: bool
    eval_positions: torch.Tensor
    paths: List[PathPayload]
    non_path_shapes: List[NonPathShapePayload]
    shape_groups: List[ShapeGroupPayload]
    filter_type: Optional[diffvg.FilterType]
    filter_radius: torch.Tensor


@dataclass(frozen=True)
class RenderRequest:
    width: int
    height: int
    num_samples_x: int
    num_samples_y: int
    seed: int
    background_image: Optional[torch.Tensor]
    config: SplatConfig
    scene: ScenePayload
    generator: Optional[torch.Generator]


@dataclass(frozen=True)
class SegmentData:
    start: torch.Tensor
    controls: Tuple[torch.Tensor, ...]
    end: torch.Tensor


@dataclass(frozen=True)
class PathSpec:
    shape_id: int
    segments: List[SegmentData]
    stroke_width: torch.Tensor
    color_rgb: torch.Tensor
    opacity: torch.Tensor


@dataclass(frozen=True)
class FillSpec:
    shape_id: int
    segments: List[SegmentData]
    color_rgb: torch.Tensor
    opacity: torch.Tensor


@dataclass(frozen=True)
class GaussianBatch:
    mu: torch.Tensor
    theta: torch.Tensor
    sigma_x: torch.Tensor
    sigma_y: torch.Tensor
    color_rgb: torch.Tensor
    opacity: torch.Tensor


@dataclass(frozen=True)
class GradSlot:
    arg_index: int
    tensor: torch.Tensor


class _SplatUnsupported(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _parse_paint(args: Iterable[object], start_idx: int) -> Tuple[PaintPayload, int]:
    idx = start_idx
    payload_type = args[idx]
    idx += 1
    if payload_type is None:
        return PaintPayload(None, ()), idx
    if payload_type == diffvg.ColorType.constant:
        color = args[idx]
        idx += 1
        return PaintPayload(payload_type, (color,)), idx
    if payload_type == diffvg.ColorType.linear_gradient:
        begin = args[idx]
        idx += 1
        end = args[idx]
        idx += 1
        offsets = args[idx]
        idx += 1
        stop_colors = args[idx]
        idx += 1
        return PaintPayload(payload_type, (begin, end, offsets, stop_colors)), idx
    if payload_type == diffvg.ColorType.radial_gradient:
        center = args[idx]
        idx += 1
        radius = args[idx]
        idx += 1
        offsets = args[idx]
        idx += 1
        stop_colors = args[idx]
        idx += 1
        return PaintPayload(payload_type, (center, radius, offsets, stop_colors)), idx
    raise ValueError(f"Unsupported paint payload type: {payload_type}")


def _deserialize_scene(args: Iterable[object]) -> ScenePayload:
    idx = 0
    canvas_width = int(args[idx])
    idx += 1
    canvas_height = int(args[idx])
    idx += 1
    num_shapes = int(args[idx])
    idx += 1
    num_shape_groups = int(args[idx])
    idx += 1
    output_type = args[idx]
    idx += 1
    use_prefiltering = bool(args[idx])
    idx += 1
    eval_positions = args[idx]
    idx += 1

    paths: List[PathPayload] = []
    non_path_shapes: List[NonPathShapePayload] = []

    for shape_id in range(num_shapes):
        shape_type = args[idx]
        idx += 1
        if shape_type == diffvg.ShapeType.path:
            num_control_points = args[idx]
            idx += 1
            points = args[idx]
            idx += 1
            thickness = args[idx]
            idx += 1
            is_closed = bool(args[idx])
            idx += 1
            use_distance_approx = bool(args[idx])
            idx += 1
            stroke_width = args[idx]
            idx += 1
            paths.append(
                PathPayload(
                    shape_id=shape_id,
                    num_control_points=num_control_points,
                    points=points,
                    thickness=thickness,
                    is_closed=is_closed,
                    use_distance_approx=use_distance_approx,
                    stroke_width=stroke_width,
                )
            )
            continue
        if shape_type == diffvg.ShapeType.circle:
            radius = args[idx]
            idx += 1
            center = args[idx]
            idx += 1
            shape_tensors = (radius, center)
        elif shape_type == diffvg.ShapeType.ellipse:
            radius = args[idx]
            idx += 1
            center = args[idx]
            idx += 1
            shape_tensors = (radius, center)
        elif shape_type == diffvg.ShapeType.rect:
            p_min = args[idx]
            idx += 1
            p_max = args[idx]
            idx += 1
            shape_tensors = (p_min, p_max)
        else:
            raise ValueError(f"Unsupported shape type in splat backend: {shape_type}")
        stroke_width = args[idx]
        idx += 1
        non_path_shapes.append(
            NonPathShapePayload(
                shape_id=shape_id,
                shape_type=shape_type,
                tensors=shape_tensors,
                stroke_width=stroke_width,
            )
        )

    shape_groups: List[ShapeGroupPayload] = []
    for _ in range(num_shape_groups):
        shape_ids = args[idx]
        idx += 1
        fill_payload, idx = _parse_paint(args, idx)
        stroke_payload, idx = _parse_paint(args, idx)
        use_even_odd_rule = bool(args[idx])
        idx += 1
        shape_to_canvas = args[idx]
        idx += 1
        shape_groups.append(
            ShapeGroupPayload(
                shape_ids=shape_ids,
                fill=fill_payload,
                stroke=stroke_payload,
                use_even_odd_rule=use_even_odd_rule,
                shape_to_canvas=shape_to_canvas,
            )
        )

    filter_type = args[idx]
    idx += 1
    filter_radius = args[idx]

    return ScenePayload(
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        output_type=output_type,
        use_prefiltering=use_prefiltering,
        eval_positions=eval_positions,
        paths=paths,
        non_path_shapes=non_path_shapes,
        shape_groups=shape_groups,
        filter_type=filter_type,
        filter_radius=filter_radius,
    )


def _prepare_render_request(
    width: int,
    height: int,
    num_samples_x: int,
    num_samples_y: int,
    seed: int,
    background_image: Optional[torch.Tensor],
    args: Iterable[object],
) -> RenderRequest:
    config = get_backend_config("splat") or SplatConfig()
    scene = _deserialize_scene(args)
    device = get_device()
    generator: Optional[torch.Generator]
    if seed is None:
        generator = None
    else:
        if device.type == "cuda":
            generator = torch.Generator(device=device)
        else:
            generator = torch.Generator()
        generator.manual_seed(int(seed) & 0xFFFFFFFFFFFFFFFF)
    return RenderRequest(
        width=width,
        height=height,
        num_samples_x=num_samples_x,
        num_samples_y=num_samples_y,
        seed=seed,
        background_image=background_image,
        config=config,
        scene=scene,
        generator=generator,
    )


def _warn_fallback(reason: str) -> None:
    key = reason or "<unspecified>"
    seen = getattr(_warn_fallback, "_seen", set())
    if key in seen:
        return
    global _TRACE_FALLBACK
    _TRACE_FALLBACK += 1
    _trace(f"fallback to baseline (reason: {reason})")
    message = (
        "Bézier Splatting backend falling back to baseline renderer"
        f" (reason: {reason}). See docs/bezier_splatting_todo.md for progress."
    )
    warnings.warn(message, RuntimeWarning, stacklevel=3)
    seen.add(key)
    setattr(_warn_fallback, "_seen", seen)


def _trace_settings() -> tuple[bool, int | None]:
    """Parse DIFFVG_SPLAT_TRACE into (enabled, limit).

    Accepted values:
    - unset/"", "0", "false", "no", "off" => disabled
    - "1", "true", "yes", "on" => enabled with default limit (5)
    - integer string (e.g. "2", "10") => enabled with that limit (<=0 means unlimited)
    - strings containing "limit=<int>" (case-insensitive) => enabled with that limit
    """
    raw = os.environ.get("DIFFVG_SPLAT_TRACE", "").strip()
    if not raw:
        return False, None
    lowered = raw.lower()
    if lowered in {"0", "false", "no", "off"}:
        return False, None
    if "limit=" in lowered:
        try:
            limit = int(lowered.split("limit=", 1)[1].strip())
        except ValueError:
            limit = 5
        return True, limit
    try:
        limit = int(raw)
        return True, limit
    except ValueError:
        pass
    if lowered in {"1", "true", "yes", "on"} or lowered:
        return True, 5
    return False, None


def _debug_enabled() -> bool:
    enabled, _ = _trace_settings()
    return enabled


def _trace(message: str) -> None:
    if not _debug_enabled():
        return
    sys.stdout.write(f"\n[splat-trace] {message}\n")
    sys.stdout.flush()


_TRACE_FORWARD = 0
_TRACE_BACKWARD = 0
_TRACE_FALLBACK = 0
_TRACE_LIMIT: int | None = None


def _trace_limit() -> int:
    global _TRACE_LIMIT
    if _TRACE_LIMIT is None:
        enabled, limit = _trace_settings()
        if not enabled:
            _TRACE_LIMIT = 0
        else:
            _TRACE_LIMIT = 5 if limit is None else int(limit)
    return _TRACE_LIMIT


def _should_print(count: int) -> bool:
    limit = _trace_limit()
    return limit <= 0 or count <= limit


def _trace_summary() -> None:
    if not _debug_enabled():
        return
    print(
        f"[splat-trace] summary: forward={_TRACE_FORWARD} backward={_TRACE_BACKWARD} fallback={_TRACE_FALLBACK}",
        file=sys.stderr,
        flush=True,
    )


atexit.register(_trace_summary)


def _align_grad_devices(
    inputs: Iterable[object], grads: Iterable[Optional[torch.Tensor]]
) -> Tuple[Optional[torch.Tensor], ...]:
    aligned: List[Optional[torch.Tensor]] = []
    for inp, grad in zip(inputs, grads):
        if grad is None or not isinstance(grad, torch.Tensor):
            aligned.append(grad)
            continue
        if isinstance(inp, torch.Tensor):
            target_device = inp.device
            target_dtype = inp.dtype
            if grad.device != target_device or grad.dtype != target_dtype:
                grad = grad.to(device=target_device, dtype=target_dtype)
        aligned.append(grad)
    return tuple(aligned)


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


def _sample_path_geometry(
    segments: List[SegmentData],
    config: SplatConfig,
    device: torch.device,
    dtype: torch.dtype,
    generator: Optional[torch.Generator],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not segments:
        return (
            torch.empty(0, 2, device=device, dtype=dtype),
            torch.empty(0, 2, device=device, dtype=dtype),
            torch.empty(0, device=device, dtype=dtype),
        )

    mid_count = max(int(config.K) - 1, 0)
    mu_parts: List[torch.Tensor] = []
    tan_parts: List[torch.Tensor] = []

    start_t = torch.zeros(1, device=device, dtype=dtype)
    pos, tan = _evaluate_segment(segments[0], start_t)
    mu_parts.append(pos)
    tan_parts.append(tan)

    for idx, segment in enumerate(segments):
        if idx > 0:
            corner_t = torch.zeros(1, device=device, dtype=dtype)
            pos_corner, tan_corner = _evaluate_segment(segment, corner_t)
            mu_parts.append(pos_corner)
            tan_parts.append(tan_corner)
        if mid_count > 0:
            t_mid = _segment_samples(mid_count, device, dtype, generator)
            if t_mid.numel() > 0:
                pos_mid, tan_mid = _evaluate_segment(segment, t_mid)
                mu_parts.append(pos_mid)
                tan_parts.append(tan_mid)

    end_t = torch.ones(1, device=device, dtype=dtype)
    pos_end, tan_end = _evaluate_segment(segments[-1], end_t)
    mu_parts.append(pos_end)
    tan_parts.append(tan_end)

    mu = torch.cat(mu_parts, dim=0)
    tangents = torch.cat(tan_parts, dim=0)
    num_samples = mu.shape[0]

    if num_samples == 0:
        return (
            torch.empty(0, 2, device=device, dtype=dtype),
            torch.empty(0, 2, device=device, dtype=dtype),
            torch.empty(0, device=device, dtype=dtype),
        )

    if num_samples == 1:
        base_distance = torch.linalg.norm(segments[0].end - segments[0].start)
        sigma_x = torch.ones(1, device=device, dtype=dtype) * (base_distance / 2.0)
    else:
        diffs = mu[1:] - mu[:-1]
        dist = torch.linalg.norm(diffs, dim=1)
        dist_next = torch.zeros(num_samples, device=device, dtype=dtype)
        dist_prev = torch.zeros(num_samples, device=device, dtype=dtype)
        dist_next[:-1] = dist
        dist_prev[1:] = dist
        if num_samples > 2:
            dist_next[-1] = dist[-1]
            dist_prev[0] = dist[0]
        else:
            dist_next[-1] = dist[0]
            dist_prev[0] = dist[0]
        sigma_x = (dist_next + dist_prev) * 0.5

    return mu, tangents, torch.clamp(sigma_x, min=1e-6)


def _path_to_gaussians(
    spec: PathSpec,
    config: SplatConfig,
    device: torch.device,
    dtype: torch.dtype,
    generator: Optional[torch.Generator],
) -> GaussianBatch:
    mu, tangents, sigma_x = _sample_path_geometry(
        spec.segments, config, device, dtype, generator
    )
    num_samples = mu.shape[0]
    if num_samples == 0:
        raise _SplatUnsupported("path without samples cannot be rasterized")

    rho = max(float(config.rho), 1e-6)
    sigma_x = torch.clamp(sigma_x / rho, min=1e-3)

    width = torch.clamp(spec.stroke_width.reshape(-1)[0], min=1e-3)
    sigma_y = torch.ones(num_samples, device=device, dtype=dtype) * (width / (2.0 * rho))

    tangent_norm = torch.linalg.norm(tangents, dim=1, keepdim=True).clamp_min(1e-6)
    normalized_tan = tangents / tangent_norm
    theta = torch.atan2(normalized_tan[:, 1], normalized_tan[:, 0])

    color_rgb = spec.color_rgb.to(device=device, dtype=dtype).unsqueeze(0).expand(num_samples, -1)
    opacity = spec.opacity.to(device=device, dtype=dtype).clamp(0.0, 1.0).expand(num_samples)

    return GaussianBatch(
        mu=mu,
        theta=theta,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        color_rgb=color_rgb,
        opacity=opacity,
    )


def _fill_to_gaussians(
    spec: FillSpec,
    config: SplatConfig,
    device: torch.device,
    dtype: torch.dtype,
    generator: Optional[torch.Generator],
) -> GaussianBatch:
    mu_boundary, tangents, sigma_x = _sample_path_geometry(
        spec.segments, config, device, dtype, generator
    )
    num_samples = mu_boundary.shape[0]
    if num_samples == 0:
        raise _SplatUnsupported("fill path yielded no samples")

    centroid = mu_boundary.mean(dim=0, keepdim=True)
    levels = max(int(config.R), 0) + 1
    if levels <= 0:
        levels = 1
    level_factors = torch.linspace(
        1.0 / (levels + 1), 1.0, steps=levels, device=device, dtype=dtype
    )
    radial = mu_boundary - centroid
    mu_levels = centroid + radial.unsqueeze(0) * level_factors.view(-1, 1, 1)
    mu = mu_levels.reshape(-1, 2)

    tangents = tangents.repeat(levels, 1)
    sigma_x = sigma_x.repeat(levels)

    radial_step = torch.linalg.norm(radial, dim=1).clamp_min(1e-3) / (levels + 1)
    sigma_y = (
        radial_step.repeat(levels) / max(float(config.rho), 1e-6)
    ).clamp(min=1e-3)

    tangent_norm = torch.linalg.norm(tangents, dim=1, keepdim=True).clamp_min(1e-6)
    normalized_tan = tangents / tangent_norm
    theta = torch.atan2(normalized_tan[:, 1], normalized_tan[:, 0])

    color_rgb = spec.color_rgb.to(device=device, dtype=dtype).unsqueeze(0).expand(mu.size(0), -1)
    opacity = spec.opacity.to(device=device, dtype=dtype).clamp(0.0, 1.0).expand(mu.size(0))

    return GaussianBatch(
        mu=mu,
        theta=theta,
        sigma_x=torch.clamp(sigma_x, min=1e-3),
        sigma_y=torch.clamp(sigma_y, min=1e-3),
        color_rgb=color_rgb,
        opacity=opacity,
    )


def _composite_gaussians_full(
    mu: torch.Tensor,
    theta: torch.Tensor,
    sigma_x: torch.Tensor,
    sigma_y: torch.Tensor,
    color_rgb: torch.Tensor,
    opacity: torch.Tensor,
    width: int,
    height: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    yy = torch.arange(height, device=device, dtype=dtype) + 0.5
    xx = torch.arange(width, device=device, dtype=dtype) + 0.5
    try:
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
    except TypeError:
        grid_y, grid_x = torch.meshgrid(yy, xx)

    image_rgb = torch.zeros(height, width, 3, device=device, dtype=dtype)
    image_alpha = torch.zeros(height, width, device=device, dtype=dtype)

    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    inv_sigma_x = 1.0 / sigma_x
    inv_sigma_y = 1.0 / sigma_y

    for idx in range(mu.shape[0]):
        dx = grid_x - mu[idx, 0]
        dy = grid_y - mu[idx, 1]
        local_x = cos_theta[idx] * dx + sin_theta[idx] * dy
        local_y = -sin_theta[idx] * dx + cos_theta[idx] * dy
        exponent = -0.5 * ((local_x * inv_sigma_x[idx]) ** 2 + (local_y * inv_sigma_y[idx]) ** 2)
        gaussian = torch.exp(exponent)
        alpha_i = torch.clamp(opacity[idx], 0.0, 1.0) * gaussian
        contribution = (1.0 - image_alpha) * alpha_i
        image_rgb = image_rgb + contribution.unsqueeze(-1) * color_rgb[idx]
        image_alpha = image_alpha + contribution

    image_alpha = torch.clamp(image_alpha, 0.0, 1.0)
    return torch.cat([image_rgb, image_alpha.unsqueeze(-1)], dim=-1)


def _composite_gaussians_tiled(
    mu: torch.Tensor,
    theta: torch.Tensor,
    sigma_x: torch.Tensor,
    sigma_y: torch.Tensor,
    color_rgb: torch.Tensor,
    opacity: torch.Tensor,
    width: int,
    height: int,
    device: torch.device,
    dtype: torch.dtype,
    tile_size: int,
) -> torch.Tensor:
    tile_size = int(tile_size)
    if tile_size <= 0:
        return _composite_gaussians_full(
            mu, theta, sigma_x, sigma_y, color_rgb, opacity, width, height, device, dtype
        )

    tiles_x = (width + tile_size - 1) // tile_size
    tiles_y = (height + tile_size - 1) // tile_size
    if tiles_x == 0 or tiles_y == 0:
        return torch.zeros(height, width, 4, device=device, dtype=dtype)

    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    inv_sigma_x = 1.0 / sigma_x
    inv_sigma_y = 1.0 / sigma_y
    opacity_clamped = torch.clamp(opacity, 0.0, 1.0)

    extent_factor = 3.0
    extent_x = extent_factor * (torch.abs(cos_theta) * sigma_x + torch.abs(sin_theta) * sigma_y)
    extent_y = extent_factor * (torch.abs(sin_theta) * sigma_x + torch.abs(cos_theta) * sigma_y)

    min_tile_x = torch.floor((mu[:, 0] - extent_x) / tile_size).to(torch.int64)
    max_tile_x = torch.floor((mu[:, 0] + extent_x) / tile_size).to(torch.int64)
    min_tile_y = torch.floor((mu[:, 1] - extent_y) / tile_size).to(torch.int64)
    max_tile_y = torch.floor((mu[:, 1] + extent_y) / tile_size).to(torch.int64)

    min_tile_x = torch.clamp(min_tile_x, 0, tiles_x - 1)
    max_tile_x = torch.clamp(max_tile_x, 0, tiles_x - 1)
    min_tile_y = torch.clamp(min_tile_y, 0, tiles_y - 1)
    max_tile_y = torch.clamp(max_tile_y, 0, tiles_y - 1)

    tile_bins: List[List[int]] = [[] for _ in range(tiles_x * tiles_y)]
    num_gaussians = mu.shape[0]
    for idx in range(num_gaussians):
        x0 = int(min_tile_x[idx].item())
        x1 = int(max_tile_x[idx].item())
        y0 = int(min_tile_y[idx].item())
        y1 = int(max_tile_y[idx].item())
        if x0 > x1 or y0 > y1:
            continue
        for ty in range(y0, y1 + 1):
            for tx in range(x0, x1 + 1):
                tile_bins[ty * tiles_x + tx].append(idx)

    image_rgb = torch.zeros(height, width, 3, device=device, dtype=dtype)
    image_alpha = torch.zeros(height, width, device=device, dtype=dtype)

    for tile_id, idx_list in enumerate(tile_bins):
        if not idx_list:
            continue
        tile_x = tile_id % tiles_x
        tile_y = tile_id // tiles_x
        x0 = tile_x * tile_size
        y0 = tile_y * tile_size
        x1 = min(width, x0 + tile_size)
        y1 = min(height, y0 + tile_size)
        if x0 >= x1 or y0 >= y1:
            continue

        tile_indices = torch.tensor(idx_list, device=device, dtype=torch.long)

        yy = torch.arange(y0, y1, device=device, dtype=dtype) + 0.5
        xx = torch.arange(x0, x1, device=device, dtype=dtype) + 0.5
        try:
            grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
        except TypeError:
            grid_y, grid_x = torch.meshgrid(yy, xx)

        tile_rgb = image_rgb[y0:y1, x0:x1]
        tile_alpha = image_alpha[y0:y1, x0:x1]
        tile_rgb_work = tile_rgb.clone()
        tile_alpha_work = tile_alpha.clone()

        for local_idx in tile_indices:
            idx = int(local_idx.item())
            dx = grid_x - mu[idx, 0]
            dy = grid_y - mu[idx, 1]
            local_x = cos_theta[idx] * dx + sin_theta[idx] * dy
            local_y = -sin_theta[idx] * dx + cos_theta[idx] * dy
            exponent = -0.5 * (
                (local_x * inv_sigma_x[idx]) ** 2 + (local_y * inv_sigma_y[idx]) ** 2
            )
            gaussian = torch.exp(exponent)
            alpha_i = opacity_clamped[idx] * gaussian
            contribution = (1.0 - tile_alpha_work) * alpha_i
            tile_rgb_work = tile_rgb_work + contribution.unsqueeze(-1) * color_rgb[idx]
            tile_alpha_work = tile_alpha_work + contribution

        image_rgb[y0:y1, x0:x1] = tile_rgb_work
        image_alpha[y0:y1, x0:x1] = torch.clamp(tile_alpha_work, 0.0, 1.0)

    return torch.cat([image_rgb, image_alpha.unsqueeze(-1)], dim=-1)


def _render_forward(request: RenderRequest) -> torch.Tensor:
    scene = request.scene
    device = get_device()
    dtype = torch.float32
    if scene.paths:
        sample_dtype = scene.paths[0].points.dtype
        if sample_dtype in (torch.float32, torch.float64):
            dtype = sample_dtype

    if scene.output_type is not None and scene.output_type != OutputType.color:
        raise _SplatUnsupported("only OutputType.color is supported")
    if scene.use_prefiltering:
        raise _SplatUnsupported("prefiltering is not supported in splat backend yet")
    if scene.eval_positions.numel() != 0:
        raise _SplatUnsupported("SDF queries are not supported in splat backend yet")
    if request.background_image is not None:
        raise _SplatUnsupported("background compositing is not supported yet")

    stroke_specs, fill_specs = _gather_specs(scene, device, dtype)
    batches: List[GaussianBatch] = []
    if stroke_specs:
        batches.extend(
            _path_to_gaussians(spec, request.config, device, dtype, request.generator)
            for spec in stroke_specs
        )
    if fill_specs:
        batches.extend(
            _fill_to_gaussians(spec, request.config, device, dtype, request.generator)
            for spec in fill_specs
        )
    if not batches:
        raise _SplatUnsupported("scene does not contain supported path primitives")
    mu = torch.cat([batch.mu for batch in batches], dim=0)
    theta = torch.cat([batch.theta for batch in batches], dim=0)
    sigma_x = torch.cat([batch.sigma_x for batch in batches], dim=0)
    sigma_y = torch.cat([batch.sigma_y for batch in batches], dim=0)
    color_rgb = torch.cat([batch.color_rgb for batch in batches], dim=0)
    opacity = torch.cat([batch.opacity for batch in batches], dim=0)

    if mu.shape[0] == 0:
        raise _SplatUnsupported("no gaussian samples were generated")

    global _TRACE_FORWARD
    _TRACE_FORWARD += 1
    if _TRACE_FORWARD == 1:
        _trace("splat backend active (first render)")
    if _should_print(_TRACE_FORWARD):
        _trace(
            f"render_forward[{_TRACE_FORWARD}] strokes={len(stroke_specs)} fills={len(fill_specs)} "
            f"gaussians={mu.shape[0]} device={device.type} grad={_scene_requires_grad(scene)}"
        )

    if request.config.depth_policy == DepthPolicy.small_first:
        sort_key = sigma_y
        order = torch.argsort(sort_key)
        mu = mu[order]
        theta = theta[order]
        sigma_x = sigma_x[order]
        sigma_y = sigma_y[order]
        color_rgb = color_rgb[order]
        opacity = opacity[order]
    tile_size = int(request.config.tile)
    # Disable tiling during autograd to keep graph intact.
    if tile_size > 0 and not _scene_requires_grad(scene):
        return _composite_gaussians_tiled(
            mu,
            theta,
            sigma_x,
            sigma_y,
            color_rgb,
            opacity,
            request.width,
            request.height,
            device,
            dtype,
            tile_size,
        )
    return _composite_gaussians_full(
        mu,
        theta,
        sigma_x,
        sigma_y,
        color_rgb,
        opacity,
        request.width,
        request.height,
        device,
        dtype,
    )


def _enable_gradient_args(args: Iterable[object]) -> Tuple[Tuple[object, ...], List[GradSlot]]:
    args_list = list(args)
    grad_slots: List[GradSlot] = []
    idx = 0

    if len(args_list) < 7:
        raise _SplatUnsupported("serialized scene args truncated")
    idx += 1  # canvas_width
    idx += 1  # canvas_height
    num_shapes = int(args_list[idx])
    idx += 1
    num_shape_groups = int(args_list[idx])
    idx += 1
    idx += 1  # output_type
    idx += 1  # use_prefiltering
    idx += 1  # eval_positions

    for _shape_id in range(num_shapes):
        shape_type = args_list[idx]
        idx += 1
        if shape_type != diffvg.ShapeType.path:
            raise _SplatUnsupported("non-path shapes are not handled yet")
        idx += 1  # num_control_points
        points_idx = idx
        points = args_list[idx]
        if not isinstance(points, torch.Tensor):
            raise _SplatUnsupported("expected tensor points for path")
        if not points.requires_grad:
            points.requires_grad_(True)
        args_list[points_idx] = points
        grad_slots.append(GradSlot(arg_index=points_idx, tensor=points))
        idx += 1
        thickness = args_list[idx]
        idx += 1
        if thickness is not None:
            raise _SplatUnsupported("per-point thickness is not supported yet")
        idx += 1  # is_closed
        idx += 1  # use_distance_approx
        stroke_idx = idx
        stroke_width = args_list[idx]
        if isinstance(stroke_width, torch.Tensor):
            if not stroke_width.requires_grad:
                stroke_width.requires_grad_(True)
            args_list[stroke_idx] = stroke_width
            grad_slots.append(GradSlot(arg_index=stroke_idx, tensor=stroke_width))
        idx += 1

    for _group_id in range(num_shape_groups):
        idx += 1  # shape_ids
        fill_color_type = args_list[idx]
        idx += 1
        if fill_color_type is not None:
            if fill_color_type != diffvg.ColorType.constant:
                raise _SplatUnsupported("only constant fill colors are supported")
            color_idx = idx
            fill_color = args_list[idx]
            if not isinstance(fill_color, torch.Tensor):
                raise _SplatUnsupported("fill color tensor expected")
            if not fill_color.requires_grad:
                fill_color.requires_grad_(True)
            args_list[color_idx] = fill_color
            grad_slots.append(GradSlot(arg_index=color_idx, tensor=fill_color))
            idx += 1
        stroke_color_type = args_list[idx]
        idx += 1
        if stroke_color_type != diffvg.ColorType.constant:
            if stroke_color_type is not None:
                raise _SplatUnsupported("only constant stroke colors supported")
            continue
        color_idx = idx
        stroke_color = args_list[idx]
        if not isinstance(stroke_color, torch.Tensor):
            raise _SplatUnsupported("stroke color tensor expected")
        if not stroke_color.requires_grad:
            stroke_color.requires_grad_(True)
        args_list[color_idx] = stroke_color
        grad_slots.append(GradSlot(arg_index=color_idx, tensor=stroke_color))
        idx += 1
        idx += 1  # use_even_odd_rule
        idx += 1  # shape_to_canvas

    idx += 1  # filter_type
    idx += 1  # filter_radius

    if idx != len(args_list):
        raise _SplatUnsupported("unexpected trailing data in serialized scene")

    return tuple(args_list), grad_slots


def _cpu_args(args: Iterable[object]) -> Tuple[object, ...]:
    cpu_args: List[object] = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            cpu_args.append(arg.to(device="cpu").contiguous())
        else:
            cpu_args.append(arg)
    return tuple(cpu_args)


def _render_backward(
    grad_img: torch.Tensor,
    width: int,
    height: int,
    num_samples_x: int,
    num_samples_y: int,
    seed: int,
    background_image: Optional[torch.Tensor],
    args: Iterable[object],
) -> Tuple[Optional[torch.Tensor], ...]:
    if background_image is not None:
        raise _SplatUnsupported("background compositing is not supported yet")
    args_with_grad, grad_slots = _enable_gradient_args(args)
    request = _prepare_render_request(
        width,
        height,
        num_samples_x,
        num_samples_y,
        seed,
        background_image,
        args_with_grad,
    )
    with torch.enable_grad():
        image = _render_forward(request)
        grad_img_cast = grad_img.to(device=image.device, dtype=image.dtype).contiguous()
        loss = torch.sum(image * grad_img_cast)
    if _debug_enabled():
        _trace(
            f"backward-check image.requires_grad={bool(image.requires_grad)} loss.requires_grad={bool(loss.requires_grad)} device={image.device}"
        )
    targets = [slot.tensor for slot in grad_slots]
    active: List[Tuple[GradSlot, torch.Tensor]] = []
    for slot, tensor in zip(grad_slots, targets):
        if isinstance(tensor, torch.Tensor) and tensor.requires_grad:
            active.append((slot, tensor))
    if _debug_enabled():
        stats = ", ".join(
            f"{idx}:{tensor.requires_grad}:{tensor.grad_fn is not None}:{tensor.device}"
            for idx, tensor in enumerate(targets)
        )
        _trace(f"render_backward targets={stats}")
    if active:
        active_slots, active_tensors = zip(*active)
        try:
            grads_active = torch.autograd.grad(loss, active_tensors, retain_graph=True, allow_unused=True)
        except RuntimeError as exc:  # pragma: no cover - defensive path, triggers fallback
            raise _SplatUnsupported(f"autograd failure: {exc}") from exc
    else:
        active_slots = ()
        grads_active = ()
    global _TRACE_BACKWARD
    _TRACE_BACKWARD += 1
    if _should_print(_TRACE_BACKWARD):
        _trace(f"render_backward[{_TRACE_BACKWARD}] autograd_targets={len(targets)}")

    total_args = len(args)
    grad_list: List[Optional[torch.Tensor]] = [None] * (6 + total_args)
    active_lookup = {id(slot.tensor): grad for slot, grad in zip(active_slots, grads_active)}
    for slot, tensor in zip(grad_slots, targets):
        grad_value = active_lookup.get(id(tensor), None)
        if grad_value is None:
            grad_tensor = torch.zeros_like(slot.tensor)
        else:
            grad_tensor = grad_value
        grad_list[6 + slot.arg_index] = grad_tensor.detach()
    return tuple(grad_list)


def apply(
    width: int,
    height: int,
    num_samples_x: int,
    num_samples_y: int,
    seed: int,
    background_image: Optional[torch.Tensor],
    *args: object,
    ) -> torch.Tensor:
    try:
        return SplatRenderFunction.apply(
            width,
            height,
            num_samples_x,
            num_samples_y,
            seed,
            background_image,
            *args,
        )
    except _SplatUnsupported as exc:
        _warn_fallback(exc.reason)
        cpu_args = _cpu_args(args)
        _trace("apply delegating to baseline RenderFunction")
        return _BaselineRF.apply(
            width,
            height,
            num_samples_x,
            num_samples_y,
            seed,
            background_image,
            *cpu_args,
        )


def render_grad(
    grad_img: torch.Tensor,
    width: int,
    height: int,
    num_samples_x: int,
    num_samples_y: int,
    seed: int,
    background_image: Optional[torch.Tensor],
    *args: object,
    ) -> Tuple[Optional[torch.Tensor], ...]:
    try:
        return _render_backward(
            grad_img,
            width,
            height,
            num_samples_x,
            num_samples_y,
            seed,
            background_image,
            args,
        )
    except _SplatUnsupported as exc:
        _warn_fallback(exc.reason)
        _trace("render_grad delegating to baseline Backward")
        # Emulate BaselineRenderFunction forward/backward to obtain gradient tuple
        cpu_args = _cpu_args(args)
        ctx = SimpleNamespace()
        # Forward to build ctx.scene etc.
        _ = _BaselineRF.forward(
            ctx,
            width,
            height,
            num_samples_x,
            num_samples_y,
            seed,
            background_image,
            *cpu_args,
        )
        # Backward to assemble gradient tuple for all inputs
        grad_img_cast = grad_img.to(device=get_device()).contiguous()
        grads = _BaselineRF.backward(ctx, grad_img_cast)
        # Align gradient tensors to match the original forward inputs of splat
        forward_inputs = (
            width,
            height,
            num_samples_x,
            num_samples_y,
            seed,
            background_image,
            *args,
        )
        return _align_grad_devices(forward_inputs, grads)


class SplatRenderFunction(torch.autograd.Function):
    @staticmethod
    def serialize_scene(
        canvas_width,
        canvas_height,
        shapes,
        shape_groups,
        filter=None,
        output_type=OutputType.color,
        use_prefiltering: bool = False,
        eval_positions: torch.Tensor = torch.tensor([]),
        *,
        keep_on_device: bool = False,
        device: Optional[torch.device | str] = None,
    ):
        return _serialize_scene(
            canvas_width,
            canvas_height,
            shapes,
            shape_groups,
            filter,
            output_type,
            use_prefiltering,
            eval_positions,
            keep_on_device=keep_on_device,
            device=device,
        )

    @staticmethod
    def forward(
        ctx,
        width: int,
        height: int,
        num_samples_x: int,
        num_samples_y: int,
        seed: int,
        background_image: Optional[torch.Tensor],
        *args: object,
    ) -> torch.Tensor:
        request = _prepare_render_request(
            width,
            height,
            num_samples_x,
            num_samples_y,
            seed,
            background_image,
            args,
        )
        ctx.request = request  # type: ignore[attr-defined]
        ctx.extra_args = args  # type: ignore[attr-defined]
        return _render_forward(request)

    @staticmethod
    def backward(ctx, *grad_outputs: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        (grad_img,) = grad_outputs
        request: RenderRequest = ctx.request  # type: ignore[attr-defined]
        args = ctx.extra_args  # type: ignore[attr-defined]
        try:
            grads = render_grad(
                grad_img,
                request.width,
                request.height,
                request.num_samples_x,
                request.num_samples_y,
                request.seed,
                request.background_image,
                *args,
            )
        except _SplatUnsupported as exc:
            _warn_fallback(exc.reason)
            if _debug_enabled():
                _trace(f"fallback baseline backward (args={len(args)})")
            cpu_args = _cpu_args(args)
            bctx = SimpleNamespace()
            _ = _BaselineRF.forward(
                bctx,
                request.width,
                request.height,
                request.num_samples_x,
                request.num_samples_y,
                request.seed,
                request.background_image,
                *cpu_args,
            )
            grad_img_cast = grad_img.to(device=get_device()).contiguous()
            grads = _BaselineRF.backward(bctx, grad_img_cast)
            forward_inputs = (
                request.width,
                request.height,
                request.num_samples_x,
                request.num_samples_y,
                request.seed,
                request.background_image,
                *args,
            )
            grads = _align_grad_devices(forward_inputs, grads)
        return grads


serialize_scene = SplatRenderFunction.serialize_scene

__all__ = [
    "apply",
    "render_grad",
    "serialize_scene",
    "SplatRenderFunction",
    "RenderRequest",
    "ScenePayload",
    "PathPayload",
    "NonPathShapePayload",
    "ShapeGroupPayload",
    "PaintPayload",
]
def _scene_requires_grad(scene: ScenePayload) -> bool:
    # Check any path geometry or styling tensors for requires_grad
    for p in scene.paths:
        if isinstance(p.points, torch.Tensor) and p.points.requires_grad:
            return True
        if isinstance(p.stroke_width, torch.Tensor) and p.stroke_width.requires_grad:
            return True
    for g in scene.shape_groups:
        for paint in (g.fill, g.stroke):
            if paint.color_type is None:
                continue
            for t in paint.params:
                if isinstance(t, torch.Tensor) and t.requires_grad:
                    return True
    return False
