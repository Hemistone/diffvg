from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Iterable, List, Optional, Tuple

import torch

from .backend import BezierGsplatConfig, get_backend_config
from .bezier_gsplat.runtime import load_gsplat_ops
from .device import get_device
from .render_pytorch import BaselineRenderFunction as _BaselineRF
from .render_pytorch import OutputType
from .serialization import serialize_scene as _serialize_scene
from .splat.geometry import (
    _evaluate_segment,
    _gather_specs,
    _segment_arclength,
    _segment_samples,
)
from .splat.scene import _deserialize_scene
from .splat.trace import (
    increment_backward as _increment_backward,
    increment_forward as _increment_forward,
    should_print as _should_print,
    trace as _trace,
    warn_fallback as _warn_fallback,
)
from .splat.types import GaussianBatch, GradSlot, PathSpec, ScenePayload, _SplatUnsupported
from .splat.vjp import _align_grad_devices, _cpu_args, _enable_gradient_args


@dataclass(frozen=True)
class BezierGsplatRequest:
    width: int
    height: int
    num_samples_x: int
    num_samples_y: int
    seed: int
    background_image: Optional[torch.Tensor]
    config: BezierGsplatConfig
    scene: ScenePayload
    generator: Optional[torch.Generator]


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
    keep_on_device: bool = True,
    device: Optional[torch.device | str] = None,
):
    dev = torch.device(device) if device is not None else get_device()
    return _serialize_scene(
        canvas_width,
        canvas_height,
        shapes,
        shape_groups,
        filter,
        output_type,
        use_prefiltering,
        eval_positions,
        keep_on_device=True,
        device=dev,
    )


def _prepare_render_request(
    width: int,
    height: int,
    num_samples_x: int,
    num_samples_y: int,
    seed: int,
    background_image: Optional[torch.Tensor],
    args: Iterable[object],
) -> BezierGsplatRequest:
    config = get_backend_config("bezier_gsplat")
    if not isinstance(config, BezierGsplatConfig):
        config = BezierGsplatConfig()
    scene = _deserialize_scene(args)
    device = get_device()
    generator: Optional[torch.Generator]
    if seed is None:
        generator = None
    else:
        generator = torch.Generator(device=device) if device.type == "cuda" else torch.Generator()
        generator.manual_seed(int(seed) & 0xFFFFFFFFFFFFFFFF)
    return BezierGsplatRequest(
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


def _sample_open_stroke(
    spec: PathSpec,
    config: BezierGsplatConfig,
    device: torch.device,
    dtype: torch.dtype,
    generator: Optional[torch.Generator],
) -> GaussianBatch:
    segments = spec.segments
    if not segments:
        raise _SplatUnsupported("path without segments cannot be rasterized")

    spacing = max(float(config.sample_spacing_px), 1e-6)
    max_mid = max(int(config.max_samples_per_segment), 0)
    lengths = torch.stack([
        _segment_arclength(seg, device=device, dtype=dtype, samples=16) for seg in segments
    ])
    mid_counts: List[int] = []
    for length in lengths:
        total = max(1, int(math.ceil(float(length.item()) / spacing)))
        mid_counts.append(min(max(total - 1, 0), max_mid))

    mu_parts: List[torch.Tensor] = []
    tan_parts: List[torch.Tensor] = []

    start_t = torch.zeros(1, device=device, dtype=dtype)
    pos_start, tan_start = _evaluate_segment(segments[0], start_t)
    mu_parts.append(pos_start)
    tan_parts.append(tan_start)

    for seg_idx, segment in enumerate(segments):
        if seg_idx > 0:
            corner_t = torch.zeros(1, device=device, dtype=dtype)
            pos_corner, tan_corner = _evaluate_segment(segment, corner_t)
            mu_parts.append(pos_corner)
            tan_parts.append(tan_corner)
        mid_count = mid_counts[seg_idx]
        if mid_count <= 0:
            continue
        t_mid = _segment_samples(mid_count, device, dtype, generator)
        if t_mid.numel() == 0:
            continue
        pos_mid, tan_mid = _evaluate_segment(segment, t_mid)
        mu_parts.append(pos_mid)
        tan_parts.append(tan_mid)

    end_t = torch.ones(1, device=device, dtype=dtype)
    pos_end, tan_end = _evaluate_segment(segments[-1], end_t)
    mu_parts.append(pos_end)
    tan_parts.append(tan_end)

    mu = torch.cat(mu_parts, dim=0)
    tangents = torch.cat(tan_parts, dim=0)
    if mu.numel() == 0:
        raise _SplatUnsupported("path without samples cannot be rasterized")

    if mu.shape[0] == 1:
        base_distance = torch.linalg.norm(segments[0].end - segments[0].start)
        sigma_x = base_distance.reshape(1) * 0.5
    else:
        diffs = mu[1:] - mu[:-1]
        dist = torch.linalg.norm(diffs, dim=1)
        dist_next = torch.zeros(mu.shape[0], device=device, dtype=dtype)
        dist_prev = torch.zeros(mu.shape[0], device=device, dtype=dtype)
        dist_next[:-1] = dist
        dist_prev[1:] = dist
        edge = dist[-1] if dist.numel() > 0 else torch.tensor(1.0, device=device, dtype=dtype)
        dist_next[-1] = edge
        dist_prev[0] = edge
        sigma_x = 0.5 * (dist_next + dist_prev)

    tangent_norm = torch.linalg.norm(tangents, dim=1, keepdim=True).clamp_min(1e-6)
    normalized_tangent = tangents / tangent_norm
    theta = torch.atan2(normalized_tangent[:, 1], normalized_tangent[:, 0]).unsqueeze(-1)

    width = torch.clamp(spec.stroke_width.reshape(-1)[0].to(device=device, dtype=dtype), min=config.min_scale)
    fwhm_coeff = 2.0 * math.sqrt(2.0 * math.log(2.0))
    sigma_y = torch.ones(mu.shape[0], device=device, dtype=dtype) * (width / fwhm_coeff)

    color_rgb = spec.color_rgb.to(device=device, dtype=dtype).unsqueeze(0).expand(mu.shape[0], -1)
    opacity = spec.opacity.to(device=device, dtype=dtype).clamp(0.0, 1.0).expand(mu.shape[0])

    return GaussianBatch(
        mu=mu,
        theta=theta,
        sigma_x=torch.clamp(sigma_x, min=config.min_scale),
        sigma_y=torch.clamp(sigma_y, min=config.min_scale),
        color_rgb=color_rgb,
        opacity=opacity,
    )


def _scene_order_depths(counts: List[int], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if not counts:
        return torch.empty(0, 1, device=device, dtype=dtype)
    num_specs = max(len(counts), 1)
    depth_parts: List[torch.Tensor] = []
    for spec_idx, count in enumerate(counts):
        if count <= 0:
            continue
        base = 1.0 - (float(spec_idx) / float(num_specs))
        if count == 1:
            depth_parts.append(torch.full((1, 1), base, device=device, dtype=dtype))
            continue
        intra = torch.linspace(0.0, 1e-4, steps=count, device=device, dtype=dtype).unsqueeze(-1)
        depth_parts.append(torch.full((count, 1), base, device=device, dtype=dtype) - intra)
    return torch.cat(depth_parts, dim=0) if depth_parts else torch.empty(0, 1, device=device, dtype=dtype)


def _normalize_means_to_ndc(means_px: torch.Tensor, width: int, height: int) -> torch.Tensor:
    width = max(int(width), 1)
    height = max(int(height), 1)
    scale = means_px.new_tensor([2.0 / float(width), 2.0 / float(height)])
    bias = means_px.new_tensor([-1.0, -1.0])
    return means_px * scale + bias


def _project_with_safe_tile_budget(
    request: BezierGsplatRequest,
    means_ndc: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    ops,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    block_h = max(int(request.config.block_h), 1)
    block_w = max(int(request.config.block_w), 1)
    width = max(int(request.width), 1)
    height = max(int(request.height), 1)

    while True:
        tile_bounds = (
            (width + block_w - 1) // block_w,
            (height + block_h - 1) // block_h,
            1,
        )
        xys, depths, radii, conics, num_tiles_hit = ops.project_gaussians_2d_scale_rot(
            means_ndc,
            scales,
            rotations,
            height,
            width,
            tile_bounds,
        )
        num_tiles = tile_bounds[0] * tile_bounds[1]
        num_intersects = int(num_tiles_hit.sum().item())
        # Work around an upstream gsplat tile-binning bug that assumes the tile-bin
        # array is at least as large as the number of tiles.
        if num_intersects >= num_tiles or (block_h >= height and block_w >= width):
            return xys, depths, radii, conics, num_tiles_hit, block_h, block_w
        block_h = min(height, block_h * 2)
        block_w = min(width, block_w * 2)


def _render_forward(request: BezierGsplatRequest) -> torch.Tensor:
    ops = load_gsplat_ops()
    device = get_device()
    if device.type != "cuda":
        raise _SplatUnsupported("bezier_gsplat requires a CUDA device")

    scene = request.scene
    dtype = torch.float32
    if scene.output_type is not None and scene.output_type != OutputType.color:
        raise _SplatUnsupported("only OutputType.color is supported")
    if scene.use_prefiltering:
        raise _SplatUnsupported("prefiltering is not supported in bezier_gsplat")
    if scene.eval_positions.numel() != 0:
        raise _SplatUnsupported("SDF queries are not supported in bezier_gsplat")
    if request.background_image is not None:
        raise _SplatUnsupported("background compositing is not supported in bezier_gsplat")

    stroke_specs, fill_specs = _gather_specs(scene, device, dtype)
    if fill_specs:
        raise _SplatUnsupported("filled paths are not supported in bezier_gsplat yet")
    if not stroke_specs:
        raise _SplatUnsupported("scene does not contain supported open-stroke paths")

    batches = [
        _sample_open_stroke(spec, request.config, device, dtype, request.generator)
        for spec in stroke_specs
    ]
    means = torch.cat([batch.mu for batch in batches], dim=0).contiguous()
    scales = torch.stack(
        [
            torch.cat([batch.sigma_x for batch in batches], dim=0),
            torch.cat([batch.sigma_y for batch in batches], dim=0),
        ],
        dim=-1,
    ).contiguous()
    rotations = torch.cat([batch.theta for batch in batches], dim=0).contiguous()
    colors = torch.cat([batch.color_rgb for batch in batches], dim=0).contiguous()
    opacity = torch.cat([batch.opacity for batch in batches], dim=0).unsqueeze(-1).contiguous()
    counts = [int(batch.mu.shape[0]) for batch in batches]
    if request.config.depth_mode != "scene_order":
        raise _SplatUnsupported(f"unsupported depth mode '{request.config.depth_mode}'")
    depths = _scene_order_depths(counts, device=device, dtype=dtype)
    if means.shape[0] == 0:
        raise _SplatUnsupported("no Gaussian samples were generated")

    xys, _depths_proj, radii, conics, num_tiles_hit, block_h, block_w = _project_with_safe_tile_budget(
        request,
        _normalize_means_to_ndc(means, request.width, request.height),
        scales,
        rotations,
        ops,
    )
    rgb, alpha = ops.rasterize_gaussians(
        xys,
        depths,
        radii,
        conics,
        num_tiles_hit,
        colors,
        opacity,
        request.height,
        request.width,
        block_h,
        block_w,
        background=torch.zeros(3, device=device, dtype=dtype),
        return_alpha=True,
    )
    rgb = rgb.reshape(-1, request.height, request.width, 3)[0]
    alpha = alpha.reshape(-1, request.height, request.width, 1)[0]
    image = torch.cat([rgb, alpha], dim=-1)

    trace_count = _increment_forward()
    if _should_print(trace_count):
        _trace(
            f"bezier_gsplat forward[{trace_count}] strokes={len(stroke_specs)} "
            f"gaussians={means.shape[0]} block={block_h}x{block_w}"
        )
    return image.clamp(0.0, 1.0)


def _autograd_backward(
    grad_img: torch.Tensor,
    width: int,
    height: int,
    num_samples_x: int,
    num_samples_y: int,
    seed: int,
    background_image: Optional[torch.Tensor],
    args: Iterable[object],
) -> Tuple[Optional[torch.Tensor], ...]:
    args_tuple = tuple(args)
    args_with_grad, grad_slots = _enable_gradient_args(args_tuple)
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

    active: List[Tuple[GradSlot, torch.Tensor]] = []
    for slot in grad_slots:
        if isinstance(slot.tensor, torch.Tensor) and slot.tensor.requires_grad:
            active.append((slot, slot.tensor))
    if active:
        active_slots, active_tensors = zip(*active)
        grads_active = torch.autograd.grad(loss, active_tensors, retain_graph=False, allow_unused=True)
        grads_active = tuple(
            torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0) if isinstance(grad, torch.Tensor) else grad
            for grad in grads_active
        )
    else:
        active_slots = ()
        grads_active = ()

    trace_count = _increment_backward()
    if _should_print(trace_count):
        _trace(f"bezier_gsplat backward[{trace_count}] autograd_targets={len(active)}")

    grad_list: List[Optional[torch.Tensor]] = [None] * (6 + len(args_tuple))
    active_lookup = {id(slot.tensor): grad for slot, grad in zip(active_slots, grads_active)}
    for slot in grad_slots:
        grad_value = active_lookup.get(id(slot.tensor), None)
        grad_list[6 + slot.arg_index] = (
            torch.zeros_like(slot.tensor) if grad_value is None else grad_value.detach()
        )
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
    request = _prepare_render_request(
        width,
        height,
        num_samples_x,
        num_samples_y,
        seed,
        background_image,
        args,
    )
    try:
        return _render_forward(request)
    except _SplatUnsupported as exc:
        _warn_fallback(exc.reason)
        cpu_args = _cpu_args(args)
        _trace("bezier_gsplat apply delegating to baseline RenderFunction")
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
        return _autograd_backward(
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
        _trace("bezier_gsplat render_grad delegating to baseline Backward")
        cpu_args = _cpu_args(args)
        ctx = SimpleNamespace()
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
        grad_img_cast = grad_img.to(device=get_device()).contiguous()
        grads = _BaselineRF.backward(ctx, grad_img_cast)
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


__all__ = ["serialize_scene", "apply", "render_grad", "BezierGsplatRequest"]
