from __future__ import annotations

import math
import os
from typing import Iterable, List, Optional, Tuple
from types import SimpleNamespace

import diffvg
import torch

from .backend import DepthPolicy, SplatConfig, get_backend_config
from .device import get_device
from .render_pytorch import OutputType, BaselineRenderFunction as _BaselineRF
from .serialization import serialize_scene as _serialize_scene
from . import triton_splat as _triton
from .splat.types import (
    GaussianBatch,
    GradSlot,
    NonPathShapePayload,
    PaintPayload,
    PathPayload,
    PathSpec,
    RenderRequest,
    ScenePayload,
    SegmentData,
    ShapeGroupPayload,
    FillSpec,
    _SplatUnsupported,
)
from .splat.trace import (
    debug_enabled as _debug_enabled,
    increment_backward as _increment_backward,
    increment_forward as _increment_forward,
    trace as _trace,
    should_print as _should_print,
    warn_fallback as _warn_fallback,
)
from .splat.env import _env_flag
from .splat.geometry import _gather_specs
from .splat.gauss import _fill_to_gaussians, _path_to_gaussians
from .splat.compositor import (
    _composite_gaussians_full,
    _composite_gaussians_full_ckpt,
    _composite_gaussians_tiled,
    _composite_gaussians_tiled_diff,
    _get_full_grid,
)
from .splat.vjp import (
    _align_grad_devices,
    _cpu_args,
    _enable_gradient_args,
    _scene_requires_grad,
)


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



def _render_forward(request: RenderRequest, ctx: Optional[object] = None) -> torch.Tensor:
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

    trace_count = _increment_forward()
    if trace_count == 1:
        _trace("splat backend active (first render)")
    if _should_print(trace_count):
        _trace(
            f"render_forward[{trace_count}] strokes={len(stroke_specs)} fills={len(fill_specs)} "
            f"gaussians={mu.shape[0]} device={device.type} grad={_scene_requires_grad(scene)}"
        )

    order: Optional[torch.Tensor] = None
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
    # Tiling policy:
    # - No-grad scenes: always use classic tiler when tile_size > 0
    # - Grad scenes: prefer tiled-diff for correctness parity with no-grad path
    if tile_size > 0:
        if not _scene_requires_grad(scene):
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
        # Default-on tiled diff to match tiler semantics under grad
        # Save CSR + SoA so Triton backward can run even if forward used Torch
        if ctx is not None and _triton.is_available() and device.type == "cuda":
            try:
                tile_ptr, tile_idx, tiles_x, tiles_y = _triton._build_tile_csr(
                    mu.detach(), theta.detach(), sigma_x.detach(), sigma_y.detach(),
                    request.width, request.height, tile_size,
                )
                payload = {
                    "mu": mu,
                    "theta": theta,
                    "sigma_x": sigma_x,
                    "sigma_y": sigma_y,
                    "color_rgb": color_rgb,
                    "opacity": opacity,
                    "tile_ptr": tile_ptr,
                    "tile_idx": tile_idx,
                    "tiles_x": tiles_x,
                    "tiles_y": tiles_y,
                    "tile_size": tile_size,
                    "width": request.width,
                    "height": request.height,
                }
                # Record sorting order if any to align VJP reconstruction
                payload["order"] = order
                setattr(ctx, "splat_saved", payload)
            except Exception as _exc:
                if _debug_enabled():
                    _trace(f"forward save skipped: {type(_exc).__name__}")
        return _composite_gaussians_tiled_diff(
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
    # Triton path (forward-only) when requested and available.
    if (
        _triton.env_wants_triton()
        and _triton.is_available()
        and device.type == "cuda"
    ):
        try:
            gchunk = int(os.environ.get("DIFFVG_SPLAT_GCHUNK", "256").strip() or "256")
        except Exception:
            gchunk = 256
        try:
            # Save SoA + CSR + per-spec offsets for Triton backward
            if ctx is not None:
                tile_size = int(request.config.tile)
                if tile_size > 0:
                    tile_ptr, tile_idx, tiles_x, tiles_y = _triton._build_tile_csr(
                        mu, theta, sigma_x, sigma_y, request.width, request.height, tile_size
                    )
                    # Per-spec sample counts and references in same order as 'batches'
                    spec_counts: List[int] = [int(b.mu.shape[0]) for b in batches]
                    # Collect references (not expanded) matching batches ordering
                    color_refs: List[torch.Tensor] = []
                    opacity_refs: List[torch.Tensor] = []
                    for spec in stroke_specs:
                        color_refs.append(spec.color_rgb)
                        opacity_refs.append(spec.opacity)
                    for spec in fill_specs:
                        color_refs.append(spec.color_rgb)
                        opacity_refs.append(spec.opacity)
                    setattr(ctx, "splat_saved", {
                        "mu": mu,
                        "theta": theta,
                        "sigma_x": sigma_x,
                        "sigma_y": sigma_y,
                        "color_rgb": color_rgb,
                        "opacity": opacity,
                        "tile_ptr": tile_ptr,
                        "tile_idx": tile_idx,
                        "tiles_x": tiles_x,
                        "tiles_y": tiles_y,
                        "tile_size": tile_size,
                        "width": request.width,
                        "height": request.height,
                        "spec_counts": spec_counts,
                        "color_refs": color_refs,
                        "opacity_refs": opacity_refs,
                        "order": order,
                    })
            if tile_size > 0:
                return _triton.composite_gaussians_tiled_triton(
                    mu, theta, sigma_x, sigma_y, color_rgb, opacity,
                    request.width, request.height, tile_size,
                )
            else:
                return _triton.composite_gaussians_full_triton(
                    mu, theta, sigma_x, sigma_y, color_rgb, opacity, request.width, request.height,
                    gchunk=gchunk,
                )
        except Exception as exc:
            _trace(f"triton compositor failed: {type(exc).__name__}")
            # fall through to torch path

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
    """Render backward pass.

    When running with Triton forward and tiling enabled, prefer a hybrid
    per-tile Torch backward that recomputes tiles sparsely using a CSR built
    from the current Gaussian set. This avoids the full-frame checkpointed
    recompute while keeping gradients flowing to original tensors.
    """
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

    # Optional hybrid tiled backward (Torch) when Triton is selected and tiling is on.
    def _hybrid_enabled() -> bool:
        wants_triton = _triton.env_wants_triton()
        # default enable when triton requested; can be forced off via env
        default_on = True if wants_triton else False
        return (int(request.config.tile) > 0) and _env_flag("DIFFVG_SPLAT_HYBRID_BWD", default_on)

    if _hybrid_enabled():
        try:
            # Assemble splats (differentiable w.r.t. original tensors)
            with torch.enable_grad():
                scene = request.scene
                device = get_device()
                dtype = torch.float32
                if scene.paths:
                    sample_dtype = scene.paths[0].points.dtype
                    if sample_dtype in (torch.float32, torch.float64):
                        dtype = sample_dtype
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
                mu = torch.cat([b.mu for b in batches], dim=0)
                theta = torch.cat([b.theta for b in batches], dim=0)
                sigma_x = torch.cat([b.sigma_x for b in batches], dim=0)
                sigma_y = torch.cat([b.sigma_y for b in batches], dim=0)
                color_rgb = torch.cat([b.color_rgb for b in batches], dim=0)
                opacity = torch.cat([b.opacity for b in batches], dim=0)
                if request.config.depth_policy == DepthPolicy.small_first:
                    order = torch.argsort(sigma_y)
                    mu = mu[order]; theta = theta[order]; sigma_x = sigma_x[order]; sigma_y = sigma_y[order]
                    color_rgb = color_rgb[order]; opacity = opacity[order]

                # Build CSR bins using detached copies (binning doesn't need gradients)
                tile_size = int(request.config.tile)
                tile_ptr, tile_idx, tiles_x, tiles_y = _triton._build_tile_csr(
                    mu.detach(), theta.detach(), sigma_x.detach(), sigma_y.detach(),
                    request.width, request.height, tile_size,
                )

                # Per-tile recompute and accumulate scalar loss
                grad_img_cast = grad_img.to(device=device, dtype=dtype).contiguous()
                loss = torch.zeros((), device=device, dtype=dtype)

                cos_theta = torch.cos(theta)
                sin_theta = torch.sin(theta)
                inv_sigma_x = 1.0 / sigma_x
                inv_sigma_y = 1.0 / sigma_y
                opacity_clamped = torch.clamp(opacity, 0.0, 1.0)

                # Gaussian chunk within a tile to control memory
                gchunk_env = os.environ.get("DIFFVG_SPLAT_TILE_GCHUNK", "128").strip() or "128"
                try:
                    gchunk = max(1, int(gchunk_env))
                except Exception:
                    gchunk = 128

                # Cached full grid to slice tiles
                full_gy, full_gx = _get_full_grid(request.height, request.width, device, dtype)

                for ty in range(int(tiles_y)):
                    for tx in range(int(tiles_x)):
                        tile_id = ty * int(tiles_x) + tx
                        s = int(tile_ptr[tile_id].item())
                        e = int(tile_ptr[tile_id + 1].item())
                        if s >= e:
                            continue
                        x0 = tx * tile_size
                        y0 = ty * tile_size
                        x1 = min(request.width, x0 + tile_size)
                        y1 = min(request.height, y0 + tile_size)
                        if x0 >= x1 or y0 >= y1:
                            continue
                        idx = tile_idx[s:e].to(torch.long)

                        grid_y = full_gy[y0:y1, x0:x1]
                        grid_x = full_gx[y0:y1, x0:x1]

                        # Local accumulators
                        tile_rgb = torch.zeros((y1 - y0, x1 - x0, 3), device=device, dtype=dtype)
                        tile_alpha = torch.zeros((y1 - y0, x1 - x0), device=device, dtype=dtype)

                        nbin = int(idx.numel())
                        for s2 in range(0, nbin, gchunk):
                            e2 = min(nbin, s2 + gchunk)
                            sel = idx[s2:e2]
                            m = int(sel.numel())
                            if m == 0:
                                continue
                            # Gather parameters
                            mu_c = mu[sel]              # [m,2]
                            ct = cos_theta[sel]         # [m]
                            st = sin_theta[sel]         # [m]
                            isx = inv_sigma_x[sel]      # [m]
                            isy = inv_sigma_y[sel]      # [m]
                            col = color_rgb[sel]        # [m,3]
                            opa = opacity_clamped[sel]  # [m]

                            dx = grid_x.unsqueeze(0) - mu_c[:, 0].view(m, 1, 1)
                            dy = grid_y.unsqueeze(0) - mu_c[:, 1].view(m, 1, 1)
                            lx = ct.view(m, 1, 1) * dx + st.view(m, 1, 1) * dy
                            ly = -st.view(m, 1, 1) * dx + ct.view(m, 1, 1) * dy
                            exponent = -0.5 * ((lx * isx.view(m, 1, 1)) ** 2 + (ly * isy.view(m, 1, 1)) ** 2)
                            a = torch.exp(exponent) * opa.view(m, 1, 1)
                            a = torch.clamp(a, 0.0, 1.0)

                            one_minus_a = 1.0 - a
                            P = torch.cumprod(one_minus_a, dim=0)
                            T = torch.cat([torch.ones(1, *P.shape[1:], device=device, dtype=dtype), P[:-1]], dim=0)

                            trans_prev = (1.0 - tile_alpha).unsqueeze(0)
                            w = (trans_prev * a * T)  # [m,Ht,Wt]
                            tile_rgb = tile_rgb + (w.unsqueeze(-1) * col.view(m, 1, 1, 3)).sum(dim=0)
                            prod_all = P[-1] if m > 0 else torch.ones_like(tile_alpha)
                            tile_alpha = tile_alpha + (1.0 - tile_alpha) * (1.0 - prod_all)

                        tile_img = torch.cat([tile_rgb, tile_alpha.unsqueeze(-1)], dim=-1)
                        gtile = grad_img_cast[y0:y1, x0:x1, :tile_img.shape[-1]]
                        loss = loss + torch.sum(tile_img * gtile)

            # Compute grads of original tensors from scalar loss
            targets = [slot.tensor for slot in grad_slots]
            active: List[Tuple[GradSlot, torch.Tensor]] = []
            for slot, tensor in zip(grad_slots, targets):
                if isinstance(tensor, torch.Tensor) and tensor.requires_grad:
                    active.append((slot, tensor))
            if active:
                active_slots, active_tensors = zip(*active)
                grads_active = torch.autograd.grad(loss, active_tensors, retain_graph=False, allow_unused=True)
                grads_active = tuple(
                    torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0) if isinstance(g, torch.Tensor) else g
                    for g in grads_active
                )
            else:
                active_slots = ()
                grads_active = ()

            trace_count = _increment_backward()
            if _should_print(trace_count):
                _trace(f"render_backward[{trace_count}] autograd_targets={len(targets)}")

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
        except Exception as exc:
            # Fall through to checkpointed full-frame path on any error
            _trace(f"hybrid backward disabled due to: {type(exc).__name__}")

    with torch.enable_grad():
        # Use checkpointed compositor to avoid retaining per-Gaussian activations.
        scene = request.scene
        device = get_device()
        dtype = torch.float32
        if scene.paths:
            sample_dtype = scene.paths[0].points.dtype
            if sample_dtype in (torch.float32, torch.float64):
                dtype = sample_dtype
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
        mu = torch.cat([b.mu for b in batches], dim=0)
        theta = torch.cat([b.theta for b in batches], dim=0)
        sigma_x = torch.cat([b.sigma_x for b in batches], dim=0)
        sigma_y = torch.cat([b.sigma_y for b in batches], dim=0)
        color_rgb = torch.cat([b.color_rgb for b in batches], dim=0)
        opacity = torch.cat([b.opacity for b in batches], dim=0)
        if request.config.depth_policy == DepthPolicy.small_first:
            order = torch.argsort(sigma_y)
            mu = mu[order]; theta = theta[order]; sigma_x = sigma_x[order]; sigma_y = sigma_y[order]
            color_rgb = color_rgb[order]; opacity = opacity[order]
        # Default chunk set to 128 for better painterly performance unless overridden
        chunk_env = os.environ.get("DIFFVG_SPLAT_CHUNK", "128").strip() or "128"
        try:
            chunk = int(chunk_env)
        except Exception:
            chunk = 128
        image = _composite_gaussians_full_ckpt(
            mu, theta, sigma_x, sigma_y, color_rgb, opacity,
            request.width, request.height, device, dtype, chunk=chunk
        )
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
            grads_active = torch.autograd.grad(loss, active_tensors, retain_graph=False, allow_unused=True)
            # Sanitize numerical issues: replace NaN/Inf in grads to prevent parameter blow-up
            grads_active = tuple(
                torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0) if isinstance(g, torch.Tensor) else g
                for g in grads_active
            )
        except RuntimeError as exc:  # pragma: no cover - defensive path, triggers fallback
            raise _SplatUnsupported(f"autograd failure: {exc}") from exc
    else:
        active_slots = ()
        grads_active = ()
    trace_count = _increment_backward()
    if _should_print(trace_count):
        _trace(f"render_backward[{trace_count}] autograd_targets={len(targets)}")

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
        keep_on_device: bool = True,
        device: Optional[torch.device | str] = None,
    ):
        # Force device-resident tensors for splat so autograd connectivity is preserved.
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
        return _render_forward(request, ctx)

    @staticmethod
    def backward(ctx, *grad_outputs: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        (grad_img,) = grad_outputs
        request: RenderRequest = ctx.request  # type: ignore[attr-defined]
        args = ctx.extra_args  # type: ignore[attr-defined]
        # Triton backward (experimental, color-only v0) if state saved and requested
        try:
            saved = getattr(ctx, "splat_saved", None)
            if saved is not None and _triton.env_wants_triton_backward():
                mu = saved["mu"]; theta = saved["theta"]; sigma_x = saved["sigma_x"]; sigma_y = saved["sigma_y"]
                color_rgb = saved["color_rgb"]; opacity = saved["opacity"]
                tile_ptr = saved["tile_ptr"]; tile_idx = saved["tile_idx"]
                width = int(saved["width"]); height = int(saved["height"]); tile_size = int(saved["tile_size"])
                # Triton backward (variant-selected)
                variant = getattr(_triton, "env_triton_backward_variant", lambda: "tile")()
                if _debug_enabled():
                    _trace(f"render_backward using triton variant={variant}")
                if variant == "pixel":
                    dcolor, dalpha, dmu_x, dmu_y, dtheta, disx, disy = _triton.backward_tiled_full_triton_pixel(
                        mu, theta, sigma_x, sigma_y, color_rgb, opacity,
                        tile_ptr, tile_idx, width, height, tile_size,
                        grad_img,
                    )
                else:
                    dcolor, dalpha, dmu_x, dmu_y, dtheta, disx, disy = _triton.backward_tiled_full_triton(
                        mu, theta, sigma_x, sigma_y, color_rgb, opacity,
                        tile_ptr, tile_idx, width, height, tile_size,
                        grad_img,
                    )
                # VJP bridge directly on saved autograd-connected tensors (no rebuild)
                # Convert inv-sigma grads to sigma grads
                dsx = -disx / (sigma_x.clamp_min(1e-6) ** 2)
                dsy = -disy / (sigma_y.clamp_min(1e-6) ** 2)
                args_with_grad, grad_slots = _enable_gradient_args(args)
                # Rebuild autograd-connected Gaussian params under grad-enabled context
                with torch.enable_grad():
                    vjp_req = _prepare_render_request(
                        request.width,
                        request.height,
                        request.num_samples_x,
                        request.num_samples_y,
                        request.seed,
                        request.background_image,
                        args_with_grad,
                    )
                    scene = vjp_req.scene
                    device = get_device()
                    dtype = torch.float32
                    if scene.paths:
                        sample_dtype = scene.paths[0].points.dtype
                        if sample_dtype in (torch.float32, torch.float64):
                            dtype = sample_dtype
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
                    mu_v = torch.cat([b.mu for b in batches], dim=0)
                    th_v = torch.cat([b.theta for b in batches], dim=0)
                    sx_v = torch.cat([b.sigma_x for b in batches], dim=0)
                    sy_v = torch.cat([b.sigma_y for b in batches], dim=0)
                    col_v = torch.cat([b.color_rgb for b in batches], dim=0)
                    opa_v = torch.cat([b.opacity for b in batches], dim=0)
                    order_saved = saved.get("order", None)
                    if order_saved is not None:
                        mu_v = mu_v[order_saved]
                        th_v = th_v[order_saved]
                        sx_v = sx_v[order_saved]
                        sy_v = sy_v[order_saved]
                        col_v = col_v[order_saved]
                        opa_v = opa_v[order_saved]
                # Map Triton grads to inputs via VJP
                vjp_all = [mu_v, th_v, sx_v, sy_v, col_v, opa_v]
                g_mu = torch.stack([dmu_x, dmu_y], dim=-1)
                gout_all = [g_mu, dtheta, dsx, dsy, dcolor, dalpha]
                vjp_tensors = []
                grad_outputs = []
                for t, g in zip(vjp_all, gout_all):
                    if isinstance(t, torch.Tensor) and t.requires_grad:
                        vjp_tensors.append(t)
                        grad_outputs.append(g)
                input_tensors = [slot.tensor for slot in grad_slots]
                vjp_grads = torch.autograd.grad(vjp_tensors, input_tensors, grad_outputs=grad_outputs, allow_unused=True)
                # Guard: if Triton produced effectively near-zero grads, fall back
                total = 0.0
                for g in vjp_grads:
                    if isinstance(g, torch.Tensor):
                        total += float(torch.sum(torch.abs(g)).item())
                if _debug_enabled():
                    _trace(f"backward-check grad_sum={total:.3e}")
                if (not math.isfinite(total)) or (total < 1e-8):
                    raise RuntimeError("near-zero grads from Triton VJP")
                total_args = len(args)
                grad_list: List[Optional[torch.Tensor]] = [None] * (6 + total_args)
                for slot, g in zip(grad_slots, vjp_grads):
                    if g is None:
                        grad_list[6 + slot.arg_index] = torch.zeros_like(slot.tensor)
                    else:
                        grad_list[6 + slot.arg_index] = g.detach()
                return tuple(grad_list)
        except Exception as exc:
            # Optional retry via per-pixel variant only if explicitly requested
            if os.environ.get("DIFFVG_SPLAT_RETRY_PIXEL", "0").strip().lower() in ("1", "true", "on"):
                try:
                    if _debug_enabled():
                        _trace(f"triton backward unavailable ({type(exc).__name__}: {exc}); retrying pixel variant")
                    saved = getattr(ctx, "splat_saved", None)
                    if saved is not None and _triton.is_available():
                        mu = saved["mu"]; theta = saved["theta"]; sigma_x = saved["sigma_x"]; sigma_y = saved["sigma_y"]
                        color_rgb = saved["color_rgb"]; opacity = saved["opacity"]
                        tile_ptr = saved["tile_ptr"]; tile_idx = saved["tile_idx"]
                        width = int(saved["width"]); height = int(saved["height"]); tile_size = int(saved["tile_size"])
                        dcolor, dalpha, dmu_x, dmu_y, dtheta, disx, disy = _triton.backward_tiled_full_triton_pixel(
                            mu, theta, sigma_x, sigma_y, color_rgb, opacity,
                            tile_ptr, tile_idx, width, height, tile_size,
                            grad_img,
                        )
                        dsx = -disx / (sigma_x.clamp_min(1e-6) ** 2)
                        dsy = -disy / (sigma_y.clamp_min(1e-6) ** 2)
                        args_with_grad, grad_slots = _enable_gradient_args(args)
                        with torch.enable_grad():
                            vjp_req = _prepare_render_request(
                                request.width,
                                request.height,
                                request.num_samples_x,
                                request.num_samples_y,
                                request.seed,
                                request.background_image,
                                args_with_grad,
                            )
                            scene = vjp_req.scene
                            device = get_device()
                            dtype = torch.float32
                            if scene.paths:
                                sample_dtype = scene.paths[0].points.dtype
                                if sample_dtype in (torch.float32, torch.float64):
                                    dtype = sample_dtype
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
                            mu_v = torch.cat([b.mu for b in batches], dim=0)
                            th_v = torch.cat([b.theta for b in batches], dim=0)
                            sx_v = torch.cat([b.sigma_x for b in batches], dim=0)
                            sy_v = torch.cat([b.sigma_y for b in batches], dim=0)
                            col_v = torch.cat([b.color_rgb for b in batches], dim=0)
                            opa_v = torch.cat([b.opacity for b in batches], dim=0)
                            order_saved = saved.get("order", None)
                            if order_saved is not None:
                                mu_v = mu_v[order_saved]
                                th_v = th_v[order_saved]
                                sx_v = sx_v[order_saved]
                                sy_v = sy_v[order_saved]
                                col_v = col_v[order_saved]
                                opa_v = opa_v[order_saved]
                        vjp_all = [mu_v, th_v, sx_v, sy_v, col_v, opa_v]
                        g_mu = torch.stack([dmu_x, dmu_y], dim=-1)
                        gout_all = [g_mu, dtheta, dsx, dsy, dcolor, dalpha]
                        vjp_tensors = []
                        grad_outputs = []
                        for t, g in zip(vjp_all, gout_all):
                            if isinstance(t, torch.Tensor) and t.requires_grad:
                                vjp_tensors.append(t)
                                grad_outputs.append(g)
                        input_tensors = [slot.tensor for slot in grad_slots]
                        vjp_grads = torch.autograd.grad(vjp_tensors, input_tensors, grad_outputs=grad_outputs, allow_unused=True)
                        total = 0.0
                        for g in vjp_grads:
                            if isinstance(g, torch.Tensor):
                                total += float(torch.sum(torch.abs(g)).item())
                        if _debug_enabled():
                            _trace(f"backward-check pixel grad_sum={total:.3e}")
                        if (not math.isfinite(total)) or (total < 1e-8):
                            raise RuntimeError("near-zero grads from Triton pixel VJP")
                        total_args = len(args)
                        grad_list: List[Optional[torch.Tensor]] = [None] * (6 + total_args)
                        for slot, g in zip(grad_slots, vjp_grads):
                            grad_list[6 + slot.arg_index] = (torch.zeros_like(slot.tensor) if g is None else g.detach())
                        return tuple(grad_list)
                except Exception as exc2:
                    if _debug_enabled():
                        _trace(f"triton pixel retry failed ({type(exc2).__name__}: {exc2})")
            _trace(f"triton backward unavailable: {type(exc).__name__}: {exc}")
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
