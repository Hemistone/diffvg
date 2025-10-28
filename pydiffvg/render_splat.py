from __future__ import annotations

import math
import os
from typing import Iterable, List, Optional, Tuple, TYPE_CHECKING
from types import SimpleNamespace

import torch

from .backend import DepthPolicy
from .device import get_device
from .render_pytorch import OutputType, BaselineRenderFunction as _BaselineRF
from .serialization import serialize_scene as _serialize_scene
from . import triton_splat as _triton
from .splat.types import (
    _SplatUnsupported,
)

# Static type-checking only (avoid runtime imports/cycles)
if TYPE_CHECKING:  # pragma: no cover - typing aid
    from .splat.types import RenderRequest, GaussianBatch, GradSlot
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
)
from .splat.vjp import (
    _align_grad_devices,
    _cpu_args,
    _enable_gradient_args,
    _scene_requires_grad,
)
from .splat.mapping import build_splat_mapping_payload, map_triton_grads_to_slots
from .splat.debug import backward_tiled_full_python as _backward_tiled_full_python
from .splat.scene import _prepare_render_request


# Scene (de)serialization and request prep moved to pydiffvg/splat/scene.py


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
    seg_meta: List[Optional[torch.Tensor]] = []
    t_meta: List[Optional[torch.Tensor]] = []
    pts_meta: List[Optional[torch.Tensor]] = []
    cc_meta: List[List[int]] = []
    if stroke_specs:
        for spec in stroke_specs:
            gb = _path_to_gaussians(spec, request.config, device, dtype, request.generator)
            batches.append(gb)
            seg_meta.append(gb.seg_idx)
            t_meta.append(gb.t)
            cc_meta.append([len(seg.controls) for seg in spec.segments])
            try:
                pref = next(p.points for p in scene.paths if p.shape_id == spec.shape_id)
            except StopIteration:
                pref = None
            pts_meta.append(pref)
    if fill_specs:
        for spec in fill_specs:
            gb = _fill_to_gaussians(spec, request.config, device, dtype, request.generator)
            batches.append(gb)
            seg_meta.append(gb.seg_idx)
            t_meta.append(gb.t)
            cc_meta.append([len(seg.controls) for seg in spec.segments])
            try:
                pref = next(p.points for p in scene.paths if p.shape_id == spec.shape_id)
            except StopIteration:
                pref = None
            pts_meta.append(pref)
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
                # Save per-spec counts and references to original input tensors for fused VJP
                try:
                    spec_counts: List[int] = [int(b.mu.shape[0]) for b in batches]
                    # Map shape_id -> color tensor for stroke/fill from scene
                    stroke_color_ref: dict[int, torch.Tensor] = {}
                    fill_color_ref: dict[int, torch.Tensor] = {}
                    for group in scene.shape_groups:
                        if group.stroke.color_type is not None:
                            for sid in group.shape_ids.to(torch.int64).tolist():
                                stroke_color_ref[sid] = group.stroke.params[0]
                        if group.fill.color_type is not None:
                            for sid in group.shape_ids.to(torch.int64).tolist():
                                fill_color_ref[sid] = group.fill.params[0]
                    stroke_width_ref: dict[int, Optional[torch.Tensor]] = {}
                    for p in scene.paths:
                        stroke_width_ref[p.shape_id] = p.stroke_width
                    color_rgba_refs: List[torch.Tensor] = []
                    stroke_width_refs: List[Optional[torch.Tensor]] = []
                    for spec in stroke_specs:
                        color_rgba_refs.append(stroke_color_ref.get(spec.shape_id))
                        stroke_width_refs.append(stroke_width_ref.get(spec.shape_id))
                    for spec in fill_specs:
                        color_rgba_refs.append(fill_color_ref.get(spec.shape_id))
                        stroke_width_refs.append(None)
                    payload["spec_counts"] = spec_counts
                    payload["color_rgba_refs"] = color_rgba_refs
                    payload["stroke_width_refs"] = stroke_width_refs
                    payload["seg_idx_list"] = seg_meta
                    payload["t_list"] = t_meta
                    payload["points_refs"] = pts_meta
                    payload["control_counts"] = cc_meta
                    payload.update(
                        build_splat_mapping_payload(
                            batches,
                            len(stroke_specs),
                            cc_meta,
                            mu,
                            spec_counts,
                            pts_meta,
                        )
                    )
                except Exception:
                    pass
                setattr(ctx, "splat_saved", payload)
            except Exception as _exc:
                if _debug_enabled():
                    _trace(f"forward save skipped: {type(_exc).__name__}")
        # Prefer Triton tiled compositor when available (strict only if explicitly requested)
        if _triton.is_available() and device.type == "cuda":
            try:
                return _triton.composite_gaussians_tiled_triton(
                    mu,
                    theta,
                    sigma_x,
                    sigma_y,
                    color_rgb,
                    opacity,
                    request.width,
                    request.height,
                    tile_size,
                )
            except Exception as exc:
                # Only treat as fatal if explicitly requested via env
                if _triton.env_wants_triton():
                    raise
                _trace(f"triton tiled compositor failed: {type(exc).__name__}")

        # Fallback to Torch tiled-diff compositor under grad
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
    # Prefer Triton full-frame compositor when available (strict only if explicitly requested)
    if _triton.is_available() and device.type == "cuda":
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
                    # Build refs to original scene tensors
                    stroke_color_ref: dict[int, torch.Tensor] = {}
                    fill_color_ref: dict[int, torch.Tensor] = {}
                    for group in scene.shape_groups:
                        if group.stroke.color_type is not None:
                            for sid in group.shape_ids.to(torch.int64).tolist():
                                stroke_color_ref[sid] = group.stroke.params[0]
                        if group.fill.color_type is not None:
                            for sid in group.shape_ids.to(torch.int64).tolist():
                                fill_color_ref[sid] = group.fill.params[0]
                    stroke_width_ref: dict[int, Optional[torch.Tensor]] = {}
                    for p in scene.paths:
                        stroke_width_ref[p.shape_id] = p.stroke_width
                    color_rgba_refs: List[torch.Tensor] = []
                    stroke_width_refs: List[Optional[torch.Tensor]] = []
                    for spec in stroke_specs:
                        color_rgba_refs.append(stroke_color_ref.get(spec.shape_id))
                        stroke_width_refs.append(stroke_width_ref.get(spec.shape_id))
                    for spec in fill_specs:
                        color_rgba_refs.append(fill_color_ref.get(spec.shape_id))
                        stroke_width_refs.append(None)
                    # no-op: per-sample index caching removed (negligible benefit)
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
                        "color_rgba_refs": color_rgba_refs,
                        "order": order,
                        "stroke_width_refs": stroke_width_refs,
                        "seg_idx_list": seg_meta,
                        "t_list": t_meta,
                        "points_refs": pts_meta,
                        "control_counts": cc_meta,
                        **build_splat_mapping_payload(
                            batches,
                            len(stroke_specs),
                            cc_meta,
                            mu,
                            spec_counts,
                            pts_meta,
                        ),
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
            if _triton.env_wants_triton():
                raise
            _trace(f"triton compositor failed: {type(exc).__name__}")

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


def _map_triton_grads_to_slots(
    saved: dict,
    request: RenderRequest,
    args_with_grad: Tuple[object, ...],
    grad_slots: List[GradSlot],
    dcolor: torch.Tensor,
    dalpha: torch.Tensor,
    dmu_x: torch.Tensor,
    dmu_y: torch.Tensor,
    dtheta: torch.Tensor,
    dsx: torch.Tensor,
    dsy: torch.Tensor,
) -> Optional[Tuple[Optional[torch.Tensor], ...]]:
    return map_triton_grads_to_slots(
        saved,
        request,
        args_with_grad,
        grad_slots,
        dcolor,
        dalpha,
        dmu_x,
        dmu_y,
        dtheta,
        dsx,
        dsy,
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

    # Remove legacy hybrid Torch backward path; rely on Triton or checkpointed path below

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
            if saved is not None and _triton.is_available():
                mu = saved["mu"]; theta = saved["theta"]; sigma_x = saved["sigma_x"]; sigma_y = saved["sigma_y"]
                color_rgb = saved["color_rgb"]; opacity = saved["opacity"]
                tile_ptr = saved["tile_ptr"]; tile_idx = saved["tile_idx"]
                width = int(saved["width"]); height = int(saved["height"]); tile_size = int(saved["tile_size"])
                dcolor, dalpha, dmu_x, dmu_y, dtheta, disx, disy = _triton.backward_tiled_full_triton(
                    mu, theta, sigma_x, sigma_y, color_rgb, opacity,
                    tile_ptr, tile_idx, width, height, tile_size,
                    grad_img,
                )
                triton_total = (
                    dcolor.abs().sum()
                    + dalpha.abs().sum()
                    + dmu_x.abs().sum()
                    + dmu_y.abs().sum()
                    + dtheta.abs().sum()
                    + disx.abs().sum()
                    + disy.abs().sum()
                )
                # Robust fallback to Python reference if non-finite or near-zero
                need_fallback = (not torch.isfinite(triton_total)) or float(triton_total.detach().cpu()) < 1e-9
                strict_bwd = os.environ.get("DIFFVG_SPLAT_BWD", "").strip().lower() == "triton"
                if need_fallback:
                    if strict_bwd:
                        raise RuntimeError("Triton backward produced non-finite or near-zero grads under strict mode")
                    if _debug_enabled():
                        _trace("render_backward detected near-zero or non-finite Triton grads; falling back to python reference")
                    (
                        dcolor,
                        dalpha,
                        dmu_x,
                        dmu_y,
                        dtheta,
                        disx,
                        disy,
                    ) = _backward_tiled_full_python(
                        mu, theta, sigma_x, sigma_y, color_rgb, opacity,
                        tile_ptr, tile_idx, width, height, tile_size,
                        grad_img,
                    )

                # End Triton backward block
                # Convert inv-sigma grads to sigma grads
                dsx = -disx / (sigma_x.clamp_min(1e-6) ** 2)
                dsy = -disy / (sigma_y.clamp_min(1e-6) ** 2)
                # Prepare grad slots and fused mappings for color/opacity/width
                args_with_grad, grad_slots = _enable_gradient_args(args)
                mapped = _map_triton_grads_to_slots(
                    saved,
                    request,
                    args_with_grad,
                    grad_slots,
                    dcolor,
                    dalpha,
                    dmu_x,
                    dmu_y,
                    dtheta,
                    dsx,
                    dsy,
                )
                if mapped is not None:
                    return mapped
                fused_map: dict[int, torch.Tensor] = {}
                try:
                    spec_counts = saved.get("spec_counts", None)
                    color_rgba_refs = saved.get("color_rgba_refs", None)
                    width_refs = saved.get("stroke_width_refs", None)
                    seg_idx_list = saved.get("seg_idx_list", None)
                    t_list = saved.get("t_list", None)
                    points_refs = saved.get("points_refs", None)
                    control_counts_list = saved.get("control_counts", None)
                    if spec_counts is not None and color_rgba_refs is not None:
                        # bring grads back to pre-sort order for per-spec grouping
                        order_saved = saved.get("order", None)
                        if isinstance(order_saved, torch.Tensor):
                            inv = torch.empty_like(order_saved)
                            inv[order_saved] = torch.arange(order_saved.numel(), device=order_saved.device)
                            dcolor_g = dcolor[inv]
                            dalpha_g = dalpha[inv]
                            dsy_g = dsy[inv]
                            dsx_g = dsx[inv]
                            dmu_x_g = dmu_x[inv]
                            dmu_y_g = dmu_y[inv]
                            dtheta_g = dtheta[inv]
                            mu_g = mu[inv]
                            sx_g = sigma_x[inv]
                        else:
                            dcolor_g = dcolor
                            dalpha_g = dalpha
                            dsy_g = dsy
                            dsx_g = dsx
                            dmu_x_g = dmu_x
                            dmu_y_g = dmu_y
                            dtheta_g = dtheta
                            mu_g = mu
                            sx_g = sigma_x
                        # width scale from sigma_y = width / (fwhm_coeff * rho)
                        rho = max(float(request.config.rho), 1e-6)
                        fwhm_coeff = 2.0 * math.sqrt(2.0 * math.log(2.0))
                        width_scale = 1.0 / (fwhm_coeff * rho)
                        idx0 = 0
                        for si, cnt in enumerate(spec_counts):
                            cnt = int(cnt)
                            idx1 = idx0 + cnt
                            if idx1 > idx0:
                                # color/opacity accumulated per spec
                                gcol = dcolor_g[idx0:idx1].sum(dim=0)
                                gopa = dalpha_g[idx0:idx1].sum()
                                cref = color_rgba_refs[si]
                                if isinstance(cref, torch.Tensor):
                                    # assemble RGBA gradient to match the original color tensor shape
                                    # try to handle [4] or [1,4] shapes
                                    if cref.shape[-1] != 4:
                                        grad_rgba = torch.cat([gcol, gopa.view(1)], dim=0)
                                    else:
                                        grad_rgba = torch.cat([gcol, gopa.view(1)], dim=0)
                                    grad_rgba = grad_rgba.view_as(cref)
                                    fused_map[id(cref)] = grad_rgba
                                # stroke width (paths only)
                                if width_refs is not None:
                                    wref = width_refs[si]
                                    if isinstance(wref, torch.Tensor):
                                        gw = dsy_g[idx0:idx1].sum() * width_scale
                                        fused_map[id(wref)] = gw.expand_as(wref)
                                # geometry fused VJP to points
                                if (
                                    isinstance(points_refs, list)
                                    and isinstance(control_counts_list, list)
                                    and isinstance(seg_idx_list, list)
                                    and isinstance(t_list, list)
                                ):
                                    pref = points_refs[si]
                                    cc = control_counts_list[si]
                                    seg_idx_spec = seg_idx_list[si]
                                    tvals_spec = t_list[si]
                                    if isinstance(pref, torch.Tensor) and isinstance(seg_idx_spec, torch.Tensor) and isinstance(tvals_spec, torch.Tensor):
                                        # Build per-sample total g_mu including sigma_x contribution
                                        mu_spec = mu_g[idx0:idx1]
                                        gmu = torch.stack([dmu_x_g[idx0:idx1], dmu_y_g[idx0:idx1]], dim=-1)
                                        extra = torch.zeros_like(gmu)
                                        if cnt >= 2:
                                            # unit vectors along segments between neighboring samples
                                            d = mu_spec[1:] - mu_spec[:-1]
                                            den = torch.clamp(torch.linalg.norm(d, dim=-1, keepdim=True), min=1e-6)
                                            u = d / den
                                            # spacing gradient from sigma_x: s = rho * sigma_x → dL/ds = (1/rho) * dL/dsigma_x
                                            # apply clamp mask: if sigma_x clamped at min, zero its gradient contribution
                                            sx_local = sx_g[idx0:idx1]
                                            clamp_mask = (sx_local <= (1e-3 + 1e-12))
                                            dsx_local = dsx_g[idx0:idx1] * (1.0 / rho)
                                            dsx_local = torch.where(clamp_mask, torch.zeros_like(dsx_local), dsx_local)
                                            # Per-segment gradient: interior seg k gets 0.5*dsx[k] + 0.5*dsx[k+1]
                                            # Boundary segments also pick up the extra 0.5 for their single-sided sample
                                            g_to_dist = 0.5 * dsx_local[:-1] + 0.5 * dsx_local[1:]
                                            if cnt >= 2:
                                                g_to_dist[0] = g_to_dist[0] + 0.5 * dsx_local[0]
                                                g_to_dist[-1] = g_to_dist[-1] + 0.5 * dsx_local[-1]
                                            extra[:-1] += -u * g_to_dist.unsqueeze(-1)
                                            extra[1:]  +=  u * g_to_dist.unsqueeze(-1)
                                        gmu = gmu + extra
                                        # Compute gradients to points via cached indices + vectorized weights
                                        D = pref.device
                                        gp = torch.zeros_like(pref, device=D)
                                        sidx = seg_idx_spec.to(torch.int64)
                                        tval = tvals_spec.to(pref.dtype)
                                        omt = 1.0 - tval
                                        cc_t = torch.tensor(cc, device=D, dtype=torch.int64)
                                        S = cc_t.numel()
                                        s_ids = torch.arange(S, device=D, dtype=torch.int64)
                                        csum = torch.cumsum(cc_t, dim=0)
                                        si0_arr = s_ids + torch.cat([torch.zeros((1,), device=D, dtype=torch.int64), csum[:-1]], dim=0)
                                        si_end_arr = (s_ids + 1) + csum
                                        deg = cc_t[sidx]
                                        si0_all = si0_arr[sidx]
                                        si_end_all = si_end_arr[sidx]
                                        m0 = (deg == 0)
                                        m1 = (deg == 1)
                                        m2 = (deg >= 2)
                                        # Position weights
                                        if m0.any():
                                            w0 = omt[m0]
                                            w1 = tval[m0]
                                            si0 = si0_all[m0]
                                            si_end = si_end_all[m0]
                                            g = gmu[m0]
                                            gp.index_add_(0, si0, g * w0.unsqueeze(-1))
                                            gp.index_add_(0, si_end, g * w1.unsqueeze(-1))
                                        if m1.any():
                                            tt = tval[m1]; oo = omt[m1]
                                            w0 = oo * oo
                                            w1 = 2.0 * oo * tt
                                            w2 = tt * tt
                                            base = si0_all[m1]
                                            ci0 = base + 1
                                            si0 = base
                                            si_end = si_end_all[m1]
                                            g = gmu[m1]
                                            gp.index_add_(0, si0, g * w0.unsqueeze(-1))
                                            gp.index_add_(0, ci0, g * w1.unsqueeze(-1))
                                            gp.index_add_(0, si_end, g * w2.unsqueeze(-1))
                                        if m2.any():
                                            tt = tval[m2]; oo = omt[m2]
                                            oo2 = oo * oo; t2 = tt * tt
                                            w0 = oo2 * oo
                                            w1 = 3.0 * oo2 * tt
                                            w2 = 3.0 * oo * t2
                                            w3 = t2 * tt
                                            base = si0_all[m2]
                                            ci0 = base + 1
                                            ci1 = base + 2
                                            si0 = base
                                            si_end = si_end_all[m2]
                                            g = gmu[m2]
                                            gp.index_add_(0, si0, g * w0.unsqueeze(-1))
                                            gp.index_add_(0, ci0, g * w1.unsqueeze(-1))
                                            gp.index_add_(0, ci1, g * w2.unsqueeze(-1))
                                            gp.index_add_(0, si_end, g * w3.unsqueeze(-1))
                                        # Theta contribution
                                        gth_all = dtheta_g[idx0:idx1]
                                        if torch.any(gth_all != 0):
                                            ptsd = pref.detach()
                                            cnt_i = cnt
                                            j_idx = torch.arange(cnt_i, device=D)
                                            mids = (j_idx >= 1) & (j_idx <= (cnt_i - 2))
                                            use_central = torch.zeros((cnt_i,), dtype=torch.bool, device=D)
                                            if cnt_i >= 3:
                                                # central-diff vector c = mu[j+1] - mu[j-1]
                                                c = mu_spec[2:] - mu_spec[:-2]
                                                cn2 = (c[:, 0] * c[:, 0] + c[:, 1] * c[:, 1])
                                                good = cn2 > 1e-8
                                                use_central[1:-1] = good
                                                # gθ wrt c: [-c_y, c_x] / ||c||^2
                                                gth_mid = gth_all[1:-1]
                                                if good.any():
                                                    cg = c[good]
                                                    cn2g = cn2[good]
                                                    gvec = torch.stack([-cg[:, 1] / cn2g, cg[:, 0] / cn2g], dim=-1) * gth_mid[good].unsqueeze(-1)
                                                    # distribute to neighbors: μ[j+1] += gvec, μ[j-1] -= gvec
                                                    idxs = torch.nonzero(good, as_tuple=False).squeeze(1)
                                                    idx_plus = idxs + 2  # shift back to absolute indices j+1
                                                    idx_minus = idxs     # absolute indices j-1 (since good aligns with c at positions 1..cnt-2 mapped to 0..cnt-3)
                                                    # Prepare zero gmu_add then index_add
                                                    gmu_add = torch.zeros_like(gmu)
                                                    gmu_add.index_add_(0, idx_plus, gvec)
                                                    gmu_add.index_add_(0, idx_minus, -gvec)
                                                    gmu = gmu + gmu_add
                                            # Tangent fallback (endpoints + bad mids) if any
                                            tan_mask = (~use_central)
                                            if tan_mask.any():
                                                m0_t = m0 & tan_mask
                                                m1_t = m1 & tan_mask
                                                m2_t = m2 & tan_mask
                                                if m0_t.any():
                                                    si0 = si0_all[m0_t]
                                                    si_end = si_end_all[m0_t]
                                                    v = ptsd[si_end] - ptsd[si0]
                                                    denom = (v[:, 0] * v[:, 0] + v[:, 1] * v[:, 1]).clamp_min(1e-6)
                                                    gv = torch.stack([-v[:, 1] / denom, v[:, 0] / denom], dim=-1) * gth_all[m0_t].unsqueeze(-1)
                                                    gp.index_add_(0, si0, gv * (-1.0))
                                                    gp.index_add_(0, si_end, gv * (1.0))
                                                if m1_t.any():
                                                    base = si0_all[m1_t]
                                                    si0 = base
                                                    ci0 = base + 1
                                                    si_end = si_end_all[m1_t]
                                                    tt = tval[m1_t]; oo = omt[m1_t]
                                                    v = 2.0 * (oo.unsqueeze(-1) * (ptsd[ci0] - ptsd[si0]) + tt.unsqueeze(-1) * (ptsd[si_end] - ptsd[ci0]))
                                                    denom = (v[:, 0] * v[:, 0] + v[:, 1] * v[:, 1]).clamp_min(1e-6)
                                                    gv = torch.stack([-v[:, 1] / denom, v[:, 0] / denom], dim=-1) * gth_all[m1_t].unsqueeze(-1)
                                                    dt0 = -2.0 * oo
                                                    dt1 = 2.0 * (1.0 - 2.0 * tt)
                                                    dt2 = 2.0 * tt
                                                    gp.index_add_(0, si0, gv * dt0.unsqueeze(-1))
                                                    gp.index_add_(0, ci0, gv * dt1.unsqueeze(-1))
                                                    gp.index_add_(0, si_end, gv * dt2.unsqueeze(-1))
                                                if m2_t.any():
                                                    base = si0_all[m2_t]
                                                    si0 = base
                                                    ci0 = base + 1
                                                    ci1 = base + 2
                                                    si_end = si_end_all[m2_t]
                                                    tt = tval[m2_t]; oo = omt[m2_t]
                                                    oo2 = oo * oo; t2 = tt * tt
                                                    v = (
                                                        3.0 * oo2.unsqueeze(-1) * (ptsd[ci0] - ptsd[si0])
                                                        + 6.0 * (oo * tt).unsqueeze(-1) * (ptsd[ci1] - ptsd[ci0])
                                                        + 3.0 * t2.unsqueeze(-1) * (ptsd[si_end] - ptsd[ci1])
                                                    )
                                                    denom = (v[:, 0] * v[:, 0] + v[:, 1] * v[:, 1]).clamp_min(1e-6)
                                                    gv = torch.stack([-v[:, 1] / denom, v[:, 0] / denom], dim=-1) * gth_all[m2_t].unsqueeze(-1)
                                                    dt0 = -3.0 * oo2
                                                    dt1 = 3.0 * oo2 - 6.0 * oo * tt
                                                    dt2 = 6.0 * oo * tt - 3.0 * t2
                                                    dt3 = 3.0 * t2
                                                    gp.index_add_(0, si0, gv * dt0.unsqueeze(-1))
                                                    gp.index_add_(0, ci0, gv * dt1.unsqueeze(-1))
                                                    gp.index_add_(0, ci1, gv * dt2.unsqueeze(-1))
                                                    gp.index_add_(0, si_end, gv * dt3.unsqueeze(-1))
                                        # Register fused grad for this points tensor
                                        fused_map[id(pref)] = gp
                            idx0 = idx1
                except Exception:
                    # best-effort fused mapping; fall back to autograd for any missing ones
                    pass
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
                    # color/opacity VJP is handled via fused_map; keep tensors available but exclude from outputs
                    col_v = torch.cat([b.color_rgb for b in batches], dim=0)
                    opa_v = torch.cat([b.opacity for b in batches], dim=0)
                    order_saved = saved.get("order", None)
                    if order_saved is not None:
                        mu_v = mu_v[order_saved]
                        th_v = th_v[order_saved]
                        sx_v = sx_v[order_saved]
                        sy_v = sy_v[order_saved]
                        # col_v/opa_v ordering not needed for fused mapping
                # If fused map covers all active slots, skip autograd VJP entirely
                input_tensors = [slot.tensor for slot in grad_slots]
                if all(id(t) in fused_map for t in input_tensors):
                    total_args = len(args)
                    grad_list: List[Optional[torch.Tensor]] = [None] * (6 + total_args)
                    for slot in grad_slots:
                        grad_list[6 + slot.arg_index] = fused_map[id(slot.tensor)].detach()
                    return tuple(grad_list)
                # Otherwise, fall back to VJP for the remaining ones
                vjp_all = [mu_v, th_v, sx_v, sy_v]
                g_mu = torch.stack([dmu_x, dmu_y], dim=-1)
                gout_all = [g_mu, dtheta, dsx, dsy]
                vjp_tensors = []
                grad_outputs = []
                for t, g in zip(vjp_all, gout_all):
                    if isinstance(t, torch.Tensor) and t.requires_grad:
                        vjp_tensors.append(t)
                        grad_outputs.append(g)
                vjp_grads = torch.autograd.grad(vjp_tensors, input_tensors, grad_outputs=grad_outputs, allow_unused=True)
                total_args = len(args)
                grad_list: List[Optional[torch.Tensor]] = [None] * (6 + total_args)
                for slot, g in zip(grad_slots, vjp_grads):
                    override = fused_map.get(id(slot.tensor))
                    if override is not None:
                        grad_list[6 + slot.arg_index] = override.detach()
                    elif g is None:
                        grad_list[6 + slot.arg_index] = torch.zeros_like(slot.tensor)
                    else:
                        grad_list[6 + slot.arg_index] = g.detach()
                return tuple(grad_list)
        except Exception as exc:
            # If user explicitly requested Triton backward via env, do not silently fallback
            strict_bwd = os.environ.get("DIFFVG_SPLAT_BWD", "").strip().lower() == "triton"
            if strict_bwd:
                raise
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
]
