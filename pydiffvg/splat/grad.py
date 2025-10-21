from __future__ import annotations

import os
from typing import Iterable, List, Optional, Tuple

import torch

from ..backend import DepthPolicy
from ..device import get_device
from ..triton_splat import _build_tile_csr
from .env import _env_flag
from .geometry import _gather_specs
from .gauss import _fill_to_gaussians, _path_to_gaussians
from .compositor import _get_full_grid
from .trace import trace as _trace, should_print as _should_print, increment_backward as _increment_backward
from .types import GaussianBatch, GradSlot
from .vjp import _enable_gradient_args


def hybrid_backward_tiled_torch(
    request,
    grad_img: torch.Tensor,
    args: Iterable[object],
) -> Tuple[Optional[torch.Tensor], ...]:
    """Hybrid per-tile backward using Torch recompute under autograd.

    Rebuilds gaussians with gradients, bins into tiles, recomputes tiles and
    accumulates a scalar loss against grad_img; then calls autograd.grad to
    obtain gradients w.r.t. original input tensors.

    Returns a gradient tuple matching the baseline backward signature.
    Raises on error to allow caller to fallback.
    """
    # Gate via env (default-on when Triton is requested from render_splat)
    def _hybrid_enabled(default_on: bool = True) -> bool:
        return (int(request.config.tile) > 0) and _env_flag("DIFFVG_SPLAT_HYBRID_BWD", default_on)

    if not _hybrid_enabled():
        # Indicate to caller that hybrid path isn't active
        raise RuntimeError("hybrid tiled backward not enabled")

    # Assemble splats (differentiable w.r.t. original tensors)
    args_with_grad, grad_slots = _enable_gradient_args(args)
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
        tile_ptr, tile_idx, tiles_x, tiles_y = _build_tile_csr(
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


__all__ = [
    "hybrid_backward_tiled_torch",
]

