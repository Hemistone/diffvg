from __future__ import annotations

import math
from functools import lru_cache
from typing import Optional

import torch

from ..backend import BezierGsplatConfig
from ..bezier_gsplat.runtime import load_gsplat_ops
from ..device import get_device
from .compiled import CompiledOpenStrokeScene, OpenStrokeUnsupported


@lru_cache(maxsize=16)
def _sample_basis(sample_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    t = (torch.arange(sample_count, dtype=torch.float32) + 0.5) / float(sample_count)
    omt = 1.0 - t
    basis = torch.stack(
        [omt ** 3, 3.0 * omt ** 2 * t, 3.0 * omt * t ** 2, t ** 3],
        dim=1,
    )
    deriv = torch.stack(
        [
            -3.0 * omt ** 2,
            3.0 * omt ** 2 - 6.0 * omt * t,
            6.0 * omt * t - 3.0 * t ** 2,
            3.0 * t ** 2,
        ],
        dim=1,
    )
    return basis, deriv


def _basis_on(device: torch.device, dtype: torch.dtype, sample_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    basis, deriv = _sample_basis(sample_count)
    return basis.to(device=device, dtype=dtype), deriv.to(device=device, dtype=dtype)


def _project_with_safe_tile_budget(
    height: int,
    width: int,
    block_h: int,
    block_w: int,
    means_ndc: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    ops,
):
    height = max(int(height), 1)
    width = max(int(width), 1)
    block_h = max(int(block_h), 1)
    block_w = max(int(block_w), 1)
    while True:
        tile_bounds = ((width + block_w - 1) // block_w, (height + block_h - 1) // block_h, 1)
        xys, depths, radii, conics, num_tiles_hit = ops.project_gaussians_2d_scale_rot(
            means_ndc,
            scales,
            rotations,
            height,
            width,
            tile_bounds,
        )
        num_tiles = tile_bounds[0] * tile_bounds[1]
        if int(num_tiles_hit.detach().sum().to(device="cpu").item()) >= num_tiles or (block_h >= height and block_w >= width):
            return xys, depths, radii, conics, num_tiles_hit, block_h, block_w
        block_h = min(height, block_h * 2)
        block_w = min(width, block_w * 2)


class ProjectToNormal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, positions, tangents):
        ctx.save_for_backward(tangents)
        return positions

    @staticmethod
    def backward(ctx, grad_output):
        tangents, = ctx.saved_tensors
        # Normalized tangent vector: t
        t_norm = torch.linalg.norm(tangents, dim=-1, keepdim=True).clamp_min(1e-6)
        t = tangents / t_norm
        
        # Remove the tangent component from the gradient: grad_output - dot(grad, t) * t
        tangent_component = (grad_output * t).sum(dim=-1, keepdim=True) * t
        projected_grad = grad_output - tangent_component

        return projected_grad, None


def _normalize_means_to_ndc(means_px: torch.Tensor, width: int, height: int) -> torch.Tensor:
    scale = means_px.new_tensor([2.0 / float(max(width, 1)), 2.0 / float(max(height, 1))])
    bias = means_px.new_tensor([-1.0, -1.0])
    return means_px * scale + bias


def render_compiled_scene(
    scene: CompiledOpenStrokeScene,
    *,
    width: int,
    height: int,
    config: BezierGsplatConfig,
    background_image: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    device = get_device()
    if device.type != "cuda":
        raise OpenStrokeUnsupported("bezier_gsplat requires a CUDA device")
    if background_image is not None:
        raise OpenStrokeUnsupported("background compositing is not supported")
    if int(width) != scene.canvas_width or int(height) != scene.canvas_height:
        raise OpenStrokeUnsupported("compiled scene size does not match render request")

    dtype = torch.float32
    flat_points = scene.point_bank.to(device=device, dtype=dtype)
    widths = scene.stroke_width_bank.to(device=device, dtype=dtype)
    colors = scene.stroke_rgba_bank.to(device=device, dtype=dtype)

    source_idx = scene.control_source_indices.view(-1)
    source_points = flat_points.index_select(0, source_idx).view(-1, scene.max_segments, 4, 2)
    cubic_controls = torch.einsum(
        "nsab,nsbc->nsac",
        scene.control_source_weights.to(device=device, dtype=dtype),
        source_points,
    )

    basis, deriv_basis = _basis_on(device, dtype, int(config.samples_per_segment))
    positions = torch.einsum("sf,nkfc->nksc", basis, cubic_controls)
    tangents = torch.einsum("sf,nkfc->nksc", deriv_basis, cubic_controls)

    import os
    if os.environ.get("DIFFVG_PROJECT_GRADIENTS", "1") == "1":
        positions = ProjectToNormal.apply(positions, tangents)

    sample_mask = scene.segment_mask[:, :, None].expand(-1, -1, basis.shape[0]).reshape(scene.stroke_count, -1)
    flat_positions = positions.reshape(scene.stroke_count, -1, 2)
    flat_tangents = tangents.reshape(scene.stroke_count, -1, 2)

    geom_positions = flat_positions.detach() if config.detach_geometry else flat_positions
    diffs = geom_positions[:, 1:, :] - geom_positions[:, :-1, :]
    dist = torch.linalg.norm(diffs, dim=-1).clamp_min(config.min_scale)
    valid_pairs = sample_mask[:, 1:] & sample_mask[:, :-1]
    dist = torch.where(valid_pairs, dist, torch.zeros_like(dist))

    sigma_x = torch.zeros_like(sample_mask, dtype=dtype)
    sigma_x[:, :-1] += dist
    sigma_x[:, 1:] += dist
    sigma_x = 0.5 * sigma_x.clamp_min(config.min_scale)

    tangent_input = flat_tangents.detach() if config.detach_geometry else flat_tangents
    tangent_norm = torch.linalg.norm(tangent_input, dim=-1, keepdim=True).clamp_min(1e-6)
    normalized_tangent = tangent_input / tangent_norm
    theta = torch.atan2(normalized_tangent[..., 1], normalized_tangent[..., 0]).unsqueeze(-1)
    if config.detach_geometry:
        theta = theta.detach()

    chunk_widths = widths.index_select(0, scene.style_index)
    chunk_colors = colors.index_select(0, scene.style_index)
    fwhm_coeff = 2.0 * math.sqrt(2.0 * math.log(2.0))
    sigma_y = (chunk_widths[:, None] / fwhm_coeff).expand_as(sigma_x).clamp_min(config.min_scale)
    rgba = chunk_colors[:, None, :].expand(scene.stroke_count, sample_mask.shape[1], 4)

    active = sample_mask.reshape(-1)
    means = flat_positions.reshape(-1, 2)[active]
    sigma_x_active = sigma_x.reshape(-1)[active]
    sigma_y_active = sigma_y.reshape(-1)[active]
    theta_active = theta.reshape(-1, 1)[active]
    rgba_active = rgba.reshape(-1, 4)[active]
    colors_rgb = rgba_active[:, :3].contiguous()
    opacity = rgba_active[:, 3:4].clamp(0.0, 1.0).contiguous()

    if means.numel() == 0:
        raise OpenStrokeUnsupported("compiled scene produced zero active Gaussian samples")

    depth_per_chunk = 1.0 - (scene.chunk_order.to(device=device, dtype=dtype) / float(max(scene.stroke_count, 1)))
    depth = depth_per_chunk[:, None].expand(scene.stroke_count, sample_mask.shape[1]).reshape(-1, 1)[active]
    if config.detach_geometry:
        depth = depth.detach()

    ops = load_gsplat_ops()
    scales = torch.stack([sigma_x_active, sigma_y_active], dim=-1).contiguous()
    xys, _depths_proj, radii, conics, num_tiles_hit, block_h, block_w = _project_with_safe_tile_budget(
        height,
        width,
        int(config.block_h),
        int(config.block_w),
        _normalize_means_to_ndc(means, width, height),
        scales,
        theta_active.contiguous(),
        ops,
    )
    rgb, alpha = ops.rasterize_gaussians(
        xys,
        depth,
        radii,
        conics,
        num_tiles_hit,
        colors_rgb,
        opacity,
        height,
        width,
        block_h,
        block_w,
        background=torch.zeros(3, device=device, dtype=dtype),
        return_alpha=True,
    )
    rgb = rgb.reshape(-1, height, width, 3)[0]
    alpha = alpha.reshape(-1, height, width, 1)[0]
    return torch.cat([rgb, alpha], dim=-1).clamp(0.0, 1.0)
