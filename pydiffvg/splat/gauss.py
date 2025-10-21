from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch

from ..backend import SplatConfig
from .geometry import (
    _evaluate_segment,
    _segment_arclength,
    _segment_samples,
)
from .types import GaussianBatch, PathSpec, FillSpec, SegmentData, _SplatUnsupported


def _sample_path_geometry(
    segments: List[SegmentData],
    config: SplatConfig,
    device: torch.device,
    dtype: torch.dtype,
    generator: Optional[torch.Generator],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not segments:
        return (
            torch.empty(0, 2, device=device, dtype=dtype),
            torch.empty(0, 2, device=device, dtype=dtype),
            torch.empty(0, device=device, dtype=dtype),
            torch.empty(0, device=device, dtype=dtype),
        )

    lengths = torch.stack([
        _segment_arclength(seg, device=device, dtype=dtype, samples=16) for seg in segments
    ])
    target_delta = 1.0
    mid_counts: List[int] = []
    for L in lengths:
        k = int(max(1, math.ceil(float(L.item()) / max(target_delta, 1e-6))))
        mid_counts.append(max(k - 1, 0))

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
        mid_count = mid_counts[idx] if idx < len(mid_counts) else 0
        if mid_count > 0:
            t_mid = _segment_samples(mid_count, device, dtype, None)
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
    delta_s = sigma_x.clone()

    return mu, tangents, torch.clamp(sigma_x, min=1e-6), torch.clamp(delta_s, min=1e-8)


def _path_to_gaussians(
    spec: PathSpec,
    config: SplatConfig,
    device: torch.device,
    dtype: torch.dtype,
    generator: Optional[torch.Generator],
) -> GaussianBatch:
    mu, tangents, sigma_x, delta_s = _sample_path_geometry(
        spec.segments, config, device, dtype, generator
    )
    num_samples = mu.shape[0]
    if num_samples == 0:
        raise _SplatUnsupported("path without samples cannot be rasterized")

    rho = max(float(config.rho), 1e-6)
    sigma_x = torch.clamp(sigma_x / rho, min=1e-3)

    width = torch.clamp(spec.stroke_width.reshape(-1)[0], min=1e-3)
    fwhm_coeff = 2.0 * math.sqrt(2.0 * math.log(2.0))
    sigma_y = torch.ones(num_samples, device=device, dtype=dtype) * (width / (fwhm_coeff * rho))
    sigma_y = torch.clamp(sigma_y, min=1e-3)

    if mu.shape[0] >= 3:
        diff_fwd = mu[2:] - mu[1:-1]
        diff_bwd = mu[1:-1] - mu[:-2]
        cen = diff_fwd + diff_bwd
        theta = torch.empty(mu.shape[0], device=device, dtype=dtype)
        cen_norm = torch.linalg.norm(cen, dim=1)
        mask_good = cen_norm > 1e-8
        th_cen = torch.atan2(cen[:, 1], cen[:, 0])
        tnorm_all = torch.linalg.norm(tangents, dim=1, keepdim=True).clamp_min(1e-6)
        ntan_all = tangents / tnorm_all
        th_tan_mid = torch.atan2(ntan_all[1:-1, 1], ntan_all[1:-1, 0])
        theta[1:-1] = torch.where(mask_good, th_cen, th_tan_mid)
        theta[0] = torch.atan2(ntan_all[0, 1], ntan_all[0, 0])
        theta[-1] = torch.atan2(ntan_all[-1, 1], ntan_all[-1, 0])
    else:
        tnorm = torch.linalg.norm(tangents, dim=1, keepdim=True).clamp_min(1e-6)
        ntan = tangents / tnorm
        theta = torch.atan2(ntan[:, 1], ntan[:, 0])

    color_rgb = spec.color_rgb.to(device=device, dtype=dtype).unsqueeze(0).expand(num_samples, -1)
    base_o = spec.opacity.to(device=device, dtype=dtype).clamp(0.0, 1.0)
    opacity = base_o.expand(num_samples)

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
    mu_boundary, tangents, sigma_x, _delta_s = _sample_path_geometry(
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


__all__ = [
    "_sample_path_geometry",
    "_path_to_gaussians",
    "_fill_to_gaussians",
]
