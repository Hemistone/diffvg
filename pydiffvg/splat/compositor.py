from __future__ import annotations

import os
from typing import List, Tuple

import torch
from torch.utils.checkpoint import checkpoint as _ckpt

_GRID_CACHE: dict[Tuple[int, int, torch.device, torch.dtype], Tuple[torch.Tensor, torch.Tensor]] = {}


def _get_full_grid(height: int, width: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    key = (height, width, device, dtype)
    cached = _GRID_CACHE.get(key)
    if cached is not None:
        return cached
    yy = torch.arange(height, device=device, dtype=dtype) + 0.5
    xx = torch.arange(width, device=device, dtype=dtype) + 0.5
    try:
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
    except TypeError:
        grid_y, grid_x = torch.meshgrid(yy, xx)
    _GRID_CACHE[key] = (grid_y, grid_x)
    return grid_y, grid_x


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
    grid_y, grid_x = _get_full_grid(height, width, device, dtype)

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


def _composite_gaussians_full_ckpt(
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
    chunk: int = 512,
) -> torch.Tensor:
    chunk = max(int(chunk), 1)
    grid_y, grid_x = _get_full_grid(height, width, device, dtype)

    image_rgb = torch.zeros(height, width, 3, device=device, dtype=dtype)
    image_alpha = torch.zeros(height, width, device=device, dtype=dtype)

    def chunk_fn(
        img_rgb_in: torch.Tensor,
        img_a_in: torch.Tensor,
        mu_c: torch.Tensor,
        th_c: torch.Tensor,
        sx_c: torch.Tensor,
        sy_c: torch.Tensor,
        col_c: torch.Tensor,
        op_c: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        img_rgb = img_rgb_in
        img_a = img_a_in
        cos_th = torch.cos(th_c)
        sin_th = torch.sin(th_c)
        inv_sx = 1.0 / sx_c
        inv_sy = 1.0 / sy_c
        for i in range(mu_c.shape[0]):
            dx = grid_x - mu_c[i, 0]
            dy = grid_y - mu_c[i, 1]
            lx = cos_th[i] * dx + sin_th[i] * dy
            ly = -sin_th[i] * dx + cos_th[i] * dy
            exponent = -0.5 * ((lx * inv_sx[i]) ** 2 + (ly * inv_sy[i]) ** 2)
            g = torch.exp(exponent)
            a_i = torch.clamp(op_c[i], 0.0, 1.0) * g
            contrib = (1.0 - img_a) * a_i
            img_rgb = img_rgb + contrib.unsqueeze(-1) * col_c[i]
            img_a = img_a + contrib
        return img_rgb, torch.clamp(img_a, 0.0, 1.0)

    n = mu.shape[0]
    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        try:
            image_rgb, image_alpha = _ckpt(
                chunk_fn,
                image_rgb,
                image_alpha,
                mu[start:end],
                theta[start:end],
                sigma_x[start:end],
                sigma_y[start:end],
                color_rgb[start:end],
                opacity[start:end],
                use_reentrant=False,
            )
        except TypeError:
            image_rgb, image_alpha = _ckpt(
                chunk_fn,
                image_rgb,
                image_alpha,
                mu[start:end],
                theta[start:end],
                sigma_x[start:end],
                sigma_y[start:end],
                color_rgb[start:end],
                opacity[start:end],
            )

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

    min_tile_x = torch.floor((mu[:, 0] - extent_x) / tile_size).to(torch.int64).clamp(0, tiles_x - 1)
    max_tile_x = torch.floor((mu[:, 0] + extent_x) / tile_size).to(torch.int64).clamp(0, tiles_x - 1)
    min_tile_y = torch.floor((mu[:, 1] - extent_y) / tile_size).to(torch.int64).clamp(0, tiles_y - 1)
    max_tile_y = torch.floor((mu[:, 1] + extent_y) / tile_size).to(torch.int64).clamp(0, tiles_y - 1)

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

    gchunk_env = os.environ.get("DIFFVG_SPLAT_TILE_GCHUNK", "128").strip() or "128"
    try:
        gchunk = max(1, int(gchunk_env))
    except Exception:
        gchunk = 128

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

        tile_rgb = torch.zeros((y1 - y0, x1 - x0, 3), device=device, dtype=dtype)
        tile_alpha = torch.zeros((y1 - y0, x1 - x0), device=device, dtype=dtype)

        nbin = tile_indices.numel()
        for s in range(0, nbin, gchunk):
            e = min(nbin, s + gchunk)
            sel = tile_indices[s:e]
            mu_c = mu[sel]
            ct = cos_theta[sel]
            st = sin_theta[sel]
            isx = inv_sigma_x[sel]
            isy = inv_sigma_y[sel]
            col = color_rgb[sel]
            opa = opacity_clamped[sel]
            m = sel.numel()
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
            w = trans_prev * a * T
            tile_rgb = tile_rgb + (w.unsqueeze(-1) * col.view(m, 1, 1, 3)).sum(dim=0)
            prod_all = P[-1] if m > 0 else torch.ones_like(tile_alpha)
            tile_alpha = tile_alpha + (1.0 - tile_alpha) * (1.0 - prod_all)

        image_rgb[y0:y1, x0:x1] = tile_rgb
        image_alpha[y0:y1, x0:x1] = torch.clamp(tile_alpha, 0.0, 1.0)

    return torch.cat([image_rgb, image_alpha.unsqueeze(-1)], dim=-1)


def _composite_gaussians_tiled_diff(
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

    min_tile_x = torch.floor((mu[:, 0] - extent_x) / tile_size).to(torch.int64).clamp(0, tiles_x - 1)
    max_tile_x = torch.floor((mu[:, 0] + extent_x) / tile_size).to(torch.int64).clamp(0, tiles_x - 1)
    min_tile_y = torch.floor((mu[:, 1] - extent_y) / tile_size).to(torch.int64).clamp(0, tiles_y - 1)
    max_tile_y = torch.floor((mu[:, 1] + extent_y) / tile_size).to(torch.int64).clamp(0, tiles_y - 1)

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

    out_rgb = torch.zeros(height, width, 3, device=device, dtype=dtype)
    out_alpha = torch.zeros(height, width, device=device, dtype=dtype)

    gchunk_env = os.environ.get("DIFFVG_SPLAT_TILE_GCHUNK", "128").strip() or "128"
    try:
        gchunk = max(1, int(gchunk_env))
    except Exception:
        gchunk = 128

    grid_y_full, grid_x_full = _get_full_grid(height, width, device, dtype)

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

        tile_rgb = torch.zeros((y1 - y0, x1 - x0, 3), device=device, dtype=dtype)
        tile_alpha = torch.zeros((y1 - y0, x1 - x0), device=device, dtype=dtype)

        tile_indices = torch.tensor(idx_list, device=device, dtype=torch.long)
        nbin = tile_indices.numel()
        for s in range(0, nbin, gchunk):
            e = min(nbin, s + gchunk)
            sel = tile_indices[s:e]
            mu_c = mu[sel]
            ct = cos_theta[sel]
            st = sin_theta[sel]
            isx = inv_sigma_x[sel]
            isy = inv_sigma_y[sel]
            col = color_rgb[sel]
            opa = opacity_clamped[sel]
            m = sel.numel()

            gy = grid_y_full[y0:y1, x0:x1]
            gx = grid_x_full[y0:y1, x0:x1]

            dx = gx.unsqueeze(0) - mu_c[:, 0].view(m, 1, 1)
            dy = gy.unsqueeze(0) - mu_c[:, 1].view(m, 1, 1)
            lx = ct.view(m, 1, 1) * dx + st.view(m, 1, 1) * dy
            ly = -st.view(m, 1, 1) * dx + ct.view(m, 1, 1) * dy
            exponent = -0.5 * ((lx * isx.view(m, 1, 1)) ** 2 + (ly * isy.view(m, 1, 1)) ** 2)
            a = torch.exp(exponent) * opa.view(m, 1, 1)
            a = torch.clamp(a, 0.0, 1.0)
            one_minus_a = 1.0 - a
            P = torch.cumprod(one_minus_a, dim=0)
            T = torch.cat([torch.ones(1, *P.shape[1:], device=device, dtype=dtype), P[:-1]], dim=0)
            trans_prev = (1.0 - tile_alpha).unsqueeze(0)
            w = trans_prev * a * T
            tile_rgb = tile_rgb + (w.unsqueeze(-1) * col.view(m, 1, 1, 3)).sum(dim=0)
            prod_all = P[-1] if m > 0 else torch.ones_like(tile_alpha)
            tile_alpha = tile_alpha + (1.0 - tile_alpha) * (1.0 - prod_all)

        out_rgb[y0:y1, x0:x1] = tile_rgb
        out_alpha[y0:y1, x0:x1] = torch.clamp(tile_alpha, 0.0, 1.0)

    return torch.cat([out_rgb, out_alpha.unsqueeze(-1)], dim=-1)


__all__ = [
    "_get_full_grid",
    "_composite_gaussians_full",
    "_composite_gaussians_full_ckpt",
    "_composite_gaussians_tiled",
    "_composite_gaussians_tiled_diff",
]
