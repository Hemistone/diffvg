from __future__ import annotations

import os
from typing import Optional, Tuple

import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


def is_available() -> bool:
    return _HAS_TRITON and torch.cuda.is_available()


@triton.jit
def _composite_full_chunk_kernel(
    out_r_ptr, out_g_ptr, out_b_ptr, out_alpha_ptr,
    mu_x_ptr, mu_y_ptr,
    cos_ptr, sin_ptr,
    invsx_ptr, invsy_ptr,
    col_r_ptr, col_g_ptr, col_b_ptr, opacity_ptr,
    H: tl.constexpr, W: tl.constexpr,
    K: tl.constexpr,  # compile-time chunk size
    Npix,
    # launch params
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < Npix

    # 1D linear index to 2D coords
    y = (offs // W).to(tl.float32)
    x = (offs % W).to(tl.float32)
    gy = y + 0.5
    gx = x + 0.5

    # Load accumulators
    r = tl.load(out_r_ptr + offs, mask=mask, other=0.0)
    g = tl.load(out_g_ptr + offs, mask=mask, other=0.0)
    b = tl.load(out_b_ptr + offs, mask=mask, other=0.0)
    a = tl.load(out_alpha_ptr + offs, mask=mask, other=0.0)

    # Loop over fixed-size chunk of Gaussians
    for i in tl.static_range(0, K):
        mu_x = tl.load(mu_x_ptr + i, mask=True, other=0.0)
        mu_y = tl.load(mu_y_ptr + i, mask=True, other=0.0)
        cth = tl.load(cos_ptr + i, mask=True, other=1.0)
        sth = tl.load(sin_ptr + i, mask=True, other=0.0)
        isx = tl.load(invsx_ptr + i, mask=True, other=0.0)
        isy = tl.load(invsy_ptr + i, mask=True, other=0.0)
        o   = tl.load(opacity_ptr + i, mask=True, other=0.0)
        col_r = tl.load(col_r_ptr + i, mask=True, other=0.0)
        col_g = tl.load(col_g_ptr + i, mask=True, other=0.0)
        col_b = tl.load(col_b_ptr + i, mask=True, other=0.0)

        dx = gx - mu_x
        dy = gy - mu_y
        lx = cth * dx + sth * dy
        ly = -sth * dx + cth * dy
        txx = (lx * isx)
        tyy = (ly * isy)
        expv = tl.exp(-0.5 * (txx * txx + tyy * tyy))
        ai = tl.maximum(tl.minimum(o * expv, 1.0), 0.0)
        contrib = (1.0 - a) * ai
        r += contrib * col_r
        g += contrib * col_g
        b += contrib * col_b
        a += contrib

    # Write back (aligned contiguous linear addresses)
    tl.store(out_r_ptr + offs, r, mask=mask)
    tl.store(out_g_ptr + offs, g, mask=mask)
    tl.store(out_b_ptr + offs, b, mask=mask)
    tl.store(out_alpha_ptr + offs, a, mask=mask)


def composite_gaussians_full_triton(
    mu: torch.Tensor,
    theta: torch.Tensor,
    sigma_x: torch.Tensor,
    sigma_y: torch.Tensor,
    color_rgb: torch.Tensor,
    opacity: torch.Tensor,
    width: int,
    height: int,
    *,
    block_h: int = 32,
    block_w: int = 32,
    gchunk: int = 256,
) -> torch.Tensor:
    """Full-frame compositor using a Triton kernel.

    Splits the M gaussians into fixed-size chunks (gchunk) and calls the kernel
    sequentially to update the same output buffers (front-to-back order).
    """
    assert is_available(), "Triton not available or CUDA disabled"
    D = mu.device
    # Force float32 for the kernel
    mu = mu.contiguous().to(dtype=torch.float32, device=D)
    theta = theta.contiguous().to(dtype=torch.float32, device=D)
    sigma_x = sigma_x.contiguous().to(dtype=torch.float32, device=D)
    sigma_y = sigma_y.contiguous().to(dtype=torch.float32, device=D)
    color_rgb = color_rgb.contiguous().to(dtype=torch.float32, device=D)
    opacity = opacity.contiguous().to(dtype=torch.float32, device=D)
    dtype = torch.float32
    H, W = height, width
    # Precompute sin/cos and inverses
    cos_th = torch.cos(theta)
    sin_th = torch.sin(theta)
    inv_sx = 1.0 / sigma_x
    inv_sy = 1.0 / sigma_y
    # Prepare output buffers
    out_r = torch.zeros((H, W), device=D, dtype=dtype)
    out_g = torch.zeros((H, W), device=D, dtype=dtype)
    out_b = torch.zeros((H, W), device=D, dtype=dtype)
    out_a = torch.zeros((H, W), device=D, dtype=dtype)

    # 1D launch over pixels for alignment-safe stores
    Npix = H * W
    # Tunable launch parameters via env
    def _env_int(name: str, default: int) -> int:
        v = os.environ.get(name)
        if v is None or v.strip() == "":
            return default
        try:
            return int(v)
        except Exception:
            return default
    block = _env_int("DIFFVG_SPLAT_BLOCK", 4096)
    warps = _env_int("DIFFVG_SPLAT_WARPS", 4)
    stages = _env_int("DIFFVG_SPLAT_STAGES", 2)
    grid = (triton.cdiv(Npix, block),)

    M = mu.shape[0]
    # Pad arrays to multiples of gchunk for final call
    pad = (gchunk - (M % gchunk)) % gchunk
    if pad:
        mu_pad = torch.zeros((pad, 2), device=D, dtype=dtype)
        mu_use = torch.cat([mu, mu_pad], dim=0)
        cos_use = torch.cat([cos_th, torch.ones(pad, device=D, dtype=dtype)])
        sin_use = torch.cat([sin_th, torch.zeros(pad, device=D, dtype=dtype)])
        invsx_use = torch.cat([inv_sx, torch.zeros(pad, device=D, dtype=dtype)])
        invsy_use = torch.cat([inv_sy, torch.zeros(pad, device=D, dtype=dtype)])
        opacity_use = torch.cat([opacity, torch.zeros(pad, device=D, dtype=dtype)])
        color_use = torch.cat([color_rgb, torch.zeros((pad, 3), device=D, dtype=dtype)], dim=0)
    else:
        mu_use = mu
        cos_use = cos_th
        sin_use = sin_th
        invsx_use = inv_sx
        invsy_use = inv_sy
        opacity_use = opacity
        color_use = color_rgb

    # Flatten color for kernel (M*3)
    col_r = color_use[:, 0].contiguous()
    col_g = color_use[:, 1].contiguous()
    col_b = color_use[:, 2].contiguous()

    for start in range(0, mu_use.shape[0], gchunk):
        end = start + gchunk
        mu_s = mu_use[start:end]
        _composite_full_chunk_kernel[grid](
            out_r, out_g, out_b, out_a,
            mu_s[:, 0], mu_s[:, 1],
            cos_use[start:end], sin_use[start:end],
            invsx_use[start:end], invsy_use[start:end],
            col_r[start:end], col_g[start:end], col_b[start:end], opacity_use[start:end],
            H, W, gchunk, Npix,
            BLOCK=block,
            num_warps=warps, num_stages=stages,
        )

    out_a = torch.clamp(out_a, 0.0, 1.0)
    out_rgb = torch.stack([out_r, out_g, out_b], dim=-1)
    return torch.cat([out_rgb, out_a.unsqueeze(-1)], dim=-1)


def env_wants_triton() -> bool:
    return (os.environ.get("DIFFVG_SPLAT_IMPL", "").strip().lower() in ("triton", "trt", "kernel"))


__all__ = [
    "is_available",
    "env_wants_triton",
    "composite_gaussians_full_triton",
]


# ----------------------- Tiled Triton (CSR) -----------------------

@triton.jit
def _composite_tiled_kernel(
    out_r_ptr, out_g_ptr, out_b_ptr, out_a_ptr,
    mu_x_ptr, mu_y_ptr,
    cos_ptr, sin_ptr,
    invsx_ptr, invsy_ptr,
    col_r_ptr, col_g_ptr, col_b_ptr, opacity_ptr,
    tile_ptr_ptr, tile_idx_ptr,
    W: tl.constexpr,
    H: tl.constexpr,
    tiles_x: tl.constexpr,
    tile_size: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    ty = tl.program_id(0)
    tx = tl.program_id(1)
    x0 = tx * tile_size
    y0 = ty * tile_size
    # Local pixel coords within tile
    oy = tl.arange(0, BLOCK_H)
    ox = tl.arange(0, BLOCK_W)
    gy = (y0 + oy)[:, None].to(tl.float32) + 0.5
    gx = (x0 + ox)[None, :].to(tl.float32) + 0.5
    # Boundaries mask
    mask_y = (y0 + oy) < H
    mask_x = (x0 + ox) < W
    mask2 = mask_y[:, None] & mask_x[None, :]

    # Initialize accumulators
    r = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    g = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    b = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    a = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)

    tile_id = ty * tiles_x + tx
    s = tl.load(tile_ptr_ptr + tile_id)
    e = tl.load(tile_ptr_ptr + tile_id + 1)

    # Iterate over splats assigned to this tile
    i = s
    while i < e:
        gi = tl.load(tile_idx_ptr + i)
        mu_x = tl.load(mu_x_ptr + gi)
        mu_y = tl.load(mu_y_ptr + gi)
        cth = tl.load(cos_ptr + gi)
        sth = tl.load(sin_ptr + gi)
        isx = tl.load(invsx_ptr + gi)
        isy = tl.load(invsy_ptr + gi)
        o   = tl.load(opacity_ptr + gi)
        cr = tl.load(col_r_ptr + gi)
        cg = tl.load(col_g_ptr + gi)
        cb = tl.load(col_b_ptr + gi)

        dx = gx - mu_x
        dy = gy - mu_y
        lx = cth * dx + sth * dy
        ly = -sth * dx + cth * dy
        txx = lx * isx
        tyy = ly * isy
        expv = tl.exp(-0.5 * (txx * txx + tyy * tyy))
        ai = tl.maximum(tl.minimum(o * expv, 1.0), 0.0)
        contrib = (1.0 - a) * ai
        r += contrib * cr
        g += contrib * cg
        b += contrib * cb
        a += contrib
        i += 1

    # Store back to image
    iy = y0 + oy
    ix = x0 + ox
    lin = (iy[:, None] * W + ix[None, :])
    tl.store(out_r_ptr + lin, r, mask=mask2)
    tl.store(out_g_ptr + lin, g, mask=mask2)
    tl.store(out_b_ptr + lin, b, mask=mask2)
    tl.store(out_a_ptr + lin, tl.minimum(a, 1.0), mask=mask2)


def _build_tile_csr(
    mu: torch.Tensor,
    theta: torch.Tensor,
    sigma_x: torch.Tensor,
    sigma_y: torch.Tensor,
    width: int,
    height: int,
    tile: int,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """Build CSR bins (tile_ptr, tile_idx) on CPU for now (fast enough for N~1e4)."""
    device = mu.device
    dtype = mu.dtype
    tile = int(tile)
    tiles_x = (width + tile - 1) // tile
    tiles_y = (height + tile - 1) // tile
    # extents
    cos_th = torch.cos(theta).to(dtype)
    sin_th = torch.sin(theta).to(dtype)
    e = 3.0
    ext_x = e * (cos_th.abs() * sigma_x + sin_th.abs() * sigma_y)
    ext_y = e * (sin_th.abs() * sigma_x + cos_th.abs() * sigma_y)
    min_tx = torch.floor((mu[:, 0] - ext_x) / tile).to(torch.int64).clamp(0, tiles_x - 1)
    max_tx = torch.floor((mu[:, 0] + ext_x) / tile).to(torch.int64).clamp(0, tiles_x - 1)
    min_ty = torch.floor((mu[:, 1] - ext_y) / tile).to(torch.int64).clamp(0, tiles_y - 1)
    max_ty = torch.floor((mu[:, 1] + ext_y) / tile).to(torch.int64).clamp(0, tiles_y - 1)

    bins: list[list[int]] = [[] for _ in range(tiles_x * tiles_y)]
    N = mu.shape[0]
    for i in range(N):
        x0 = int(min_tx[i].item()); x1 = int(max_tx[i].item())
        y0 = int(min_ty[i].item()); y1 = int(max_ty[i].item())
        if x0 > x1 or y0 > y1:
            continue
        for ty in range(y0, y1 + 1):
            base = ty * tiles_x
            for tx in range(x0, x1 + 1):
                bins[base + tx].append(i)
    counts = [len(b) for b in bins]
    tile_ptr = torch.empty(len(bins) + 1, dtype=torch.int32)
    acc = 0
    for t, c in enumerate(counts):
        tile_ptr[t] = acc
        acc += c
    tile_ptr[-1] = acc
    tile_idx = torch.empty(acc, dtype=torch.int32)
    off = 0
    for b in bins:
        if b:
            tile_idx[off:off + len(b)] = torch.tensor(b, dtype=torch.int32)
            off += len(b)
    return tile_ptr.to(device), tile_idx.to(device), tiles_x, tiles_y


def composite_gaussians_tiled_triton(
    mu: torch.Tensor,
    theta: torch.Tensor,
    sigma_x: torch.Tensor,
    sigma_y: torch.Tensor,
    color_rgb: torch.Tensor,
    opacity: torch.Tensor,
    width: int,
    height: int,
    tile_size: int,
) -> torch.Tensor:
    assert is_available() and torch.cuda.is_available()
    D = mu.device
    # fp32, contiguous
    mu = mu.contiguous().to(dtype=torch.float32, device=D)
    theta = theta.contiguous().to(dtype=torch.float32, device=D)
    sigma_x = sigma_x.contiguous().to(dtype=torch.float32, device=D)
    sigma_y = sigma_y.contiguous().to(dtype=torch.float32, device=D)
    color_rgb = color_rgb.contiguous().to(dtype=torch.float32, device=D)
    opacity = opacity.contiguous().to(dtype=torch.float32, device=D)
    # Build CSR bins (CPU but cheap)
    tile_ptr, tile_idx, tiles_x, tiles_y = _build_tile_csr(mu, theta, sigma_x, sigma_y, width, height, tile_size)
    # Outputs
    out_r = torch.zeros((height, width), device=D, dtype=torch.float32)
    out_g = torch.zeros_like(out_r)
    out_b = torch.zeros_like(out_r)
    out_a = torch.zeros_like(out_r)
    # Params
    cos_th = torch.cos(theta)
    sin_th = torch.sin(theta)
    inv_sx = 1.0 / sigma_x
    inv_sy = 1.0 / sigma_y
    col_r = color_rgb[:, 0]
    col_g = color_rgb[:, 1]
    col_b = color_rgb[:, 2]
    grid = (tiles_y, tiles_x)
    BLOCK = int(tile_size)
    _composite_tiled_kernel[grid](
        out_r, out_g, out_b, out_a,
        mu[:, 0], mu[:, 1],
        cos_th, sin_th, inv_sx, inv_sy,
        col_r, col_g, col_b, opacity,
        tile_ptr, tile_idx,
        width, height, tiles_x, tile_size,
        BLOCK_H=BLOCK, BLOCK_W=BLOCK,
    )
    out_a = torch.clamp(out_a, 0.0, 1.0)
    out_rgb = torch.stack([out_r, out_g, out_b], dim=-1)
    return torch.cat([out_rgb, out_a.unsqueeze(-1)], dim=-1)
