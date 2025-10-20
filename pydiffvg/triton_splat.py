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
    # backward
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


# ----------------------- Backward (per-tile, color-only v0) -----------------------

def env_wants_triton_backward() -> bool:
    v = (os.environ.get("DIFFVG_SPLAT_BWD", "").strip().lower())
    return v in ("triton", "trt", "kernel")


@triton.jit
def _backward_tiled_color_kernel(
    dcol_r_ptr, dcol_g_ptr, dcol_b_ptr, dopa_ptr,
    mu_x_ptr, mu_y_ptr,
    cos_ptr, sin_ptr,
    invsx_ptr, invsy_ptr,
    opacity_ptr,
    grad_r_ptr, grad_g_ptr, grad_b_ptr, grad_a_ptr,
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
    oy = tl.arange(0, BLOCK_H)
    ox = tl.arange(0, BLOCK_W)
    iy = y0 + oy[:, None]
    ix = x0 + ox[None, :]
    mask = (iy < H) & (ix < W)
    gy = iy.to(tl.float32) + 0.5
    gx = ix.to(tl.float32) + 0.5

    tile_id = ty * tiles_x + tx
    s = tl.load(tile_ptr_ptr + tile_id)
    e = tl.load(tile_ptr_ptr + tile_id + 1)

    # Loop pixels; accumulate color grads via atomics
    # Each pixel iterates splats in front-to-back order within the tile.
    # T is transmittance prefix.
    rlin = (iy * W + ix)
    gr = tl.load(grad_r_ptr + rlin, mask=mask, other=0.0)
    gg = tl.load(grad_g_ptr + rlin, mask=mask, other=0.0)
    gb = tl.load(grad_b_ptr + rlin, mask=mask, other=0.0)
    ga = tl.load(grad_a_ptr + rlin, mask=mask, other=0.0)

    T = tl.where(mask, 1.0, 0.0)

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

        dx = gx - mu_x
        dy = gy - mu_y
        lx = cth * dx + sth * dy
        ly = -sth * dx + cth * dy
        txx = lx * isx
        tyy = ly * isy
        expv = tl.exp(-0.5 * (txx * txx + tyy * tyy))
        ai = tl.maximum(tl.minimum(o * expv, 1.0), 0.0)

        wprefix = T * ai
        val_r = tl.sum(gr * wprefix, where=mask)
        val_g = tl.sum(gg * wprefix, where=mask)
        val_b = tl.sum(gb * wprefix, where=mask)
        tl.atomic_add(dcol_r_ptr + gi, val_r)
        tl.atomic_add(dcol_g_ptr + gi, val_g)
        tl.atomic_add(dcol_b_ptr + gi, val_b)

        eps = 1e-6
        g_over_o = tl.where(o > eps, ai / o, 0.0)
        val_a = tl.sum(ga * T * g_over_o, where=mask)
        tl.atomic_add(dopa_ptr + gi, val_a)

        T = T * (1.0 - ai)

        i += 1


def backward_tiled_color_triton(
    mu: torch.Tensor,
    theta: torch.Tensor,
    sigma_x: torch.Tensor,
    sigma_y: torch.Tensor,
    color_rgb: torch.Tensor,
    opacity: torch.Tensor,
    tile_ptr: torch.Tensor,
    tile_idx: torch.Tensor,
    width: int,
    height: int,
    tile_size: int,
    grad_img: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute dL/dcolor and dL/dopacity in Triton per tile (experimental)."""
    assert is_available()
    D = mu.device
    dtype = torch.float32
    mu = mu.contiguous().to(dtype)
    theta = theta.contiguous().to(dtype)
    sigma_x = sigma_x.contiguous().to(dtype)
    sigma_y = sigma_y.contiguous().to(dtype)
    opacity = opacity.contiguous().to(dtype)
    # Grad image channels (assume HxWxC on device)
    g = grad_img.to(device=D, dtype=dtype).contiguous()
    if g.shape[-1] == 4:
        gr, gg, gb, ga = g[..., 0], g[..., 1], g[..., 2], g[..., 3]
    else:
        gr, gg, gb = g[..., 0], g[..., 1], g[..., 2]
        ga = torch.zeros_like(gr)

    # Outputs
    N = mu.shape[0]
    dcol = torch.zeros((N, 3), device=D, dtype=dtype)
    dopa = torch.zeros((N,), device=D, dtype=dtype)

    cos_th = torch.cos(theta)
    sin_th = torch.sin(theta)
    inv_sx = 1.0 / sigma_x
    inv_sy = 1.0 / sigma_y

    tiles_x = (width + tile_size - 1) // tile_size
    grid = (int((height + tile_size - 1) // tile_size), int(tiles_x))
    BLOCK = int(tile_size)
    _backward_tiled_color_kernel[grid](
        dcol[:, 0], dcol[:, 1], dcol[:, 2], dopa,
        mu[:, 0], mu[:, 1],
        cos_th, sin_th,
        inv_sx, inv_sy,
        opacity,
        gr, gg, gb, ga,
        tile_ptr, tile_idx,
        width, height, tiles_x, tile_size,
        BLOCK_H=BLOCK, BLOCK_W=BLOCK,
    )
    return dcol, dopa


@triton.jit
def _backward_tiled_full_kernel(
    dcol_r_ptr, dcol_g_ptr, dcol_b_ptr, dopa_ptr,
    dmu_x_ptr, dmu_y_ptr, dtheta_ptr, disx_ptr, disy_ptr,
    mu_x_ptr, mu_y_ptr,
    cos_ptr, sin_ptr,
    invsx_ptr, invsy_ptr,
    opacity_ptr,
    col_r_ptr, col_g_ptr, col_b_ptr,
    grad_r_ptr, grad_g_ptr, grad_b_ptr, grad_a_ptr,
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
    oy = tl.arange(0, BLOCK_H)
    ox = tl.arange(0, BLOCK_W)
    iy = y0 + oy[:, None]
    ix = x0 + ox[None, :]
    mask = (iy < H) & (ix < W)
    gy = iy.to(tl.float32) + 0.5
    gx = ix.to(tl.float32) + 0.5

    tile_id = ty * tiles_x + tx
    s = tl.load(tile_ptr_ptr + tile_id)
    e = tl.load(tile_ptr_ptr + tile_id + 1)

    rlin = (iy * W + ix)
    gr = tl.load(grad_r_ptr + rlin, mask=mask, other=0.0)
    gg = tl.load(grad_g_ptr + rlin, mask=mask, other=0.0)
    gb = tl.load(grad_b_ptr + rlin, mask=mask, other=0.0)
    ga = tl.load(grad_a_ptr + rlin, mask=mask, other=0.0)

    # Pass 1: total transmittance per pixel
    Ttot = tl.where(mask, 1.0, 0.0)
    j = s
    while j < e:
        gi = tl.load(tile_idx_ptr + j)
        mu_x = tl.load(mu_x_ptr + gi)
        mu_y = tl.load(mu_y_ptr + gi)
        cth = tl.load(cos_ptr + gi)
        sth = tl.load(sin_ptr + gi)
        isx = tl.load(invsx_ptr + gi)
        isy = tl.load(invsy_ptr + gi)
        o   = tl.load(opacity_ptr + gi)
        dx = gx - mu_x
        dy = gy - mu_y
        lx = cth * dx + sth * dy
        ly = -sth * dx + cth * dy
        txx = lx * isx
        tyy = ly * isy
        expv = tl.exp(-0.5 * (txx * txx + tyy * tyy))
        ai = tl.maximum(tl.minimum(o * expv, 1.0), 0.0)
        Ttot = Ttot * (1.0 - ai)
        j += 1

    # Pass 2: reverse loop for grads
    U = tl.where(mask, 1.0, 0.0)
    Rr = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    Rg = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    Rb = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    j = e - 1
    while j >= s:
        gi = tl.load(tile_idx_ptr + j)
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

        eps = 1e-6
        denom = (1.0 - ai) * U
        denom = tl.where(denom > eps, denom, eps)
        T_i = Ttot / denom

        inv1ma = tl.where((1.0 - ai) > eps, 1.0 / (1.0 - ai), 1.0 / eps)
        Sr = T_i * Rr
        Sg = T_i * Rg
        Sb = T_i * Rb
        dcol_term = gr * (T_i * cr - Sr * inv1ma) + gg * (T_i * cg - Sg * inv1ma) + gb * (T_i * cb - Sb * inv1ma)
        dalpha_term = ga * (T_i * U)
        dLda = dcol_term + dalpha_term

        # color grads
        wprefix = T_i * ai
        val_r = tl.sum(gr * wprefix, where=mask)
        val_g = tl.sum(gg * wprefix, where=mask)
        val_b = tl.sum(gb * wprefix, where=mask)
        tl.atomic_add(dcol_r_ptr + gi, val_r)
        tl.atomic_add(dcol_g_ptr + gi, val_g)
        tl.atomic_add(dcol_b_ptr + gi, val_b)

        # opacity grad
        dopa = tl.sum(dLda * expv, where=mask)
        tl.atomic_add(dopa_ptr + gi, dopa)

        # geometry
        dG = dLda * o
        dlx = dG * (-(isx * txx))
        dly = dG * (-(isy * tyy))
        disx = tl.sum(dG * (-(lx * txx)), where=mask)
        disy = tl.sum(dG * (-(ly * tyy)), where=mask)
        tl.atomic_add(disx_ptr + gi, disx)
        tl.atomic_add(disy_ptr + gi, disy)
        dmx = tl.sum(dlx * (-cth) + dly * (sth), where=mask)
        dmy = tl.sum(dlx * (-sth) + dly * (-cth), where=mask)
        dth = tl.sum(dlx * (-sth * dx + cth * dy) + dly * (-cth * dx - sth * dy), where=mask)
        tl.atomic_add(dmu_x_ptr + gi, dmx)
        tl.atomic_add(dmu_y_ptr + gi, dmy)
        tl.atomic_add(dtheta_ptr + gi, dth)

        # update suffix
        Rr = cr * ai + (1.0 - ai) * Rr
        Rg = cg * ai + (1.0 - ai) * Rg
        Rb = cb * ai + (1.0 - ai) * Rb
        U = (1.0 - ai) * U
        j -= 1


def backward_tiled_full_triton(
    mu: torch.Tensor,
    theta: torch.Tensor,
    sigma_x: torch.Tensor,
    sigma_y: torch.Tensor,
    color_rgb: torch.Tensor,
    opacity: torch.Tensor,
    tile_ptr: torch.Tensor,
    tile_idx: torch.Tensor,
    width: int,
    height: int,
    tile_size: int,
    grad_img: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert is_available()
    D = mu.device
    dtype = torch.float32
    mu = mu.contiguous().to(dtype)
    theta = theta.contiguous().to(dtype)
    sigma_x = sigma_x.contiguous().to(dtype)
    sigma_y = sigma_y.contiguous().to(dtype)
    opacity = opacity.contiguous().to(dtype)
    g = grad_img.to(device=D, dtype=dtype).contiguous()
    if g.shape[-1] == 4:
        gr, gg, gb, ga = g[..., 0], g[..., 1], g[..., 2], g[..., 3]
    else:
        gr, gg, gb = g[..., 0], g[..., 1], g[..., 2]
        ga = torch.zeros_like(gr)

    N = mu.shape[0]
    dcol = torch.zeros((N, 3), device=D, dtype=dtype)
    dopa = torch.zeros((N,), device=D, dtype=dtype)
    dmu_x = torch.zeros((N,), device=D, dtype=dtype)
    dmu_y = torch.zeros((N,), device=D, dtype=dtype)
    dtheta = torch.zeros((N,), device=D, dtype=dtype)
    disx = torch.zeros((N,), device=D, dtype=dtype)
    disy = torch.zeros((N,), device=D, dtype=dtype)

    cos_th = torch.cos(theta)
    sin_th = torch.sin(theta)
    inv_sx = 1.0 / sigma_x
    inv_sy = 1.0 / sigma_y

    tiles_x = (width + tile_size - 1) // tile_size
    grid = (int((height + tile_size - 1) // tile_size), int(tiles_x))
    BLOCK = int(tile_size)
    _backward_tiled_full_kernel[grid](
        dcol[:, 0], dcol[:, 1], dcol[:, 2], dopa,
        dmu_x, dmu_y, dtheta, disx, disy,
        mu[:, 0], mu[:, 1],
        cos_th, sin_th,
        inv_sx, inv_sy,
        opacity,
        color_rgb[:, 0], color_rgb[:, 1], color_rgb[:, 2],
        gr, gg, gb, ga,
        tile_ptr, tile_idx,
        width, height, tiles_x, tile_size,
        BLOCK_H=BLOCK, BLOCK_W=BLOCK,
    )
    return dcol, dopa, dmu_x, dmu_y, dtheta, disx, disy
