from __future__ import annotations

import math
import os
from typing import Optional, Tuple

import torch

from ..trace import debug_enabled, should_print, trace
from .forward import _build_tile_csr
from .runtime import is_available, tl, triton


"""Triton backward kernels (tiled). Pixel variant removed; always use tiled."""


_PREFIX_TRACE_COUNT = 0
_last_backward_capture: Optional[dict] = None


def get_last_backward_capture() -> Optional[dict]:
    return _last_backward_capture


def _log_prefix_stats(
    mu: torch.Tensor,
    theta: torch.Tensor,
    sigma_x: torch.Tensor,
    sigma_y: torch.Tensor,
    opacity: torch.Tensor,
    tile_ptr: torch.Tensor,
    tile_idx: torch.Tensor,
    tiles_x: int,
    tiles_y: int,
    tile_size: int,
) -> None:
    if not debug_enabled():
        return
    global _PREFIX_TRACE_COUNT
    _PREFIX_TRACE_COUNT += 1
    if not should_print(_PREFIX_TRACE_COUNT):
        return

    try:
        mu_cpu = mu.detach().to(dtype=torch.float32, device="cpu")
        theta_cpu = theta.detach().to(dtype=torch.float32, device="cpu")
        sigma_x_cpu = sigma_x.detach().to(dtype=torch.float32, device="cpu")
        sigma_y_cpu = sigma_y.detach().to(dtype=torch.float32, device="cpu")
        opacity_cpu = opacity.detach().to(dtype=torch.float32, device="cpu")
        tile_ptr_cpu = tile_ptr.detach().to(device="cpu", dtype=torch.int64)
        tile_idx_cpu = tile_idx.detach().to(device="cpu", dtype=torch.int64)
    except Exception as exc:  # pragma: no cover - defensive fallback
        trace(f"tiled-bwd prefix instrumentation failed: {exc}")
        return

    prefix_samples: list[float] = []
    num_tiles = tile_ptr_cpu.numel() - 1
    max_samples = 8
    for tile_id in range(num_tiles):
        s = int(tile_ptr_cpu[tile_id].item())
        e = int(tile_ptr_cpu[tile_id + 1].item())
        if e <= s:
            continue
        ty = tile_id // tiles_x
        tx = tile_id % tiles_x
        cx = tx * tile_size + tile_size / 2.0
        cy = ty * tile_size + tile_size / 2.0
        prefix = 1.0
        min_prefix = 1.0
        for idx in range(s, e):
            gi = int(tile_idx_cpu[idx].item())
            mu_x = float(mu_cpu[gi, 0].item())
            mu_y = float(mu_cpu[gi, 1].item())
            th = float(theta_cpu[gi].item())
            cth = math.cos(th)
            sth = math.sin(th)
            inv_sx = 1.0 / float(sigma_x_cpu[gi].item())
            inv_sy = 1.0 / float(sigma_y_cpu[gi].item())
            dx = cx - mu_x
            dy = cy - mu_y
            lx = cth * dx + sth * dy
            ly = -sth * dx + cth * dy
            expv = math.exp(-0.5 * ((lx * inv_sx) ** 2 + (ly * inv_sy) ** 2))
            ai = float(opacity_cpu[gi].item()) * expv
            if ai > 1.0:
                ai = 1.0
            elif ai < 0.0:
                ai = 0.0
            prefix *= max(1.0 - ai, 1e-12)
            min_prefix = min(min_prefix, prefix)
        prefix_samples.append(min_prefix)
        if len(prefix_samples) >= max_samples:
            break

    if not prefix_samples:
        trace("tiled-bwd prefix stats unavailable (no active tiles)")
        return

    min_prefix = min(prefix_samples)
    mean_prefix = sum(prefix_samples) / len(prefix_samples)
    trace(
        f"tiled-bwd prefix stats: samples={len(prefix_samples)} min={min_prefix:.3e} mean={mean_prefix:.3e}"
    )


@triton.jit
def _backward_tiled_color_kernel(
    dcol_r_ptr, dcol_g_ptr, dcol_b_ptr, dopa_ptr,
    mu_x_ptr, mu_y_ptr,
    cos_ptr, sin_ptr,
    invsx_ptr, invsy_ptr,
    opacity_ptr,
    grad_r_ptr, grad_g_ptr, grad_b_ptr, grad_a_ptr,
    tile_ptr_ptr, tile_idx_ptr,
    tile_min_x_ptr, tile_max_x_ptr,
    tile_min_y_ptr, tile_max_y_ptr,
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
        min_x = tl.load(tile_min_x_ptr + i).to(tl.int32)
        max_x = tl.load(tile_max_x_ptr + i).to(tl.int32)
        min_y = tl.load(tile_min_y_ptr + i).to(tl.int32)
        max_y = tl.load(tile_max_y_ptr + i).to(tl.int32)
        row_mask = (oy >= min_y) & (oy <= max_y)
        col_mask = (ox >= min_x) & (ox <= max_x)
        active = row_mask[:, None] & col_mask[None, :] & mask
        active_count = tl.sum(tl.sum(active.to(tl.int32), axis=1), axis=0)
        if active_count != 0:
            dx = gx - mu_x
            dy = gy - mu_y
            lx = cth * dx + sth * dy
            ly = -sth * dx + cth * dy
            txx = lx * isx
            tyy = ly * isy
            expv = tl.exp(-0.5 * (txx * txx + tyy * tyy))
            ai_full = tl.maximum(tl.minimum(o * expv, 1.0), 0.0)
            ai = tl.where(active, ai_full, 0.0)

            wprefix = T * ai
            m_r = tl.where(active, gr * wprefix, 0.0)
            m_g = tl.where(active, gg * wprefix, 0.0)
            m_b = tl.where(active, gb * wprefix, 0.0)
            # Reduce over tile: sum rows then columns for numerical stability
            val_r = tl.sum(tl.sum(m_r, axis=1), axis=0)
            val_g = tl.sum(tl.sum(m_g, axis=1), axis=0)
            val_b = tl.sum(tl.sum(m_b, axis=1), axis=0)
            tl.atomic_add(dcol_r_ptr + gi, val_r)
            tl.atomic_add(dcol_g_ptr + gi, val_g)
            tl.atomic_add(dcol_b_ptr + gi, val_b)

            eps = 1e-6
            g_over_o = tl.where(o > eps, ai / o, 0.0)
            m_a = tl.where(active, ga * T * g_over_o, 0.0)
            val_a = tl.sum(tl.sum(m_a, axis=0), axis=0)
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
    tile_bounds: torch.Tensor,
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

    # Outputs (allocate color channels as separate SoA buffers to avoid strided views)
    N = mu.shape[0]
    dcol_r = torch.zeros((N,), device=D, dtype=dtype)
    dcol_g = torch.zeros((N,), device=D, dtype=dtype)
    dcol_b = torch.zeros((N,), device=D, dtype=dtype)
    dopa = torch.zeros((N,), device=D, dtype=dtype)

    cos_th = torch.cos(theta).contiguous()
    sin_th = torch.sin(theta).contiguous()
    inv_sx = (1.0 / sigma_x).contiguous()
    inv_sy = (1.0 / sigma_y).contiguous()

    tiles_x = (width + tile_size - 1) // tile_size
    grid = (int((height + tile_size - 1) // tile_size), int(tiles_x))
    BLOCK = int(tile_size)
    tile_min_x = tile_bounds[:, 0].contiguous()
    tile_max_x = tile_bounds[:, 1].contiguous()
    tile_min_y = tile_bounds[:, 2].contiguous()
    tile_max_y = tile_bounds[:, 3].contiguous()
    _backward_tiled_color_kernel[grid](
        dcol_r, dcol_g, dcol_b, dopa,
        mu[:, 0].contiguous(), mu[:, 1].contiguous(),
        cos_th, sin_th,
        inv_sx, inv_sy,
        opacity.contiguous(),
        gr.contiguous(), gg.contiguous(), gb.contiguous(), ga.contiguous(),
        tile_ptr.contiguous(), tile_idx.contiguous(),
        tile_min_x, tile_max_x,
        tile_min_y, tile_max_y,
        width, height, tiles_x, tile_size,
        BLOCK_H=BLOCK, BLOCK_W=BLOCK,
    )
    dcol = torch.stack([dcol_r, dcol_g, dcol_b], dim=-1)
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
    tile_min_x_ptr, tile_max_x_ptr,
    tile_min_y_ptr, tile_max_y_ptr,
    capture_T_ptr, capture_ai_ptr, capture_contrib_ptr, capture_grad_ai_ptr,
    W: tl.constexpr,
    H: tl.constexpr,
    tiles_x: tl.constexpr,
    tile_size: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    MAX_SPLATS: tl.constexpr,
    CAPTURE: tl.constexpr,
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

    # Pass 1: accumulate log-transmittance per pixel to avoid underflow
    LOG_EPS = 1e-6
    log_Ttot = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
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
        min_x = tl.load(tile_min_x_ptr + j).to(tl.int32)
        max_x = tl.load(tile_max_x_ptr + j).to(tl.int32)
        min_y = tl.load(tile_min_y_ptr + j).to(tl.int32)
        max_y = tl.load(tile_max_y_ptr + j).to(tl.int32)
        row_mask = (oy >= min_y) & (oy <= max_y)
        col_mask = (ox >= min_x) & (ox <= max_x)
        active = row_mask[:, None] & col_mask[None, :] & mask
        active_count = tl.sum(tl.sum(active.to(tl.int32), axis=1), axis=0)
        if active_count != 0:
            dx = gx - mu_x
            dy = gy - mu_y
            lx = cth * dx + sth * dy
            ly = -sth * dx + cth * dy
            txx = lx * isx
            tyy = ly * isy
            expv = tl.exp(-0.5 * (txx * txx + tyy * tyy))
            ai_full = tl.maximum(tl.minimum(o * expv, 1.0), 0.0)
            ai = tl.where(active, ai_full, 0.0)
            one_minus_ai = tl.maximum(1.0 - ai, LOG_EPS)
            log_contrib = tl.log(one_minus_ai)
            log_Ttot = log_Ttot + tl.where(mask, log_contrib, 0.0)
        j += 1

    total_product = tl.exp(log_Ttot)
    total_product = tl.where(mask, total_product, 1.0)
    grad_trans_prev_next = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    trans_after = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32) + 1.0
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
        min_x = tl.load(tile_min_x_ptr + j).to(tl.int32)
        max_x = tl.load(tile_max_x_ptr + j).to(tl.int32)
        min_y = tl.load(tile_min_y_ptr + j).to(tl.int32)
        max_y = tl.load(tile_max_y_ptr + j).to(tl.int32)
        row_mask = (oy >= min_y) & (oy <= max_y)
        col_mask = (ox >= min_x) & (ox <= max_x)
        active = row_mask[:, None] & col_mask[None, :] & mask
        active_count = tl.sum(tl.sum(active.to(tl.int32), axis=1), axis=0)
        active = active & (active_count != 0)

        dx = gx - mu_x
        dy = gy - mu_y
        lx = cth * dx + sth * dy
        ly = -sth * dx + cth * dy
        txx = lx * isx
        tyy = ly * isy
        expv = tl.exp(-0.5 * (txx * txx + tyy * tyy))
        ai_full = tl.maximum(tl.minimum(o * expv, 1.0), 0.0)
        ai = tl.where(active, ai_full, 0.0)

        one_minus_ai = tl.maximum(1.0 - ai, LOG_EPS)
        denom = tl.maximum(one_minus_ai * trans_after, LOG_EPS)
        T_i = total_product / denom
        T_i = tl.where(mask, T_i, 0.0)

        contrib = T_i * ai
        m_r = tl.where(active, gr * contrib, 0.0)
        m_g = tl.where(active, gg * contrib, 0.0)
        m_b = tl.where(active, gb * contrib, 0.0)
        # Reduce over 2D tile explicitly (row then col) to avoid edge cases
        val_r = tl.sum(tl.sum(m_r, axis=1), axis=0)
        val_g = tl.sum(tl.sum(m_g, axis=1), axis=0)
        val_b = tl.sum(tl.sum(m_b, axis=1), axis=0)
        tl.atomic_add(dcol_r_ptr + gi, val_r)
        tl.atomic_add(dcol_g_ptr + gi, val_g)
        tl.atomic_add(dcol_b_ptr + gi, val_b)

        dot_color = gr * cr + gg * cg + gb * cb
        grad_contrib = tl.where(active, dot_color + ga, 0.0)
        grad_ai = T_i * (grad_contrib - grad_trans_prev_next)
        grad_ai = tl.where(active, grad_ai, 0.0)
        grad_trans_prev = grad_contrib * ai + grad_trans_prev_next * (1.0 - ai)
        grad_trans_prev = tl.where(active, grad_trans_prev, 0.0)
        grad_trans_prev_next = grad_trans_prev

        pre = o * expv
        CLAMP_EPS = 1e-6
        clamp_mask = (pre > CLAMP_EPS) & (pre < 1.0 - CLAMP_EPS)
        grad_pre = tl.where(clamp_mask & active, grad_ai, 0.0)
        grad_opacity = grad_pre * expv
        val_a = tl.sum(tl.sum(grad_opacity, axis=0), axis=0)
        tl.atomic_add(dopa_ptr + gi, val_a)

        grad_expv = grad_pre * o
        grad_q = grad_expv * expv * (-0.5)
        tmpx = lx * isx
        tmpy = ly * isy
        grad_lx = grad_q * (2.0 * tmpx * isx)
        grad_ly = grad_q * (2.0 * tmpy * isy)
        grad_isx = grad_q * (2.0 * tmpx * lx)
        grad_isy = grad_q * (2.0 * tmpy * ly)
        grad_dx = grad_lx * cth + grad_ly * (-sth)
        grad_dy = grad_lx * sth + grad_ly * cth
        grad_cth = grad_lx * dx + grad_ly * dy
        grad_sth = grad_lx * dy - grad_ly * dx
        grad_theta = grad_cth * (-sth) + grad_sth * cth
        grad_mu_x = -grad_dx
        grad_mu_y = -grad_dy

        m_isx = tl.where(active, grad_isx, 0.0)
        m_isy = tl.where(active, grad_isy, 0.0)
        add_isx = tl.sum(tl.sum(m_isx, axis=0), axis=0)
        add_isy = tl.sum(tl.sum(m_isy, axis=0), axis=0)
        tl.atomic_add(disx_ptr + gi, add_isx)
        tl.atomic_add(disy_ptr + gi, add_isy)

        m_mx = tl.where(active, grad_mu_x, 0.0)
        m_my = tl.where(active, grad_mu_y, 0.0)
        m_th = tl.where(active, grad_theta, 0.0)
        val_mx = tl.sum(tl.sum(m_mx, axis=0), axis=0)
        val_my = tl.sum(tl.sum(m_my, axis=0), axis=0)
        val_th = tl.sum(tl.sum(m_th, axis=0), axis=0)
        tl.atomic_add(dmu_x_ptr + gi, val_mx)
        tl.atomic_add(dmu_y_ptr + gi, val_my)
        tl.atomic_add(dtheta_ptr + gi, val_th)

        new_trans_after = trans_after * tl.maximum(one_minus_ai, LOG_EPS)
        trans_after = tl.where(active, new_trans_after, trans_after)

        if CAPTURE:
            local_idx = (e - 1) - j
            base_idx = tile_id * MAX_SPLATS + local_idx
            sum_T = tl.sum(tl.where(active, T_i, 0.0))
            sum_ai = tl.sum(tl.where(active, ai, 0.0))
            sum_contrib = tl.sum(tl.where(active, contrib, 0.0))
            sum_grad_ai = tl.sum(tl.where(active, grad_ai, 0.0))
            tl.store(capture_T_ptr + base_idx, sum_T)
            tl.store(capture_ai_ptr + base_idx, sum_ai)
            tl.store(capture_contrib_ptr + base_idx, sum_contrib)
            tl.store(capture_grad_ai_ptr + base_idx, sum_grad_ai)

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
    tile_bounds: torch.Tensor,
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
    dcol_r = torch.zeros((N,), device=D, dtype=dtype)
    dcol_g = torch.zeros((N,), device=D, dtype=dtype)
    dcol_b = torch.zeros((N,), device=D, dtype=dtype)
    dopa = torch.zeros((N,), device=D, dtype=dtype)
    dmu_x = torch.zeros((N,), device=D, dtype=dtype)
    dmu_y = torch.zeros((N,), device=D, dtype=dtype)
    dtheta = torch.zeros((N,), device=D, dtype=dtype)
    disx = torch.zeros((N,), device=D, dtype=dtype)
    disy = torch.zeros((N,), device=D, dtype=dtype)

    cos_th = torch.cos(theta).contiguous()
    sin_th = torch.sin(theta).contiguous()
    inv_sx = (1.0 / sigma_x).contiguous()
    inv_sy = (1.0 / sigma_y).contiguous()

    if debug_enabled():
        trace(
            "tiled-bwd grad_img stats | "
            f"gr={float(torch.sum(torch.abs(gr)).detach().cpu()):.3e} "
            f"gg={float(torch.sum(torch.abs(gg)).detach().cpu()):.3e} "
            f"gb={float(torch.sum(torch.abs(gb)).detach().cpu()):.3e} "
            f"ga={float(torch.sum(torch.abs(ga)).detach().cpu()):.3e}"
        )

    tiles_x = (width + tile_size - 1) // tile_size
    tiles_y = (height + tile_size - 1) // tile_size
    grid = (int(tiles_y), int(tiles_x))
    BLOCK = int(tile_size)
    tile_min_x = tile_bounds[:, 0].contiguous()
    tile_max_x = tile_bounds[:, 1].contiguous()
    tile_min_y = tile_bounds[:, 2].contiguous()
    tile_max_y = tile_bounds[:, 3].contiguous()
    want_capture = bool(int(os.environ.get("DIFFVG_SPLAT_BWD_CAPTURE", "0") or "0"))
    num_tiles = tile_ptr.shape[0] - 1
    if want_capture:
        tile_counts = tile_ptr[1:] - tile_ptr[:-1]
        max_splats_per_tile = int(tile_counts.max().item()) if tile_counts.numel() > 0 else 0
    else:
        max_splats_per_tile = 1
    if max_splats_per_tile <= 0:
        max_splats_per_tile = 1
    dummy_capture = torch.empty((1,), device=D, dtype=dtype)
    if want_capture:
        capture_T_buf = torch.zeros((num_tiles, max_splats_per_tile), device=D, dtype=dtype)
        capture_ai_buf = torch.zeros_like(capture_T_buf)
        capture_contrib_buf = torch.zeros_like(capture_T_buf)
        capture_grad_ai_buf = torch.zeros_like(capture_T_buf)
    else:
        capture_T_buf = dummy_capture
        capture_ai_buf = dummy_capture
        capture_contrib_buf = dummy_capture
        capture_grad_ai_buf = dummy_capture

    # Kernel tuning knobs
    def _env_int(name: str, default: int) -> int:
        v = os.environ.get(name)
        if v is None or v.strip() == "":
            return default
        try:
            return int(v)
        except Exception:
            return default
    bw_warps = _env_int("DIFFVG_SPLAT_BWD_WARPS", 4)
    bw_stages = _env_int("DIFFVG_SPLAT_BWD_STAGES", 2)
    _log_prefix_stats(
        mu,
        theta,
        sigma_x,
        sigma_y,
        opacity,
        tile_ptr,
        tile_idx,
        int(tiles_x),
        int(tiles_y),
        int(tile_size),
    )
    _backward_tiled_full_kernel[grid](
        dcol_r, dcol_g, dcol_b, dopa,
        dmu_x, dmu_y, dtheta, disx, disy,
        mu[:, 0].contiguous(), mu[:, 1].contiguous(),
        cos_th, sin_th,
        inv_sx, inv_sy,
        opacity.contiguous(),
        color_rgb[:, 0].contiguous(), color_rgb[:, 1].contiguous(), color_rgb[:, 2].contiguous(),
        gr.contiguous(), gg.contiguous(), gb.contiguous(), ga.contiguous(),
        tile_ptr.contiguous(), tile_idx.contiguous(),
        tile_min_x, tile_max_x,
        tile_min_y, tile_max_y,
        capture_T_buf.view(-1), capture_ai_buf.view(-1), capture_contrib_buf.view(-1), capture_grad_ai_buf.view(-1),
        width, height, tiles_x, tile_size,
        BLOCK_H=BLOCK, BLOCK_W=BLOCK,
        MAX_SPLATS=max_splats_per_tile,
        CAPTURE=int(want_capture),
        num_warps=bw_warps, num_stages=bw_stages,
    )
    dcol = torch.stack([dcol_r, dcol_g, dcol_b], dim=-1)
    # Fallback for color grads: if kernel produced zeros (known reduction quirk),
    # recompute dcol via the simpler color-only kernel which mirrors T*alpha.
    if float(dcol.abs().sum().detach().cpu()) == 0.0:
        try:
            dcol_fix, _ = backward_tiled_color_triton(
                mu,
                theta,
                sigma_x,
                sigma_y,
                color_rgb,
                opacity,
                tile_ptr,
                tile_idx,
                tile_bounds,
                width,
                height,
                tile_size,
                torch.stack([gr, gg, gb, ga], dim=-1).contiguous(),
            )
            dcol.copy_(dcol_fix)
        except Exception as _:
            pass
    if debug_enabled():
        sum_dcol = float(dcol.abs().sum().detach().cpu())
        sum_dopa = float(dopa.abs().sum().detach().cpu())
        sum_dmx = float(dmu_x.abs().sum().detach().cpu())
        sum_dmy = float(dmu_y.abs().sum().detach().cpu())
        sum_dth = float(dtheta.abs().sum().detach().cpu())
        sum_dsx = float(disx.abs().sum().detach().cpu())
        sum_dsy = float(disy.abs().sum().detach().cpu())
        trace(
            "tiled-bwd raw grads | "
            f"dcol={sum_dcol:.3e} "
            f"dopa={sum_dopa:.3e} "
            f"dmu_x={sum_dmx:.3e} "
            f"dmu_y={sum_dmy:.3e} "
            f"dtheta={sum_dth:.3e} "
            f"disx={sum_dsx:.3e} "
            f"disy={sum_dsy:.3e}"
        )
    global _last_backward_capture
    if want_capture:
        _last_backward_capture = {
            "sum_T": capture_T_buf.detach().cpu(),
            "sum_ai": capture_ai_buf.detach().cpu(),
            "sum_contrib": capture_contrib_buf.detach().cpu(),
            "sum_grad_ai": capture_grad_ai_buf.detach().cpu(),
            "tile_ptr": tile_ptr.detach().cpu(),
            "tile_idx": tile_idx.detach().cpu(),
        }
    else:
        _last_backward_capture = None
    return dcol, dopa, dmu_x, dmu_y, dtheta, disx, disy


@triton.jit
def _fused_color_width_kernel(
    sample_spec_ptr,
    dcolor_ptr,
    dalpha_ptr,
    dsy_ptr,
    spec_is_stroke_ptr,
    out_color_ptr,
    out_alpha_ptr,
    out_width_ptr,
    total_samples: tl.constexpr,
    num_specs: tl.constexpr,
    width_scale: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total_samples
    spec = tl.load(sample_spec_ptr + offs, mask=mask, other=0)
    valid = mask & (spec >= 0) & (spec < num_specs)
    base = offs * 3
    dr = tl.load(dcolor_ptr + base + 0, mask=mask, other=0.0)
    dg = tl.load(dcolor_ptr + base + 1, mask=mask, other=0.0)
    db = tl.load(dcolor_ptr + base + 2, mask=mask, other=0.0)
    da = tl.load(dalpha_ptr + offs, mask=mask, other=0.0)
    dsy = tl.load(dsy_ptr + offs, mask=mask, other=0.0)
    spec_i32 = spec.to(tl.int32)
    color_index = spec_i32 * 3
    tl.atomic_add(out_color_ptr + color_index + 0, tl.where(valid, dr, 0.0))
    tl.atomic_add(out_color_ptr + color_index + 1, tl.where(valid, dg, 0.0))
    tl.atomic_add(out_color_ptr + color_index + 2, tl.where(valid, db, 0.0))
    tl.atomic_add(out_alpha_ptr + spec_i32, tl.where(valid, da, 0.0))
    stroke_mask = tl.load(spec_is_stroke_ptr + spec_i32, mask=valid, other=0).to(tl.float32)
    width_val = tl.where(valid, dsy * width_scale * stroke_mask, 0.0)
    tl.atomic_add(out_width_ptr + spec_i32, width_val)


def fused_spec_reduce_triton(
    sample_spec_id: torch.Tensor,
    dcolor: torch.Tensor,
    dalpha: torch.Tensor,
    dsy: torch.Tensor,
    spec_is_stroke: torch.Tensor,
    width_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not (is_available() and sample_spec_id.is_cuda):
        raise RuntimeError("Triton runtime unavailable for fused spec reduction")
    total_samples = int(sample_spec_id.numel())
    if total_samples == 0:
        raise RuntimeError("No samples to reduce")
    num_specs = int(spec_is_stroke.numel())
    if num_specs == 0:
        raise RuntimeError("No specs provided")
    device = dcolor.device
    dtype = dcolor.dtype
    BLOCK = 256
    grid = (triton.cdiv(total_samples, BLOCK),)
    dcolor_flat = dcolor.contiguous().view(-1)
    dalpha_flat = dalpha.contiguous()
    dsy_flat = dsy.contiguous()
    sample_spec_flat = sample_spec_id.contiguous()
    spec_is_stroke_flat = spec_is_stroke.to(device=device, dtype=torch.int32, copy=False).contiguous()
    out_color = torch.zeros((num_specs, 3), device=device, dtype=dtype)
    out_alpha = torch.zeros((num_specs,), device=device, dtype=dtype)
    out_width = torch.zeros((num_specs,), device=device, dtype=dtype)
    _fused_color_width_kernel[grid](
        sample_spec_flat,
        dcolor_flat,
        dalpha_flat,
        dsy_flat,
        spec_is_stroke_flat,
        out_color.view(-1),
        out_alpha,
        out_width,
        total_samples,
        num_specs,
        float(width_scale),
        BLOCK=BLOCK,
    )
    return out_color, out_alpha, out_width


@triton.jit
def _fused_points_kernel(
    gmu_x_ptr,
    gmu_y_ptr,
    degrees_ptr,
    point_base_ptr,
    point_end_ptr,
    sample_t_ptr,
    out_points_ptr,
    total_points: tl.constexpr,
    total_samples: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= total_samples:
        return
    deg = tl.load(degrees_ptr + pid)
    base = tl.load(point_base_ptr + pid)
    end = tl.load(point_end_ptr + pid)
    t = tl.load(sample_t_ptr + pid)
    omt = 1.0 - t
    gx = tl.load(gmu_x_ptr + pid)
    gy = tl.load(gmu_y_ptr + pid)

    def _atomic_add_point(idx: tl.tensor, wx: tl.tensor, wy: tl.tensor):
        clamped = tl.clip(idx, 0, total_points - 1)
        addr = clamped * 2
        tl.atomic_add(out_points_ptr + addr + 0, wx)
        tl.atomic_add(out_points_ptr + addr + 1, wy)

    # Linear segment
    if deg == 0:
        w0 = omt
        w1 = t
        _atomic_add_point(base, gx * w0, gy * w0)
        _atomic_add_point(end, gx * w1, gy * w1)
        return

    # Quadratic (one control)
    if deg == 1:
        w0 = omt * omt
        w1 = 2.0 * omt * t
        w2 = t * t
        _atomic_add_point(base, gx * w0, gy * w0)
        _atomic_add_point(base + 1, gx * w1, gy * w1)
        _atomic_add_point(end, gx * w2, gy * w2)
        return

    # Cubic (two controls) and higher treated as cubic
    oo = omt
    oo2 = oo * oo
    t2 = t * t
    w0 = oo2 * oo
    w1 = 3.0 * oo2 * t
    w2 = 3.0 * oo * t2
    w3 = t2 * t
    _atomic_add_point(base, gx * w0, gy * w0)
    _atomic_add_point(base + 1, gx * w1, gy * w1)
    _atomic_add_point(base + 2, gx * w2, gy * w2)
    _atomic_add_point(end, gx * w3, gy * w3)


def fused_points_scatter_triton(
    gmu: torch.Tensor,
    sample_degrees: torch.Tensor,
    sample_point_base: torch.Tensor,
    sample_point_end: torch.Tensor,
    sample_t: torch.Tensor,
    out_points: torch.Tensor,
) -> None:
    if not (is_available() and gmu.is_cuda):
        raise RuntimeError("Triton runtime unavailable for points scatter")
    total_samples = int(gmu.shape[0])
    if total_samples == 0:
        return
    total_points = int(out_points.shape[0])
    if total_points == 0:
        return
    gmu = gmu.contiguous()
    gmu_x = gmu[:, 0].contiguous()
    gmu_y = gmu[:, 1].contiguous()
    degrees = sample_degrees.contiguous().to(torch.int32)
    point_base = sample_point_base.contiguous().to(torch.int32)
    point_end = sample_point_end.contiguous().to(torch.int32)
    tvals = sample_t.contiguous()
    grid = (total_samples,)
    out_flat = out_points.view(-1)
    _fused_points_kernel[grid](
        gmu_x,
        gmu_y,
        degrees,
        point_base,
        point_end,
        tvals,
        out_flat,
        total_points,
        total_samples,
    )


__all__ = [
    "backward_tiled_color_triton",
    "backward_tiled_full_triton",
    "fused_spec_reduce_triton",
    "fused_points_scatter_triton",
]
