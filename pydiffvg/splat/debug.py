from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from .trace import trace, debug_enabled


_LOG_EPS = 1e-6
_CLAMP_EPS = 1e-6


def _collect_tile_records(
    idxs: torch.Tensor,
    gy: torch.Tensor,
    gx: torch.Tensor,
    mu: torch.Tensor,
    cos_th: torch.Tensor,
    sin_th: torch.Tensor,
    inv_sx: torch.Tensor,
    inv_sy: torch.Tensor,
    opacity: torch.Tensor,
    color_rgb: torch.Tensor,
) -> Tuple[List[Dict[str, torch.Tensor]], torch.Tensor, torch.Tensor]:
    """Collect per-splat intermediates for a tile to share between debug helpers."""
    device = gy.device
    records: List[Dict[str, torch.Tensor]] = []
    trans_prev = torch.ones((gy.shape[0], gx.shape[1]), device=device, dtype=torch.float64)
    log_Ttot = torch.zeros_like(trans_prev)

    for k in range(idxs.shape[0]):
        gi = int(idxs[k].item())
        mu_x_val = mu[gi, 0]
        mu_y_val = mu[gi, 1]
        cos_val = cos_th[gi]
        sin_val = sin_th[gi]
        invsx_val = inv_sx[gi]
        invsy_val = inv_sy[gi]
        opacity_val = opacity[gi]
        color_val = color_rgb[gi]

        dx = gx - mu_x_val
        dy = gy - mu_y_val
        lx = cos_val * dx + sin_val * dy
        ly = -sin_val * dx + cos_val * dy
        txx = lx * invsx_val
        tyy = ly * invsy_val
        expv = torch.exp(-0.5 * (txx * txx + tyy * tyy))
        pre = opacity_val * expv
        ai = torch.clamp(pre, 0.0, 1.0)
        one_minus_ai = torch.clamp(1.0 - ai, min=_LOG_EPS)

        records.append(
            {
                "idx": gi,
                "T": trans_prev.clone(),
                "ai": ai,
                "pre": pre,
                "expv": expv,
                "lx": lx,
                "ly": ly,
                "txx": txx,
                "tyy": tyy,
                "dx": dx,
                "dy": dy,
                "cos": cos_val,
                "sin": sin_val,
                "invsx": invsx_val,
                "invsy": invsy_val,
                "opacity": opacity_val,
                "color": color_val,
                "clamp_mask": (pre > _CLAMP_EPS) & (pre < 1.0 - _CLAMP_EPS),
            }
        )

        log_Ttot = log_Ttot + torch.log(one_minus_ai).to(torch.float64)
        trans_prev = trans_prev * one_minus_ai.to(torch.float64)

    return records, log_Ttot, trans_prev


def backward_tiled_full_python(
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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = mu.device
    dtype = mu.dtype
    N = mu.shape[0]
    dcol = torch.zeros((N, 3), device=device, dtype=dtype)
    dopa = torch.zeros((N,), device=device, dtype=dtype)
    dmu_x = torch.zeros((N,), device=device, dtype=dtype)
    dmu_y = torch.zeros((N,), device=device, dtype=dtype)
    dtheta = torch.zeros((N,), device=device, dtype=dtype)
    dinv_sx = torch.zeros((N,), device=device, dtype=dtype)
    dinv_sy = torch.zeros((N,), device=device, dtype=dtype)

    tiles_x = (width + tile_size - 1) // tile_size
    tiles_y = (height + tile_size - 1) // tile_size

    gy_coords = torch.arange(height, device=device, dtype=dtype).unsqueeze(1) + 0.5
    gx_coords = torch.arange(width, device=device, dtype=dtype).unsqueeze(0) + 0.5

    cos_th = torch.cos(theta)
    sin_th = torch.sin(theta)
    inv_sx = 1.0 / sigma_x
    inv_sy = 1.0 / sigma_y

    for ty in range(tiles_y):
        for tx in range(tiles_x):
            tile_id = ty * tiles_x + tx
            start = int(tile_ptr[tile_id].item())
            end = int(tile_ptr[tile_id + 1].item())
            if end <= start:
                continue
            idxs = tile_idx[start:end].to(dtype=torch.int64)

            y0 = ty * tile_size
            x0 = tx * tile_size
            y1 = min(y0 + tile_size, height)
            x1 = min(x0 + tile_size, width)
            if y0 >= y1 or x0 >= x1:
                continue

            gy = gy_coords[y0:y1, :1]
            gx = gx_coords[:1, x0:x1]

            grad_patch = grad_img[y0:y1, x0:x1]
            if grad_patch.shape[-1] == 4:
                gr = grad_patch[..., 0]
                gg = grad_patch[..., 1]
                gb = grad_patch[..., 2]
                ga = grad_patch[..., 3]
            else:
                gr = grad_patch[..., 0]
                gg = grad_patch[..., 1]
                gb = grad_patch[..., 2]
                ga = torch.zeros_like(gr)

            records, _, trans_prev = _collect_tile_records(
                idxs, gy, gx, mu, cos_th, sin_th, inv_sx, inv_sy, opacity, color_rgb
            )

            grad_trans_prev_next = torch.zeros_like(trans_prev)

            for rec in reversed(records):
                gi = int(rec["idx"])
                T = rec["T"]
                ai = rec["ai"]
                expv = rec["expv"]
                lx = rec["lx"]
                ly = rec["ly"]
                txx = rec["txx"]
                tyy = rec["tyy"]
                dx = rec["dx"]
                dy = rec["dy"]
                cos_val = rec["cos"]
                sin_val = rec["sin"]
                invsx_val = rec["invsx"]
                invsy_val = rec["invsy"]
                opacity_val = rec["opacity"]
                color_val = rec["color"]
                clamp_mask = rec["clamp_mask"]

                T_f32 = T.to(dtype)
                ai_f32 = ai

                contrib = T_f32 * ai_f32
                dcol_val = torch.stack(
                    [
                        torch.sum(gr * contrib),
                        torch.sum(gg * contrib),
                        torch.sum(gb * contrib),
                    ]
                )
                dcol[gi] += dcol_val

                dot_color = gr * color_val[0] + gg * color_val[1] + gb * color_val[2]
                grad_contrib = dot_color + ga
                grad_ai = T_f32 * (grad_contrib - grad_trans_prev_next.to(dtype))
                grad_trans_prev = grad_contrib * ai_f32 + grad_trans_prev_next.to(dtype) * (1.0 - ai_f32)
                grad_trans_prev_next = grad_trans_prev.to(torch.float64)

                grad_ai = torch.where(clamp_mask, grad_ai, torch.zeros_like(grad_ai))

                grad_opacity = grad_ai * expv
                grad_expv = grad_ai * opacity_val
                dopa[gi] += torch.sum(grad_opacity)

                grad_txx = grad_expv * (-expv) * txx
                grad_tyy = grad_expv * (-expv) * tyy

                grad_lx = grad_txx * invsx_val
                grad_ly = grad_tyy * invsy_val
                grad_invsx = grad_txx * lx
                grad_invsy = grad_tyy * ly

                grad_cth = grad_lx * dx + grad_ly * dy
                grad_sth = grad_lx * dy - grad_ly * dx

                grad_dx = grad_lx * cos_val - grad_ly * sin_val
                grad_dy = grad_lx * sin_val + grad_ly * cos_val

                dmu_x[gi] += -torch.sum(grad_dx)
                dmu_y[gi] += -torch.sum(grad_dy)

                grad_theta = grad_cth * (-sin_val) + grad_sth * cos_val
                dtheta[gi] += torch.sum(grad_theta)

                dinv_sx[gi] += torch.sum(grad_invsx)
                dinv_sy[gi] += torch.sum(grad_invsy)

    disx = dinv_sx
    disy = dinv_sy

    if debug_enabled():
        trace(
            "python-bwd raw grads | "
            f"dcol={float(dcol.abs().sum().detach().cpu()):.3e} "
            f"dopa={float(dopa.abs().sum().detach().cpu()):.3e} "
            f"dmu_x={float(dmu_x.abs().sum().detach().cpu()):.3e} "
            f"dmu_y={float(dmu_y.abs().sum().detach().cpu()):.3e} "
            f"dtheta={float(dtheta.abs().sum().detach().cpu()):.3e} "
            f"disx={float(disx.abs().sum().detach().cpu()):.3e} "
            f"disy={float(disy.abs().sum().detach().cpu()):.3e}"
        )

    return dcol, dopa, dmu_x, dmu_y, dtheta, disx, disy


def backward_tiled_full_kernel_formula(
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
    *,
    return_intermediates: bool = False,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[List[Dict[str, float]]],
]:
    """Pure-Torch mirror of the Triton backward kernel math for debugging."""
    device = mu.device
    dtype = mu.dtype
    N = mu.shape[0]
    dcol = torch.zeros((N, 3), device=device, dtype=dtype)
    dopa = torch.zeros((N,), device=device, dtype=dtype)
    dmu_x = torch.zeros((N,), device=device, dtype=dtype)
    dmu_y = torch.zeros((N,), device=device, dtype=dtype)
    dtheta = torch.zeros((N,), device=device, dtype=dtype)
    disx = torch.zeros((N,), device=device, dtype=dtype)
    disy = torch.zeros((N,), device=device, dtype=dtype)

    tiles_x = (width + tile_size - 1) // tile_size
    tiles_y = (height + tile_size - 1) // tile_size

    gy_coords = torch.arange(height, device=device, dtype=dtype).unsqueeze(1) + 0.5
    gx_coords = torch.arange(width, device=device, dtype=dtype).unsqueeze(0) + 0.5

    cos_th = torch.cos(theta)
    sin_th = torch.sin(theta)
    inv_sx = 1.0 / sigma_x
    inv_sy = 1.0 / sigma_y

    intermediates: Optional[List[Dict[str, float]]] = [] if return_intermediates else None

    for ty in range(tiles_y):
        for tx in range(tiles_x):
            tile_id = ty * tiles_x + tx
            start = int(tile_ptr[tile_id].item())
            end = int(tile_ptr[tile_id + 1].item())
            if end <= start:
                continue
            idxs = tile_idx[start:end].to(dtype=torch.int64)

            y0 = ty * tile_size
            x0 = tx * tile_size
            y1 = min(y0 + tile_size, height)
            x1 = min(x0 + tile_size, width)
            if y0 >= y1 or x0 >= x1:
                continue

            gy = gy_coords[y0:y1, :1]
            gx = gx_coords[:1, x0:x1]

            grad_patch = grad_img[y0:y1, x0:x1]
            if grad_patch.shape[-1] == 4:
                gr = grad_patch[..., 0]
                gg = grad_patch[..., 1]
                gb = grad_patch[..., 2]
                ga = grad_patch[..., 3]
            else:
                gr = grad_patch[..., 0]
                gg = grad_patch[..., 1]
                gb = grad_patch[..., 2]
                ga = torch.zeros_like(gr)

            records, log_Ttot, _ = _collect_tile_records(
                idxs, gy, gx, mu, cos_th, sin_th, inv_sx, inv_sy, opacity, color_rgb
            )
            if not records:
                continue

            total_product = torch.exp(log_Ttot)
            grad_trans_prev_next = torch.zeros_like(total_product)
            trans_after = torch.ones_like(total_product)

            if intermediates is not None:
                tile_entries: List[Dict[str, float]] = []
            else:
                tile_entries = []

            for rec in reversed(records):
                gi = int(rec["idx"])
                ai = rec["ai"]
                expv = rec["expv"]
                lx = rec["lx"]
                ly = rec["ly"]
                dx = rec["dx"]
                dy = rec["dy"]
                cth = rec["cos"]
                sth = rec["sin"]
                invsx_val = rec["invsx"]
                invsy_val = rec["invsy"]
                opacity_val = rec["opacity"]
                color_val = rec["color"]
                clamp_mask = rec["clamp_mask"]

                one_minus_ai = torch.clamp(1.0 - ai, min=_LOG_EPS)
                denom = torch.clamp(one_minus_ai.to(torch.float64) * trans_after, min=_LOG_EPS)
                T_i = total_product / denom
                ai64 = ai.to(torch.float64)

                contrib = (T_i * ai64).to(dtype)
                dcol[gi, 0] += torch.sum(gr * contrib)
                dcol[gi, 1] += torch.sum(gg * contrib)
                dcol[gi, 2] += torch.sum(gb * contrib)

                dot_color = gr * color_val[0] + gg * color_val[1] + gb * color_val[2]
                grad_contrib = dot_color + ga
                grad_ai64 = T_i * (grad_contrib.to(torch.float64) - grad_trans_prev_next)
                grad_trans_prev = grad_contrib.to(torch.float64) * ai64 + grad_trans_prev_next * (1.0 - ai64)
                grad_trans_prev_next = grad_trans_prev

                grad_pre64 = torch.where(clamp_mask, grad_ai64, torch.zeros_like(grad_ai64))
                grad_pre = grad_pre64.to(dtype)
                grad_opacity = grad_pre * expv
                dopa[gi] += torch.sum(grad_opacity)

                grad_expv = grad_pre * opacity_val
                grad_q = grad_expv * expv * (-0.5)
                tmpx = lx * invsx_val
                tmpy = ly * invsy_val
                grad_lx = grad_q * (2.0 * tmpx * invsx_val)
                grad_ly = grad_q * (2.0 * tmpy * invsy_val)
                grad_isx = grad_q * (2.0 * tmpx * lx)
                grad_isy = grad_q * (2.0 * tmpy * ly)
                grad_dx = grad_lx * cth + grad_ly * (-sth)
                grad_dy = grad_lx * sth + grad_ly * cth
                grad_cth = grad_lx * dx + grad_ly * dy
                grad_sth = grad_lx * dy - grad_ly * dx
                grad_theta = grad_cth * (-sth) + grad_sth * cth
                grad_mu_x = -grad_dx
                grad_mu_y = -grad_dy

                disx[gi] += torch.sum(grad_isx)
                disy[gi] += torch.sum(grad_isy)
                dmu_x[gi] += torch.sum(grad_mu_x)
                dmu_y[gi] += torch.sum(grad_mu_y)
                dtheta[gi] += torch.sum(grad_theta)

                if intermediates is not None:
                    T_record = rec["T"].to(torch.float64)
                    diff_max = torch.abs(T_record - T_i).max()
                    tile_entries.insert(
                        0,
                        {
                            "tile_id": tile_id,
                            "idx": gi,
                            "T_prefix_sum": float(T_record.sum().detach().cpu()),
                            "T_recon_sum": float(T_i.sum().detach().cpu()),
                            "T_diff_max": float(diff_max.detach().cpu()),
                            "alpha_sum": float(ai.to(torch.float64).sum().detach().cpu()),
                        },
                    )

                trans_after = trans_after * torch.clamp(one_minus_ai.to(torch.float64), min=_LOG_EPS)

            if intermediates is not None:
                intermediates.extend(tile_entries)

    info: Optional[List[Dict[str, float]]] = intermediates if return_intermediates else None
    return dcol, dopa, dmu_x, dmu_y, dtheta, disx, disy, info


__all__ = ["backward_tiled_full_python", "backward_tiled_full_kernel_formula"]
