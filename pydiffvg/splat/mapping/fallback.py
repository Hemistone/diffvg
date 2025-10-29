from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch

from ..trace import debug_enabled as _debug_enabled, trace as _trace


def _mapper_fail(saved: dict, reason: str) -> None:
    if isinstance(saved, dict) and "_fused_mapper_error" not in saved:
        saved["_fused_mapper_error"] = reason
    return None


def _prepare_mapping_inputs(
    saved: dict,
    request,
    dcolor: torch.Tensor,
    dalpha: torch.Tensor,
    dmu_x: torch.Tensor,
    dmu_y: torch.Tensor,
    dtheta: torch.Tensor,
    dsx: torch.Tensor,
    dsy: torch.Tensor,
):
    spec_counts = saved.get("spec_counts")
    color_refs = saved.get("color_rgba_refs")
    width_refs = saved.get("stroke_width_refs")
    seg_idx_list = saved.get("seg_idx_list")
    t_list = saved.get("t_list")
    points_refs = saved.get("points_refs")
    control_counts_list = saved.get("control_counts")
    if not isinstance(spec_counts, list) or color_refs is None or width_refs is None:
        return None
    if seg_idx_list is None or t_list is None or points_refs is None or control_counts_list is None:
        return None

    total_samples = int(dcolor.shape[0])
    if total_samples == 0:
        return None

    mu = saved.get("mu")
    theta_vals = saved.get("theta")
    sigma_x_vals = saved.get("sigma_x")
    sigma_y_vals = saved.get("sigma_y")
    if not isinstance(mu, torch.Tensor) or not isinstance(theta_vals, torch.Tensor):
        return None
    if not isinstance(sigma_x_vals, torch.Tensor) or not isinstance(sigma_y_vals, torch.Tensor):
        return None

    order = saved.get("order")
    if isinstance(order, torch.Tensor) and order.numel() == total_samples:
        inv = torch.empty_like(order)
        inv[order] = torch.arange(order.numel(), device=order.device, dtype=order.dtype)

        def _unsort(t: torch.Tensor) -> torch.Tensor:
            if not isinstance(t, torch.Tensor) or t.shape[0] != inv.shape[0]:
                return t
            return t.index_select(0, inv)

        dcolor_g = _unsort(dcolor)
        dalpha_g = _unsort(dalpha)
        dmu_x_g = _unsort(dmu_x)
        dmu_y_g = _unsort(dmu_y)
        dtheta_g = _unsort(dtheta)
        dsx_g = _unsort(dsx)
        dsy_g = _unsort(dsy)
        mu_g = _unsort(mu)
        theta_g = _unsort(theta_vals)
        sigma_x_g = _unsort(sigma_x_vals)
        sigma_y_g = _unsort(sigma_y_vals)
    else:
        dcolor_g, dalpha_g = dcolor, dalpha
        dmu_x_g, dmu_y_g = dmu_x, dmu_y
        dtheta_g, dsx_g, dsy_g = dtheta, dsx, dsy
        mu_g, theta_g = mu, theta_vals
        sigma_x_g, sigma_y_g = sigma_x_vals, sigma_y_vals

    device = mu_g.device
    if spec_counts:
        counts_tensor = torch.tensor(spec_counts, device=device, dtype=torch.int32)
        spec_offsets = torch.zeros((counts_tensor.numel() + 1,), device=device, dtype=torch.int32)
        spec_offsets[0] = 0
        spec_offsets[1:] = torch.cumsum(counts_tensor, dim=0)
    else:
        return None
    if spec_offsets[-1].item() != total_samples:
        return None

    rho = max(float(request.config.rho), 1e-6)
    fwhm_coeff = 2.0 * math.sqrt(2.0 * math.log(2.0))
    width_scale = 1.0 / (fwhm_coeff * rho)

    return {
        "rho": rho,
        "spec_counts": spec_counts,
        "color_refs": color_refs,
        "width_refs": width_refs,
        "seg_idx_list": seg_idx_list,
        "t_list": t_list,
        "points_refs": points_refs,
        "control_counts_list": control_counts_list,
        "mu_g": mu_g,
        "theta_g": theta_g,
        "sigma_x_g": sigma_x_g,
        "sigma_y_g": sigma_y_g,
        "dcolor_g": dcolor_g,
        "dalpha_g": dalpha_g,
        "dmu_x_g": dmu_x_g,
        "dmu_y_g": dmu_y_g,
        "dtheta_g": dtheta_g,
        "dsx_g": dsx_g,
        "dsy_g": dsy_g,
        "spec_offsets": spec_offsets,
        "width_scale": width_scale,
    }


def _map_triton_grads_to_slots_python(
    saved: dict,
    request,
    args_with_grad: Tuple[object, ...],
    grad_slots,
    dcolor: torch.Tensor,
    dalpha: torch.Tensor,
    dmu_x: torch.Tensor,
    dmu_y: torch.Tensor,
    dtheta: torch.Tensor,
    dsx: torch.Tensor,
    dsy: torch.Tensor,
) -> Optional[Tuple[Optional[torch.Tensor], ...]]:
    """Bridge Triton gradients back to original scene tensors (Python fallback)."""

    prepared = _prepare_mapping_inputs(
        saved, request, dcolor, dalpha, dmu_x, dmu_y, dtheta, dsx, dsy
    )
    if prepared is None:
        return None

    spec_counts = prepared["spec_counts"]
    color_refs = prepared["color_refs"]
    width_refs = prepared["width_refs"]
    seg_idx_list = prepared["seg_idx_list"]
    t_list = prepared["t_list"]
    points_refs = prepared["points_refs"]
    control_counts_list = prepared["control_counts_list"]
    rho = prepared["rho"]
    mu_g = prepared["mu_g"]
    theta_g = prepared["theta_g"]
    sigma_x_g = prepared["sigma_x_g"]
    sigma_y_g = prepared["sigma_y_g"]
    dcolor_g = prepared["dcolor_g"]
    dalpha_g = prepared["dalpha_g"]
    dmu_x_g = prepared["dmu_x_g"]
    dmu_y_g = prepared["dmu_y_g"]
    dtheta_g = prepared["dtheta_g"]
    dsx_g = prepared["dsx_g"]
    dsy_g = prepared["dsy_g"]
    spec_offsets = prepared["spec_offsets"]
    width_scale = prepared["width_scale"]

    device = mu_g.device
    dtype = mu_g.dtype

    override: dict[int, torch.Tensor] = {}

    for si, count in enumerate(spec_counts):
        idx0 = int(spec_offsets[si].item())
        idx1 = int(spec_offsets[si + 1].item())
        cref = color_refs[si] if si < len(color_refs) else None
        if isinstance(cref, torch.Tensor):
            if idx1 > idx0:
                gcol = dcolor_g[idx0:idx1].sum(dim=0)
                gopa = dalpha_g[idx0:idx1].sum()
            else:
                gcol = dcolor_g.new_zeros((3,))
                gopa = dalpha_g.new_zeros(())
            grad_rgba = torch.cat([gcol, gopa.unsqueeze(0)], dim=0).to(
                dtype=cref.dtype, device=cref.device
            ).view_as(cref).detach()
            override[id(cref)] = grad_rgba

        wref = width_refs[si] if si < len(width_refs) else None
        if isinstance(wref, torch.Tensor):
            gw = dsy_g[idx0:idx1].sum() * width_scale if idx1 > idx0 else dsy_g.new_zeros(())
            grad_width = gw.to(dtype=wref.dtype, device=wref.device).expand_as(wref).detach()
            override[id(wref)] = grad_width

        pref = points_refs[si] if si < len(points_refs) else None
        seg_idx_spec = seg_idx_list[si] if si < len(seg_idx_list) else None
        tvals_spec = t_list[si] if si < len(t_list) else None
        ctrl_counts = control_counts_list[si] if si < len(control_counts_list) else None
        if not (
            isinstance(pref, torch.Tensor)
            and isinstance(seg_idx_spec, torch.Tensor)
            and isinstance(tvals_spec, torch.Tensor)
            and isinstance(ctrl_counts, list)
        ):
            continue
        if idx1 <= idx0:
            override[id(pref)] = torch.zeros_like(pref)
            continue

        mu_spec = mu_g[idx0:idx1]
        gmu = torch.stack([dmu_x_g[idx0:idx1], dmu_y_g[idx0:idx1]], dim=-1)
        dsx_local = dsx_g[idx0:idx1]
        sx_local = sigma_x_g[idx0:idx1]
        seg_idx = seg_idx_spec.to(torch.int64)
        tvals = tvals_spec.to(dtype)
        omt = 1.0 - tvals
        cnt = idx1 - idx0

        extra = torch.zeros_like(gmu)
        if cnt >= 2:
            diffs = mu_spec[1:] - mu_spec[:-1]
            den = torch.linalg.norm(diffs, dim=-1, keepdim=True).clamp_min(1e-6)
            u = diffs / den
            clamp_mask = sx_local <= (1e-3 + 1e-12)
            dsx_scaled = torch.where(clamp_mask, torch.zeros_like(dsx_local), dsx_local * (1.0 / rho))
            g_to_dist = 0.5 * dsx_scaled[:-1] + 0.5 * dsx_scaled[1:]
            g_to_dist[0] = g_to_dist[0] + 0.5 * dsx_scaled[0]
            g_to_dist[-1] = g_to_dist[-1] + 0.5 * dsx_scaled[-1]
            extra[:-1] += -u * g_to_dist.unsqueeze(-1)
            extra[1:] += u * g_to_dist.unsqueeze(-1)
        gmu = gmu + extra

        D = pref.device
        pref_dtype = pref.dtype
        gp = torch.zeros_like(pref, device=D)
        cc_t = torch.tensor(ctrl_counts, device=D, dtype=torch.int64)
        if cc_t.numel() == 0:
            override[id(pref)] = gp
            continue
        S = cc_t.numel()
        s_ids = torch.arange(S, device=D, dtype=torch.int64)
        csum = torch.cumsum(cc_t, dim=0)
        offsets = torch.cat([torch.zeros((1,), device=D, dtype=torch.int64), csum[:-1]], dim=0)
        si0_arr = s_ids + offsets
        si_end_arr = (s_ids + 1) + csum
        si0_all = si0_arr[seg_idx]
        si_end_all = si_end_arr[seg_idx]
        deg = cc_t[seg_idx]

        m0 = deg == 0
        m1 = deg == 1
        m2 = deg >= 2

        if m0.any():
            w0 = omt[m0]
            w1 = tvals[m0]
            si0 = si0_all[m0]
            si_end = si_end_all[m0]
            g = gmu[m0]
            gp.index_add_(0, si0, g * w0.unsqueeze(-1))
            gp.index_add_(0, si_end, g * w1.unsqueeze(-1))
        if m1.any():
            tt = tvals[m1]
            oo = omt[m1]
            base = si0_all[m1]
            ci0 = base + 1
            si0 = base
            si_end = si_end_all[m1]
            g = gmu[m1]
            gp.index_add_(0, si0, g * (oo * oo).unsqueeze(-1))
            gp.index_add_(0, ci0, g * (2.0 * oo * tt).unsqueeze(-1))
            gp.index_add_(0, si_end, g * (tt * tt).unsqueeze(-1))
        if m2.any():
            tt = tvals[m2]
            oo = omt[m2]
            oo2 = oo * oo
            t2 = tt * tt
            base = si0_all[m2]
            ci0 = base + 1
            ci1 = base + 2
            si0 = base
            si_end = si_end_all[m2]
            g = gmu[m2]
            gp.index_add_(0, si0, g * (oo2 * oo).unsqueeze(-1))
            gp.index_add_(0, ci0, g * (3.0 * oo2 * tt).unsqueeze(-1))
            gp.index_add_(0, ci1, g * (3.0 * oo * t2).unsqueeze(-1))
            gp.index_add_(0, si_end, g * (t2 * tt).unsqueeze(-1))

        gth_all = dtheta_g[idx0:idx1]
        if torch.any(gth_all != 0):
            ptsd = pref.detach()
            cnt_i = pref.shape[0]
            use_central = torch.zeros((cnt,), dtype=torch.bool, device=D)
            if cnt >= 3:
                c_vec = mu_spec[2:] - mu_spec[:-2]
                cn2 = (c_vec[:, 0] * c_vec[:, 0] + c_vec[:, 1] * c_vec[:, 1])
                good = cn2 > 1e-8
                use_central[1:-1] = good
                if good.any():
                    cg = c_vec[good]
                    cn2g = cn2[good]
                    gvec = torch.stack(
                        [-cg[:, 1] / cn2g, cg[:, 0] / cn2g], dim=-1
                    ) * gth_all[1:-1][good].unsqueeze(-1)
                    gmu_add = torch.zeros_like(gmu)
                    idxs = torch.nonzero(good, as_tuple=False).squeeze(1)
                    idx_plus = idxs + 2
                    idx_minus = idxs
                    gmu_add.index_add_(0, idx_plus, gvec)
                    gmu_add.index_add_(0, idx_minus, -gvec)
                    gmu = gmu + gmu_add
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
                    gp.index_add_(0, si_end, gv)
                if m1_t.any():
                    base = si0_all[m1_t]
                    si0 = base
                    ci0 = base + 1
                    si_end = si_end_all[m1_t]
                    tt = tvals[m1_t]
                    oo = omt[m1_t]
                    v = (
                        (2.0 * oo * (pref[ci0] - pref[si0]))
                        + (2.0 * tt * (pref[si_end] - pref[ci0]))
                    )
                    denom = (v[:, 0] * v[:, 0] + v[:, 1] * v[:, 1]).clamp_min(1e-6)
                    gv = torch.stack([-v[:, 1] / denom, v[:, 0] / denom], dim=-1) * gth_all[m1_t].unsqueeze(-1)
                    dt0 = -oo * oo
                    dt1 = (oo * oo) - (2.0 * oo * tt)
                    dt2 = 2.0 * oo * tt
                    gp.index_add_(0, si0, gv * dt0.unsqueeze(-1))
                    gp.index_add_(0, ci0, gv * dt1.unsqueeze(-1))
                    gp.index_add_(0, si_end, gv * dt2.unsqueeze(-1))
                if m2_t.any():
                    base = si0_all[m2_t]
                    ci0 = base + 1
                    ci1 = base + 2
                    si_end = si_end_all[m2_t]
                    tt = tvals[m2_t]
                    oo = omt[m2_t]
                    oo2 = oo * oo
                    t2 = tt * tt
                    v = (
                        3.0 * oo2.unsqueeze(-1) * (pref[ci0] - pref[base])
                        + 6.0 * (oo * tt).unsqueeze(-1) * (pref[ci1] - pref[ci0])
                        + 3.0 * t2.unsqueeze(-1) * (pref[si_end] - pref[ci1])
                    )
                    denom = (v[:, 0] * v[:, 0] + v[:, 1] * v[:, 1]).clamp_min(1e-6)
                    gv = torch.stack(
                        [-v[:, 1] / denom, v[:, 0] / denom],
                        dim=-1,
                    ) * gth_all[m2_t].unsqueeze(-1)
                    dt0 = -3.0 * oo2
                    dt1 = 3.0 * oo2 - 6.0 * oo * tt
                    dt2 = 6.0 * oo * tt - 3.0 * t2
                    dt3 = 3.0 * t2
                    gp.index_add_(0, base, gv * dt0.unsqueeze(-1))
                    gp.index_add_(0, ci0, gv * dt1.unsqueeze(-1))
                    gp.index_add_(0, ci1, gv * dt2.unsqueeze(-1))
                    gp.index_add_(0, si_end, gv * dt3.unsqueeze(-1))

        override[id(pref)] = gp.to(dtype=pref_dtype, device=pref.device).detach()

    if not grad_slots:
        return None

    total_args = len(args_with_grad)
    grad_list: List[Optional[torch.Tensor]] = [None] * (6 + total_args)
    input_tensors = [slot.tensor for slot in grad_slots]
    if not all(id(t) in override for t in input_tensors):
        return _mapper_fail(saved, "incomplete-overrides")

    for slot in grad_slots:
        grad_tensor = override.get(id(slot.tensor))
        if grad_tensor is None:
            grad_tensor = torch.zeros_like(slot.tensor)
        grad_list[6 + slot.arg_index] = grad_tensor.to(
            device=slot.tensor.device, dtype=slot.tensor.dtype
        )
    return tuple(grad_list)


__all__ = [
    "_mapper_fail",
    "_prepare_mapping_inputs",
    "_map_triton_grads_to_slots_python",
]
