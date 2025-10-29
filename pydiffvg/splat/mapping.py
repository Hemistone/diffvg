from __future__ import annotations

import math
import os
from typing import List, Optional, Tuple

import torch

from .trace import debug_enabled as _debug_enabled, trace as _trace


def _strict_fused_required() -> bool:
    mode = os.environ.get("DIFFVG_SPLAT_IMPL", "").strip().lower()
    return mode == "triton"


def _mapper_fail(saved: dict, reason: str) -> None:
    if isinstance(saved, dict) and "_fused_mapper_error" not in saved:
        saved["_fused_mapper_error"] = reason
    return None


def build_splat_mapping_payload(
    batches,
    stroke_count: int,
    cc_meta: List[List[int]],
    mu: torch.Tensor,
    spec_counts: List[int],
    points_refs: List[Optional[torch.Tensor]],
) -> dict[str, object]:
    """Return per-splat metadata consumed by the fused VJP mapper."""

    total = int(mu.shape[0])
    device_map = mu.device
    dtype_map = mu.dtype

    sample_spec_id = torch.empty(total, device=device_map, dtype=torch.int32)
    sample_seg_idx = torch.full((total,), -1, device=device_map, dtype=torch.int32)
    sample_t = torch.zeros(total, device=device_map, dtype=dtype_map)
    sample_is_stroke = torch.zeros(total, device=device_map, dtype=torch.uint8)

    offset = 0
    for spec_idx, batch in enumerate(batches):
        count = batch.mu.shape[0]
        if count == 0:
            continue
        end = offset + count
        sample_spec_id[offset:end] = spec_idx
        if batch.seg_idx is not None:
            seg_idx = batch.seg_idx.detach().to(device=device_map, dtype=torch.int32)
            sample_seg_idx[offset:end] = seg_idx
        if batch.t is not None:
            tvals = batch.t.detach().to(device=device_map, dtype=dtype_map)
            sample_t[offset:end] = tvals
        if spec_idx < stroke_count:
            sample_is_stroke[offset:end] = 1
        offset = end

    if spec_counts:
        counts_tensor = torch.tensor(spec_counts, device=device_map, dtype=torch.int32)
        spec_offsets = torch.zeros((len(spec_counts) + 1,), device=device_map, dtype=torch.int32)
        spec_offsets[1:] = torch.cumsum(counts_tensor, dim=0)
    else:
        spec_offsets = torch.zeros((1,), device=device_map, dtype=torch.int32)

    # Points offsets (flattened pool) for all specs
    points_offsets_list = [0]
    for idx in range(len(spec_counts)):
        pref = points_refs[idx] if idx < len(points_refs) else None
        count = int(pref.shape[0]) if isinstance(pref, torch.Tensor) else 0
        points_offsets_list.append(points_offsets_list[-1] + count)
    points_offsets = torch.tensor(points_offsets_list, device=device_map, dtype=torch.int32)

    segment_degrees: List[torch.Tensor] = []
    segment_start_idx: List[torch.Tensor] = []
    segment_end_idx: List[torch.Tensor] = []
    for cc in cc_meta:
        if cc:
            deg = torch.tensor(cc, device=device_map, dtype=torch.int32)
            csum = torch.cumsum(deg, dim=0)
            offsets = torch.cat(
                [torch.zeros((1,), device=device_map, dtype=torch.int32), csum[:-1]],
                dim=0,
            )
            si0 = torch.arange(deg.numel(), device=device_map, dtype=torch.int32) + offsets
            si_end = torch.arange(1, deg.numel() + 1, device=device_map, dtype=torch.int32) + csum
        else:
            deg = torch.empty(0, device=device_map, dtype=torch.int32)
            si0 = torch.empty(0, device=device_map, dtype=torch.int32)
            si_end = torch.empty(0, device=device_map, dtype=torch.int32)
        segment_degrees.append(deg)
        segment_start_idx.append(si0)
        segment_end_idx.append(si_end)

    sample_degrees = torch.empty(total, device=device_map, dtype=torch.int32)
    sample_point_base = torch.empty(total, device=device_map, dtype=torch.int32)
    sample_point_end = torch.empty(total, device=device_map, dtype=torch.int32)
    offset = 0
    for spec_idx, batch in enumerate(batches):
        count = batch.mu.shape[0]
        if count == 0:
            continue
        end = offset + count
        seg_idx = sample_seg_idx[offset:end].to(torch.int64)
        deg_tensor = segment_degrees[spec_idx]
        si0 = segment_start_idx[spec_idx]
        si_end = segment_end_idx[spec_idx]
        base = si0[seg_idx]
        end_idx = si_end[seg_idx]
        sample_degrees[offset:end] = deg_tensor[seg_idx]
        point_offset = points_offsets[spec_idx]
        sample_point_base[offset:end] = base + point_offset
        sample_point_end[offset:end] = end_idx + point_offset
        offset = end

    spec_is_stroke = torch.zeros((len(spec_counts),), device=device_map, dtype=torch.uint8)
    if stroke_count > 0:
        spec_is_stroke[:stroke_count] = 1

    return {
        "sample_spec_id": sample_spec_id,
        "sample_seg_idx": sample_seg_idx,
        "sample_t": sample_t,
        "sample_is_stroke": sample_is_stroke,
        "spec_offsets": spec_offsets,
        "spec_is_stroke": spec_is_stroke,
        "segment_degrees": segment_degrees,
        "segment_start_idx": segment_start_idx,
        "segment_end_idx": segment_end_idx,
        "points_offsets": points_offsets,
        "sample_degrees": sample_degrees,
        "sample_point_base": sample_point_base,
        "sample_point_end": sample_point_end,
    }


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
                    tt = tvals[m2_t]
                    oo = omt[m2_t]
                    oo2 = oo * oo
                    t2 = tt * tt
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


def _map_triton_grads_to_slots_gpu(
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
    if isinstance(saved, dict):
        saved.pop("_fused_mapper_error", None)

    if not grad_slots:
        return None

    prepared = _prepare_mapping_inputs(
        saved, request, dcolor, dalpha, dmu_x, dmu_y, dtheta, dsx, dsy
    )
    if prepared is None:
        return _mapper_fail(saved, "missing-prep")

    spec_counts: List[int] = prepared["spec_counts"]
    if not spec_counts:
        return _mapper_fail(saved, "empty-spec-counts")
    num_specs = len(spec_counts)

    color_refs = prepared["color_refs"]
    width_refs = prepared["width_refs"]
    points_refs = prepared["points_refs"]
    if not isinstance(color_refs, list) or not isinstance(width_refs, list) or not isinstance(points_refs, list):
        return _mapper_fail(saved, "invalid-ref-lists")

    mapping_keys = (
        "sample_spec_id",
        "sample_t",
        "sample_degrees",
        "sample_point_base",
        "sample_point_end",
        "points_offsets",
    )
    mapping_tensors = {}
    for key in mapping_keys:
        tensor = saved.get(key)
        if not isinstance(tensor, torch.Tensor):
            return _mapper_fail(saved, f"missing-{key}")
        mapping_tensors[key] = tensor

    sample_spec_id = mapping_tensors["sample_spec_id"]
    sample_t = mapping_tensors["sample_t"]
    sample_degrees = mapping_tensors["sample_degrees"]
    sample_point_base = mapping_tensors["sample_point_base"]
    sample_point_end = mapping_tensors["sample_point_end"]
    points_offsets = mapping_tensors["points_offsets"]
    spec_is_stroke = saved.get("spec_is_stroke")

    total_samples = int(prepared["dcolor_g"].shape[0])
    if sample_spec_id.shape[0] != total_samples:
        return _mapper_fail(saved, "sample-id-mismatch")
    if sample_degrees.shape[0] != total_samples:
        return _mapper_fail(saved, "sample-degree-mismatch")

    device = sample_spec_id.device
    dtype = prepared["mu_g"].dtype

    spec_offsets = prepared["spec_offsets"]
    if not isinstance(spec_offsets, torch.Tensor) or spec_offsets.shape[0] != num_specs + 1:
        return _mapper_fail(saved, "invalid-spec-offsets")
    spec_offsets = spec_offsets.to(torch.int64)

    sample_spec_id_i64 = sample_spec_id.to(torch.int64)

    color_accum = prepared["dcolor_g"].new_zeros((num_specs, 3))
    alpha_accum = prepared["dalpha_g"].new_zeros((num_specs,))
    width_accum = prepared["dsy_g"].new_zeros((num_specs,))

    used_triton_reduce = False
    if sample_spec_id.is_cuda:
        try:
            from .. import triton_splat as _triton

            if _triton.is_available():
                color_accum, alpha_accum, width_accum = _triton.fused_spec_reduce_triton(
                    sample_spec_id,
                    prepared["dcolor_g"],
                    prepared["dalpha_g"],
                    prepared["dsy_g"],
                    spec_is_stroke if isinstance(spec_is_stroke, torch.Tensor) else torch.zeros(
                        (num_specs,), device=sample_spec_id.device, dtype=torch.int32
                    ),
                    prepared["width_scale"],
                )
                used_triton_reduce = True
        except Exception as exc:
            if _debug_enabled():
                _trace(f"fused spec reduce fallback: {exc}")

    if not used_triton_reduce:
        color_accum.index_add_(0, sample_spec_id_i64, prepared["dcolor_g"])
        alpha_accum.index_add_(0, sample_spec_id_i64, prepared["dalpha_g"])
        width_accum.index_add_(0, sample_spec_id_i64, prepared["dsy_g"])
        width_accum = width_accum * prepared["width_scale"]
        if isinstance(spec_is_stroke, torch.Tensor) and spec_is_stroke.shape[0] == num_specs:
            width_accum = width_accum * spec_is_stroke.to(
                device=width_accum.device, dtype=width_accum.dtype
            )

    gmu_total = torch.stack([prepared["dmu_x_g"], prepared["dmu_y_g"]], dim=-1).clone()

    mu_g = prepared["mu_g"]
    sigma_x_g = prepared["sigma_x_g"]
    dtheta_g = prepared["dtheta_g"]
    dsx_g = prepared["dsx_g"]
    sample_t = sample_t.to(dtype)
    sample_point_base = sample_point_base.to(torch.int64)
    sample_point_end = sample_point_end.to(torch.int64)
    sample_degrees = sample_degrees.to(torch.int64)

    rho = prepared["rho"]
    rho_inv = 1.0 / rho if rho > 0 else 0.0

    total_points = int(points_offsets[-1].item()) if points_offsets.numel() > 0 else 0
    points_grad_pool = (
        prepared["mu_g"].new_zeros((total_points, 2)) if total_points > 0 else None
    )

    position_kernel_used = False
    if points_grad_pool is not None and sample_spec_id.is_cuda:
        try:
            from .. import triton_splat as _triton

            if _triton.is_available():
                points_grad_pool.zero_()
                _triton.fused_points_scatter_triton(
                    gmu_total,
                    sample_degrees,
                    sample_point_base,
                    sample_point_end,
                    sample_t,
                    points_grad_pool,
                )
                position_kernel_used = True
        except Exception as exc:
            if _debug_enabled():
                _trace(f"fused point scatter fallback: {exc}")
            points_grad_pool.zero_()

    if points_grad_pool is not None and not position_kernel_used:
        points_grad_pool.zero_()

    override: dict[int, torch.Tensor] = {}

    for si in range(num_specs):
        start = int(spec_offsets[si].item())
        end = int(spec_offsets[si + 1].item())
        if end <= start:
            continue

        mu_slice = mu_g[start:end]
        gmu_slice = gmu_total[start:end]
        dsx_slice = dsx_g[start:end]
        sx_slice = sigma_x_g[start:end]

        count = end - start
        if count >= 2:
            diffs = mu_slice[1:] - mu_slice[:-1]
            den = torch.linalg.norm(diffs, dim=-1, keepdim=True).clamp_min(1e-6)
            unit = diffs / den
            clamp_mask = sx_slice <= (1e-3 + 1e-12)
            dsx_scaled = torch.where(clamp_mask, torch.zeros_like(dsx_slice), dsx_slice * rho_inv)
            g_to_dist = 0.5 * dsx_scaled[:-1] + 0.5 * dsx_scaled[1:]
            g_to_dist[0] = g_to_dist[0] + 0.5 * dsx_scaled[0]
            g_to_dist[-1] = g_to_dist[-1] + 0.5 * dsx_scaled[-1]
            gmu_slice[:-1] += -unit * g_to_dist.unsqueeze(-1)
            gmu_slice[1:] += unit * g_to_dist.unsqueeze(-1)

    total_samples = sample_spec_id.shape[0]
    valid_mask = sample_point_base >= 0

    points_data = None
    if points_grad_pool is not None:
        points_data = prepared["mu_g"].new_zeros(points_grad_pool.shape)
        for si in range(num_specs):
            pref = points_refs[si] if si < len(points_refs) else None
            if isinstance(pref, torch.Tensor):
                start_pt = int(points_offsets[si].item())
                end_pt = start_pt + pref.shape[0]
                points_data[start_pt:end_pt] = pref.detach().to(device=device, dtype=dtype)

    indices = torch.arange(total_samples, device=device, dtype=torch.int64)
    spec_offsets_i64 = spec_offsets.to(torch.int64)
    spec_start = spec_offsets_i64[sample_spec_id_i64]
    spec_end = spec_offsets_i64[sample_spec_id_i64 + 1] - 1
    has_prev = indices > spec_start
    has_next = indices < spec_end
    central_mask = has_prev & has_next
    left_idx = indices - 1
    right_idx = indices + 1
    good_mask = torch.zeros((total_samples,), dtype=torch.bool, device=device)
    if central_mask.any():
        left = mu_g[left_idx.clamp(0, total_samples - 1)]
        right = mu_g[right_idx.clamp(0, total_samples - 1)]
        cvec = right - left
        cn2 = (cvec[:, 0] * cvec[:, 0] + cvec[:, 1] * cvec[:, 1])
        good_mask = central_mask & (cn2 > 1e-8)
        if torch.any(good_mask):
            gtheta_good = dtheta_g[good_mask]
            inv_cn2 = 1.0 / cn2[good_mask]
            cvec_good = cvec[good_mask]
            gvec = torch.stack(
                [-cvec_good[:, 1] * inv_cn2, cvec_good[:, 0] * inv_cn2],
                dim=-1,
            )
            gvec = gvec * gtheta_good.unsqueeze(-1)
            gmu_total.index_add_(0, right_idx[good_mask], gvec)
            gmu_total.index_add_(0, left_idx[good_mask], -gvec)

    position_kernel_used = False
    if points_grad_pool is not None and sample_spec_id.is_cuda:
        try:
            from .. import triton_splat as _triton

            if _triton.is_available():
                points_grad_pool.zero_()
                _triton.fused_points_scatter_triton(
                    gmu_total,
                    sample_degrees,
                    sample_point_base,
                    sample_point_end,
                    sample_t,
                    points_grad_pool,
                )
                position_kernel_used = True
        except Exception as exc:
            if _debug_enabled():
                _trace(f"fused point scatter fallback: {exc}")
            if points_grad_pool is not None:
                points_grad_pool.zero_()

    if points_grad_pool is not None and not position_kernel_used:
        points_grad_pool.zero_()
        deg0 = (sample_degrees == 0) & valid_mask
        if deg0.any():
            si0_idx = sample_point_base[deg0]
            si_end_idx = sample_point_end[deg0]
            g = gmu_total[deg0]
            w0 = (1.0 - sample_t[deg0]).unsqueeze(-1)
            w1 = sample_t[deg0].unsqueeze(-1)
            points_grad_pool.index_add_(0, si0_idx, g * w0)
            points_grad_pool.index_add_(0, si_end_idx, g * w1)
        deg1 = (sample_degrees == 1) & valid_mask
        if deg1.any():
            base_idx = sample_point_base[deg1]
            ci0_idx = base_idx + 1
            si_end_idx = sample_point_end[deg1]
            g = gmu_total[deg1]
            tt = sample_t[deg1]
            oo = 1.0 - tt
            points_grad_pool.index_add_(0, base_idx, g * (oo * oo).unsqueeze(-1))
            points_grad_pool.index_add_(0, ci0_idx, g * (2.0 * oo * tt).unsqueeze(-1))
            points_grad_pool.index_add_(0, si_end_idx, g * (tt * tt).unsqueeze(-1))
        deg2 = (sample_degrees >= 2) & valid_mask
        if deg2.any():
            base_idx = sample_point_base[deg2]
            ci0_idx = base_idx + 1
            ci1_idx = base_idx + 2
            si_end_idx = sample_point_end[deg2]
            g = gmu_total[deg2]
            tt = sample_t[deg2]
            oo = 1.0 - tt
            oo2 = oo * oo
            t2 = tt * tt
            points_grad_pool.index_add_(0, base_idx, g * (oo2 * oo).unsqueeze(-1))
            points_grad_pool.index_add_(0, ci0_idx, g * (3.0 * oo2 * tt).unsqueeze(-1))
            points_grad_pool.index_add_(0, ci1_idx, g * (3.0 * oo * t2).unsqueeze(-1))
            points_grad_pool.index_add_(0, si_end_idx, g * (t2 * tt).unsqueeze(-1))

    if points_grad_pool is not None and points_data is not None:
        tan_mask = (~good_mask) & valid_mask & (dtheta_g != 0)
        if tan_mask.any():
            deg = sample_degrees
            tvals = sample_t
            omt_vals = 1.0 - tvals

            mask0 = tan_mask & (deg == 0)
            if mask0.any():
                base_idx = sample_point_base[mask0]
                end_idx = sample_point_end[mask0]
                v = points_data[end_idx] - points_data[base_idx]
                denom = (v[:, 0] * v[:, 0] + v[:, 1] * v[:, 1]).clamp_min(1e-6)
                gv = torch.stack(
                    [-v[:, 1] / denom, v[:, 0] / denom], dim=-1
                ) * dtheta_g[mask0].unsqueeze(-1)
                points_grad_pool.index_add_(0, base_idx, -gv)
                points_grad_pool.index_add_(0, end_idx, gv)

            mask1 = tan_mask & (deg == 1)
            if mask1.any():
                base_idx = sample_point_base[mask1]
                ci0_idx = base_idx + 1
                end_idx = sample_point_end[mask1]
                tt = tvals[mask1]
                oo = omt_vals[mask1]
                p0 = points_data[base_idx]
                pc = points_data[ci0_idx]
                p1 = points_data[end_idx]
                v = 2.0 * (oo.unsqueeze(-1) * (pc - p0) + tt.unsqueeze(-1) * (p1 - pc))
                denom = (v[:, 0] * v[:, 0] + v[:, 1] * v[:, 1]).clamp_min(1e-6)
                gv = torch.stack(
                    [-v[:, 1] / denom, v[:, 0] / denom], dim=-1
                ) * dtheta_g[mask1].unsqueeze(-1)
                dt0 = -2.0 * oo
                dt1 = 2.0 * (1.0 - 2.0 * tt)
                dt2 = 2.0 * tt
                points_grad_pool.index_add_(0, base_idx, gv * dt0.unsqueeze(-1))
                points_grad_pool.index_add_(0, ci0_idx, gv * dt1.unsqueeze(-1))
                points_grad_pool.index_add_(0, end_idx, gv * dt2.unsqueeze(-1))

            mask2 = tan_mask & (deg >= 2)
            if mask2.any():
                base_idx = sample_point_base[mask2]
                ci0_idx = base_idx + 1
                ci1_idx = base_idx + 2
                end_idx = sample_point_end[mask2]
                tt = tvals[mask2]
                oo = omt_vals[mask2]
                oo2 = oo * oo
                t2 = tt * tt
                p0 = points_data[base_idx]
                p1 = points_data[ci0_idx]
                p2 = points_data[ci1_idx]
                p3 = points_data[end_idx]
                v = (
                    3.0 * oo2.unsqueeze(-1) * (p1 - p0)
                    + 6.0 * (oo * tt).unsqueeze(-1) * (p2 - p1)
                    + 3.0 * t2.unsqueeze(-1) * (p3 - p2)
                )
                denom = (v[:, 0] * v[:, 0] + v[:, 1] * v[:, 1]).clamp_min(1e-6)
                gv = torch.stack(
                    [-v[:, 1] / denom, v[:, 0] / denom], dim=-1
                ) * dtheta_g[mask2].unsqueeze(-1)
                dt0 = -3.0 * oo2
                dt1 = 3.0 * oo2 - 6.0 * oo * tt
                dt2 = 6.0 * oo * tt - 3.0 * t2
                dt3 = 3.0 * t2
                points_grad_pool.index_add_(0, base_idx, gv * dt0.unsqueeze(-1))
                points_grad_pool.index_add_(0, ci0_idx, gv * dt1.unsqueeze(-1))
                points_grad_pool.index_add_(0, ci1_idx, gv * dt2.unsqueeze(-1))
                points_grad_pool.index_add_(0, end_idx, gv * dt3.unsqueeze(-1))

    color_refs_list: List[Optional[torch.Tensor]] = color_refs
    width_refs_list: List[Optional[torch.Tensor]] = width_refs

    for si in range(num_specs):
        cref = color_refs_list[si] if si < len(color_refs_list) else None
        if isinstance(cref, torch.Tensor):
            grad_rgba = torch.cat(
                [color_accum[si], alpha_accum[si : si + 1]], dim=0
            )
            grad_rgba = grad_rgba.to(dtype=cref.dtype, device=cref.device).view_as(cref).detach()
            override[id(cref)] = grad_rgba

        wref = width_refs_list[si] if si < len(width_refs_list) else None
        if isinstance(wref, torch.Tensor):
            grad_width = width_accum[si].to(dtype=wref.dtype, device=wref.device).expand_as(wref).detach()
            override[id(wref)] = grad_width

        pref = points_refs[si] if si < len(points_refs) else None
        if isinstance(pref, torch.Tensor):
            start_pt = int(points_offsets[si].item())
            end_pt = start_pt + pref.shape[0]
            if points_grad_pool is None:
                override[id(pref)] = torch.zeros_like(pref)
            else:
                grad_points = points_grad_pool[start_pt:end_pt]
                override[id(pref)] = grad_points.to(dtype=pref.dtype, device=pref.device).detach()

    input_tensors = [slot.tensor for slot in grad_slots]
    if not all(id(t) in override for t in input_tensors):
        return _mapper_fail(saved, "incomplete-overrides")

    total_args = len(args_with_grad)
    grad_list: List[Optional[torch.Tensor]] = [None] * (6 + total_args)
    for slot in grad_slots:
        grad_tensor = override.get(id(slot.tensor))
        if grad_tensor is None:
            grad_tensor = torch.zeros_like(slot.tensor)
        grad_list[6 + slot.arg_index] = grad_tensor.to(
            device=slot.tensor.device, dtype=slot.tensor.dtype
        )
    return tuple(grad_list)


def map_triton_grads_to_slots(
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
    mapped = _map_triton_grads_to_slots_gpu(
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
        if isinstance(saved, dict):
            saved.pop("_fused_mapper_error", None)
        return mapped
    reason = None
    if isinstance(saved, dict):
        reason = saved.pop("_fused_mapper_error", None)
    if reason and _debug_enabled():
        _trace(f"fused VJP mapper falling back to python path (reason={reason})")
    if _strict_fused_required() and grad_slots:
        raise RuntimeError(
            f"DIFFVG_SPLAT_IMPL=triton requires the fused mapper (reason={reason or 'unknown'})"
        )
    mapped_python = _map_triton_grads_to_slots_python(
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
    if mapped_python is not None:
        return mapped_python
    if reason and isinstance(saved, dict):
        saved["_fused_mapper_error"] = reason
    return None


__all__ = ["build_splat_mapping_payload", "map_triton_grads_to_slots"]
