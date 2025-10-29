from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from ..trace import debug_enabled as _debug_enabled, trace as _trace
from .fallback import _mapper_fail, _prepare_mapping_inputs


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

    if torch.any(dtheta_g != 0):
        deg0 = (sample_degrees == 0) & valid_mask
        if deg0.any():
            si0_idx = sample_point_base[deg0]
            si_end_idx = sample_point_end[deg0]
            v = points_data[si_end_idx] - points_data[si0_idx]
            denom = (v[:, 0] * v[:, 0] + v[:, 1] * v[:, 1]).clamp_min(1e-6)
            gv = torch.stack(
                [-v[:, 1] / denom, v[:, 0] / denom],
                dim=-1,
            ) * dtheta_g[deg0].unsqueeze(-1)
            points_grad_pool.index_add_(0, si0_idx, gv * (-1.0))
            points_grad_pool.index_add_(0, si_end_idx, gv)
        deg1 = (sample_degrees == 1) & valid_mask
        if deg1.any():
            base_idx = sample_point_base[deg1]
            ci0_idx = base_idx + 1
            si_end_idx = sample_point_end[deg1]
            tt = sample_t[deg1]
            oo = 1.0 - tt
            omt_vals = oo
            t_vals = tt
            p0 = points_data[base_idx]
            p1 = points_data[ci0_idx]
            p2 = points_data[si_end_idx]
            v = (
                (2.0 * omt_vals.unsqueeze(-1) * (p1 - p0))
                + (2.0 * t_vals.unsqueeze(-1) * (p2 - p1))
            )
            denom = (v[:, 0] * v[:, 0] + v[:, 1] * v[:, 1]).clamp_min(1e-6)
            gv = torch.stack(
                [-v[:, 1] / denom, v[:, 0] / denom],
                dim=-1,
            ) * dtheta_g[deg1].unsqueeze(-1)
            dt0 = -(omt_vals * omt_vals)
            dt1 = (omt_vals * omt_vals) - (2.0 * omt_vals * t_vals)
            dt2 = 2.0 * omt_vals * t_vals
            points_grad_pool.index_add_(0, base_idx, gv * dt0.unsqueeze(-1))
            points_grad_pool.index_add_(0, ci0_idx, gv * dt1.unsqueeze(-1))
            points_grad_pool.index_add_(0, si_end_idx, gv * dt2.unsqueeze(-1))
        deg2 = (sample_degrees >= 2) & valid_mask
        if deg2.any():
            base_idx = sample_point_base[deg2]
            ci0_idx = base_idx + 1
            ci1_idx = base_idx + 2
            si_end_idx = sample_point_end[deg2]
            tt = sample_t[deg2]
            oo = 1.0 - tt
            oo2 = oo * oo
            t2 = tt * tt
            omt_vals = oo
            t_vals = tt
            p0 = points_data[base_idx]
            p1 = points_data[ci0_idx]
            p2 = points_data[ci1_idx]
            p3 = points_data[si_end_idx]
            v = (
                3.0 * oo2.unsqueeze(-1) * (p1 - p0)
                + 6.0 * (omt_vals * t_vals).unsqueeze(-1) * (p2 - p1)
                + 3.0 * t2.unsqueeze(-1) * (p3 - p2)
            )
            denom = (v[:, 0] * v[:, 0] + v[:, 1] * v[:, 1]).clamp_min(1e-6)
            gv = torch.stack(
                [-v[:, 1] / denom, v[:, 0] / denom],
                dim=-1,
            ) * dtheta_g[deg2].unsqueeze(-1)
            dt0 = -3.0 * oo2
            dt1 = 3.0 * oo2 - 6.0 * omt_vals * t_vals
            dt2 = 6.0 * omt_vals * t_vals - 3.0 * t2
            dt3 = 3.0 * t2
            points_grad_pool.index_add_(0, base_idx, gv * dt0.unsqueeze(-1))
            points_grad_pool.index_add_(0, ci0_idx, gv * dt1.unsqueeze(-1))
            points_grad_pool.index_add_(0, ci1_idx, gv * dt2.unsqueeze(-1))
            points_grad_pool.index_add_(0, si_end_idx, gv * dt3.unsqueeze(-1))

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


__all__ = ["_map_triton_grads_to_slots_gpu"]
