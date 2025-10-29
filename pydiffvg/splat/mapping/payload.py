from __future__ import annotations

from typing import List, Optional

import torch


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


__all__ = ["build_splat_mapping_payload"]
