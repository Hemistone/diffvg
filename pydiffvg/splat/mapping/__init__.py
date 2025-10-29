from __future__ import annotations

import os
from typing import Optional, Tuple

import torch

from ..trace import debug_enabled as _debug_enabled, trace as _trace
from .core import _map_triton_grads_to_slots_gpu
from .fallback import _map_triton_grads_to_slots_python
from .payload import build_splat_mapping_payload


def _strict_fused_required() -> bool:
    mode = os.environ.get("DIFFVG_SPLAT_IMPL", "").strip().lower()
    return mode == "triton"


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
