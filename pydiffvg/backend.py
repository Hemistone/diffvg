"""Backend selection and configuration for pydiffvg."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from .backends.registry import RenderAPI, get_api, list_backends

@dataclass(frozen=True)
class BezierGsplatConfig:
    samples_per_segment: int = 64
    block_h: int = 16
    block_w: int = 16
    min_scale: float = 1e-3
    depth_mode: str = "scene_order"
    detach_geometry: bool = True


BackendConfig = BezierGsplatConfig


def _normalize_backend_name(name: Optional[str]) -> str:
    key = (name or "bezier_gsplat").strip().lower()
    if key in ("default", "bezier", "openstroke", "bezier-gsplat"):
        return "bezier_gsplat"
    if key == "bezier_gsplat":
        return key
    return key


_BACKEND: str = _normalize_backend_name(os.environ.get("DIFFVG_BACKEND", "bezier_gsplat"))
if _BACKEND not in list_backends():
    _BACKEND = "bezier_gsplat"


def set_backend(name: str) -> None:
    global _BACKEND
    key = _normalize_backend_name(name)
    if key not in list_backends():
        expected = ", ".join(f"'{item}'" for item in list_backends())
        raise ValueError(f"backend must be one of {expected}")
    _BACKEND = key


def get_backend() -> str:
    return _BACKEND


def current_api() -> RenderAPI:
    return get_api(_BACKEND)


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None or v.strip() == "":
        return default
    try:
        return int(v)
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    if v is None or v.strip() == "":
        return default
    try:
        return float(v)
    except Exception:
        return default


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None or v.strip() == "":
        return default
    return v.strip().lower() not in ("0", "false", "no", "off")


def _env_str(name: str, default: str) -> str:
    v = os.environ.get(name)
    if v is None or v.strip() == "":
        return default
    return v.strip()


def get_backend_config(backend: Optional[str] = None) -> BackendConfig:
    name = _normalize_backend_name(backend or _BACKEND)
    if name == "bezier_gsplat":
        base = BezierGsplatConfig()
        depth_mode = _env_str("DIFFVG_BEZIER_GSPLAT_DEPTH_MODE", base.depth_mode).lower()
        if depth_mode != "scene_order":
            depth_mode = base.depth_mode
        return BezierGsplatConfig(
            samples_per_segment=_env_int(
                "DIFFVG_BEZIER_GSPLAT_SAMPLES_PER_SEGMENT",
                _env_int("DIFFVG_BEZIER_GSPLAT_MAX_SAMPLES_PER_SEGMENT", base.samples_per_segment),
            ),
            block_h=_env_int("DIFFVG_BEZIER_GSPLAT_BLOCK_H", base.block_h),
            block_w=_env_int("DIFFVG_BEZIER_GSPLAT_BLOCK_W", base.block_w),
            min_scale=_env_float("DIFFVG_BEZIER_GSPLAT_MIN_SCALE", base.min_scale),
            depth_mode=depth_mode,
            detach_geometry=_env_bool("DIFFVG_BEZIER_GSPLAT_DETACH_GEOMETRY", base.detach_geometry),
        )
    raise ValueError(f"Unknown backend '{name}'")


__all__ = [
    "set_backend",
    "get_backend",
    "current_api",
    "list_backends",
    "BezierGsplatConfig",
    "BackendConfig",
    "get_backend_config",
]
