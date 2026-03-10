"""Backend selection and configuration for pydiffvg.

- set_backend/get_backend: choose between the registered render backends.
- get_backend_config: return config relevant to the current backend.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .backends.registry import get_api, list_backends, RenderAPI


class DepthPolicy(str, Enum):
    none = "none"
    small_first = "small_first"


@dataclass(frozen=True)
class SplatConfig:
    K: int = 8          # samples per segment
    R: int = 2          # radial refinement factor
    rho: float = 1.0    # width scale for Gaussian
    tile: int = 32      # tile size for tiled blending
    depth_policy: DepthPolicy = DepthPolicy.none


@dataclass(frozen=True)
class BezierGsplatConfig:
    sample_spacing_px: float = 1.0
    max_samples_per_segment: int = 64
    block_h: int = 16
    block_w: int = 16
    min_scale: float = 1e-3
    depth_mode: str = "scene_order"


BackendConfig = SplatConfig | BezierGsplatConfig


def _normalize_backend_name(name: Optional[str]) -> str:
    key = (name or "baseline").strip().lower()
    if key in ("baseline", "default"):
        return "baseline"
    if key == "splat":
        return "splat"
    if key in ("bezier_gsplat", "bezier-gsplat"):
        return "bezier_gsplat"
    return key


_BACKEND: str = _normalize_backend_name(os.environ.get("DIFFVG_BACKEND", "baseline"))
if _BACKEND not in ("baseline", "splat", "bezier_gsplat"):
    _BACKEND = "baseline"


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


def _env_str(name: str, default: str) -> str:
    v = os.environ.get(name)
    if v is None or v.strip() == "":
        return default
    return v.strip()


def _env_depth_policy(default: DepthPolicy) -> DepthPolicy:
    v = (os.environ.get("DIFFVG_DEPTH_POLICY", "").strip() or default.value).lower()
    if v in (DepthPolicy.none.value, DepthPolicy.small_first.value):
        return DepthPolicy(v)
    return default


def get_backend_config(backend: Optional[str] = None) -> Optional[BackendConfig]:
    """Return configuration for the given backend (or current backend).

    For 'baseline' this returns None. Backend-specific configs are returned with
    env overrides applied.
    """
    name = _normalize_backend_name(backend or _BACKEND)
    if name == "baseline":
        return None
    if name == "splat":
        base = SplatConfig()
        return SplatConfig(
            K=_env_int("DIFFVG_SPLAT_K", base.K),
            R=_env_int("DIFFVG_SPLAT_R", base.R),
            rho=_env_float("DIFFVG_SPLAT_RHO", base.rho),
            tile=_env_int("DIFFVG_SPLAT_TILE", base.tile),
            depth_policy=_env_depth_policy(base.depth_policy),
        )
    if name == "bezier_gsplat":
        base = BezierGsplatConfig()
        depth_mode = _env_str("DIFFVG_BEZIER_GSPLAT_DEPTH_MODE", base.depth_mode).lower()
        if depth_mode != "scene_order":
            depth_mode = base.depth_mode
        return BezierGsplatConfig(
            sample_spacing_px=_env_float(
                "DIFFVG_BEZIER_GSPLAT_SAMPLE_SPACING_PX",
                base.sample_spacing_px,
            ),
            max_samples_per_segment=_env_int(
                "DIFFVG_BEZIER_GSPLAT_MAX_SAMPLES_PER_SEGMENT",
                base.max_samples_per_segment,
            ),
            block_h=_env_int("DIFFVG_BEZIER_GSPLAT_BLOCK_H", base.block_h),
            block_w=_env_int("DIFFVG_BEZIER_GSPLAT_BLOCK_W", base.block_w),
            min_scale=_env_float("DIFFVG_BEZIER_GSPLAT_MIN_SCALE", base.min_scale),
            depth_mode=depth_mode,
        )
    raise ValueError(f"Unknown backend '{name}'")


__all__ = [
    "set_backend",
    "get_backend",
    "current_api",
    "list_backends",
    "SplatConfig",
    "BezierGsplatConfig",
    "BackendConfig",
    "DepthPolicy",
    "get_backend_config",
]
