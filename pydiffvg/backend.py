"""Backend selection and configuration for pydiffvg.

- set_backend/get_backend: choose between 'baseline' and 'splat'.
- get_backend_config: return config relevant to the current backend.

Environment variables:
- DIFFVG_BACKEND: select backend ('baseline' or 'splat')
- DIFFVG_SPLAT_K, DIFFVG_SPLAT_R, DIFFVG_SPLAT_RHO
- DIFFVG_SPLAT_TILE: tile size (int)
- DIFFVG_DEPTH_POLICY: 'none' or 'small_first' (ignored by baseline)
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


_BACKEND: str = (os.environ.get("DIFFVG_BACKEND", "baseline").strip() or "baseline").lower()
if _BACKEND not in ("baseline", "splat"):
    _BACKEND = "baseline"


def set_backend(name: str) -> None:
    global _BACKEND
    key = (name or "").strip().lower()
    if key not in ("baseline", "splat"):
        raise ValueError("backend must be 'baseline' or 'splat'")
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


def _env_depth_policy(default: DepthPolicy) -> DepthPolicy:
    v = (os.environ.get("DIFFVG_DEPTH_POLICY", "").strip() or default.value).lower()
    if v in (DepthPolicy.none.value, DepthPolicy.small_first.value):
        return DepthPolicy(v)
    return default


def get_backend_config(backend: Optional[str] = None) -> Optional[SplatConfig]:
    """Return configuration for the given backend (or current backend).

    For 'baseline' this returns None. For 'splat' returns SplatConfig with
    env overrides applied.
    """
    name = (backend or _BACKEND).lower()
    if name != "splat":
        return None
    base = SplatConfig()
    return SplatConfig(
        K=_env_int("DIFFVG_SPLAT_K", base.K),
        R=_env_int("DIFFVG_SPLAT_R", base.R),
        rho=_env_float("DIFFVG_SPLAT_RHO", base.rho),
        tile=_env_int("DIFFVG_SPLAT_TILE", base.tile),
        depth_policy=_env_depth_policy(base.depth_policy),
    )


__all__ = [
    "set_backend",
    "get_backend",
    "current_api",
    "list_backends",
    "SplatConfig",
    "DepthPolicy",
    "get_backend_config",
]

