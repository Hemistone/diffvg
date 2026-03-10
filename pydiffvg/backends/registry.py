"""Backend registry for pydiffvg (internal).

Provides a tiny indirection so the library can switch between the original
diffvg baseline renderer and alternative differentiable rasterizers without
changing the public API surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class RenderAPI:
    """Simple function bundle for a render backend.

    - serialize_scene: builds the argument tuple for RenderFunction
    - apply: forward pass (typically RenderFunction.apply)
    - render_grad: backward helper (typically RenderFunction.render_grad)
    """

    serialize_scene: Callable[..., tuple]
    apply: Callable[..., object]
    render_grad: Callable[..., tuple]
    prefer_device_serialization: bool = False


def _baseline_api() -> RenderAPI:
    # Import locally to avoid import cycles at package import time
    from ..render_pytorch import BaselineRenderFunction as RF

    return RenderAPI(
        serialize_scene=RF.serialize_scene,
        apply=RF.apply,
        render_grad=RF.render_grad,
        prefer_device_serialization=False,
    )


def _splat_api() -> RenderAPI:
    from .. import render_splat as _splat

    return RenderAPI(
        serialize_scene=_splat.serialize_scene,
        apply=_splat.apply,
        render_grad=_splat.render_grad,
        prefer_device_serialization=True,
    )


def _bezier_gsplat_api() -> RenderAPI:
    from .. import render_bezier_gsplat as _bezier_gsplat

    return RenderAPI(
        serialize_scene=_bezier_gsplat.serialize_scene,
        apply=_bezier_gsplat.apply,
        render_grad=_bezier_gsplat.render_grad,
        prefer_device_serialization=True,
    )


def get_api(name: Optional[str] = None) -> RenderAPI:
    """Return the RenderAPI for the given backend name.

    Known names: "baseline", "splat", "bezier_gsplat" (case-insensitive).
    """
    key = (name or "baseline").strip().lower()
    if key in ("baseline", "default"):
        return _baseline_api()
    if key == "splat":
        return _splat_api()
    if key in ("bezier_gsplat", "bezier-gsplat"):
        return _bezier_gsplat_api()
    raise ValueError(
        f"Unknown backend '{name}'. Expected 'baseline', 'splat', or 'bezier_gsplat'."
    )


def list_backends() -> tuple[str, ...]:
    return ("baseline", "splat", "bezier_gsplat")


__all__ = ["RenderAPI", "get_api", "list_backends"]
