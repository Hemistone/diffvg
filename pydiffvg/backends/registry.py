"""Backend registry for pydiffvg.

The maintained product path is the compiled open-stroke `bezier_gsplat`
backend. Legacy backends remain opt-in via DIFFVG_ENABLE_LEGACY=1.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class RenderAPI:
    serialize_scene: Callable[..., tuple]
    apply: Callable[..., object]
    render_grad: Callable[..., tuple]
    prefer_device_serialization: bool = False


def _legacy_disabled_api(name: str) -> RenderAPI:
    def _fail(*args, **kwargs):
        raise RuntimeError(
            f"The '{name}' backend is now legacy-only. "
            "Use 'bezier_gsplat' for the maintained stroke-first engine, or set "
            "DIFFVG_ENABLE_LEGACY=1 to re-enable legacy backends for comparison."
        )

    return RenderAPI(
        serialize_scene=_fail,
        apply=_fail,
        render_grad=_fail,
        prefer_device_serialization=False,
    )


def _legacy_enabled() -> bool:
    return os.environ.get("DIFFVG_ENABLE_LEGACY", "").strip().lower() not in ("", "0", "false", "no", "off")


def _baseline_api() -> RenderAPI:
    if not _legacy_enabled():
        return _legacy_disabled_api("baseline")
    from ..render_pytorch import BaselineRenderFunction as RF

    return RenderAPI(
        serialize_scene=RF.serialize_scene,
        apply=RF.apply,
        render_grad=RF.render_grad,
        prefer_device_serialization=False,
    )


def _splat_api() -> RenderAPI:
    if not _legacy_enabled():
        return _legacy_disabled_api("splat")
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
    key = (name or "bezier_gsplat").strip().lower()
    if key in ("default", "bezier", "openstroke", "bezier_gsplat", "bezier-gsplat"):
        return _bezier_gsplat_api()
    if key == "baseline":
        return _baseline_api()
    if key == "splat":
        return _splat_api()
    raise ValueError(
        f"Unknown backend '{name}'. Expected 'bezier_gsplat', 'baseline', or 'splat'."
    )


def list_backends() -> tuple[str, ...]:
    return ("bezier_gsplat", "baseline", "splat")


__all__ = ["RenderAPI", "get_api", "list_backends"]
