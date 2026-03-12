"""Backend registry for pydiffvg."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class RenderAPI:
    serialize_scene: Callable[..., tuple]
    apply: Callable[..., object]
    render_grad: Callable[..., tuple]
    prefer_device_serialization: bool = False

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
    raise ValueError(f"Unknown backend '{name}'. Expected 'bezier_gsplat'.")


def list_backends() -> tuple[str, ...]:
    return ("bezier_gsplat",)


__all__ = ["RenderAPI", "get_api", "list_backends"]
