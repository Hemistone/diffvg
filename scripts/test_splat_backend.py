#!/usr/bin/env python3
"""Smoke checks for the Bézier splat backend.

The script exercises the pieces we have in place today:

1. Render an open path stroke through `pydiffvg.Renderer` with
   `DIFFVG_BACKEND=splat` and confirm the splat implementation handles it
   without falling back.
2. Render a circle to confirm unsupported shapes trigger the explicit
   fallback into the legacy baseline renderer.
"""

from __future__ import annotations

import os
from typing import Iterable

import torch
import pydiffvg as d
import pydiffvg.render_splat as splat


def _reset_fallback_cache() -> None:
    """Clear the splat fallback cache so we can assert per-test behaviour."""
    if hasattr(splat._warn_fallback, "_seen"):
        getattr(splat._warn_fallback, "_seen").clear()  # type: ignore[call-overload,union-attr]
    else:
        setattr(splat._warn_fallback, "_seen", set())


def _renderer(*, use_cuda: bool = False) -> d.Renderer:
    """Construct a Renderer routed to the splat backend on the requested device."""
    d.set_backend("splat")
    d.set_use_gpu(use_cuda and torch.cuda.is_available())
    renderer = d.Renderer()
    assert renderer._api.apply.__module__ == "pydiffvg.render_splat"  # type: ignore[attr-defined]
    return renderer


def _render(renderer: d.Renderer, args: Iterable[object]) -> torch.Tensor:
    """Helper to invoke the renderer with a deterministic seed."""
    w = h = 128
    return renderer.apply(w, h, 2, 2, 0, None, *args)


def smoke_path_stroke() -> None:
    """Run a basic forward pass for an open stroke."""
    renderer = _renderer()
    _reset_fallback_cache()

    w = h = 128
    # One linear segment: start + end point, no control points.
    num_control_points = torch.tensor([0], dtype=torch.int32)
    points = torch.tensor([[16.0, 24.0], [112.0, 96.0]], requires_grad=False)
    path = d.Path(
        num_control_points=num_control_points,
        points=points,
        is_closed=False,
        stroke_width=torch.tensor(4.0),
    )
    stroke_color = torch.tensor([0.9, 0.1, 0.2, 1.0])
    group = d.ShapeGroup(
        shape_ids=torch.tensor([0], dtype=torch.int32),
        fill_color=None,
        stroke_color=stroke_color,
    )

    args = renderer.serialize_scene(w, h, [path], [group], cache_key="stroke")
    img = _render(renderer, args)
    assert img.shape == (h, w, 4)
    assert torch.isfinite(img).all()

    # No fallback should have been triggered for a supported stroke scene.
    assert not getattr(splat._warn_fallback, "_seen", set())
    print("[splat] stroke forward: ok (splat backend active)")


def smoke_circle_fallback() -> None:
    """Ensure unsupported primitives fall back to the baseline renderer."""
    renderer = _renderer()
    _reset_fallback_cache()

    w = h = 128
    circle = d.Circle(radius=torch.tensor(32.0), center=torch.tensor([64.0, 64.0]))
    group = d.ShapeGroup(
        shape_ids=torch.tensor([0], dtype=torch.int32),
        fill_color=torch.tensor([0.2, 0.6, 0.9, 1.0]),
    )

    args = renderer.serialize_scene(w, h, [circle], [group], cache_key="circle")
    img = _render(renderer, args)
    assert img.shape == (h, w, 4)
    assert torch.isfinite(img).all()

    reasons = getattr(splat._warn_fallback, "_seen", set())
    assert "non-path shapes are not handled yet" in reasons, reasons
    print("[splat] non-path fallback: ok (baseline reached as expected)")


def main() -> None:
    os.environ.setdefault("DIFFVG_BACKEND", "splat")
    smoke_path_stroke()
    smoke_circle_fallback()
    print("[splat] splat backend smoke tests all ok")


if __name__ == "__main__":
    main()
