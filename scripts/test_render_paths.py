#!/usr/bin/env python3
"""Tiny smoke test for baseline renderer (CPU).

Renders a circle with constant fill to ensure the Python API and the
serialization path are functional. No gradients/backward.
"""
from __future__ import annotations

import os
import torch
import pydiffvg as d


def main() -> None:
    os.environ.setdefault("DIFFVG_BACKEND", "baseline")
    d.set_use_gpu(False)

    w, h = 128, 128
    circle = d.Circle(radius=torch.tensor(40.0), center=torch.tensor([64.0, 64.0]))
    group = d.ShapeGroup(shape_ids=torch.tensor([0]), fill_color=torch.tensor([0.1, 0.7, 0.2, 1.0]))
    args = d.RenderFunction.serialize_scene(w, h, [circle], [group])
    render = d.RenderFunction.apply
    img = render(w, h, 2, 2, 0, None, *args)
    assert torch.isfinite(img).all()
    print("ok", tuple(img.shape))


if __name__ == "__main__":
    main()

