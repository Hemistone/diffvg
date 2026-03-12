#!/usr/bin/env python3
"""Small smoke test for the maintained stroke-first runtime.

Checks three invariants only:
- frontend path tensors are rebound to the compiled point bank
- bezier_gsplat renders a finite RGBA image for a supported open-stroke scene
- save_svg/svg_to_scene round-trip preserves a supported stroke-only scene
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

import pydiffvg


def _make_path(device: torch.device, offset_x: float, color: tuple[float, float, float, float]):
    points = torch.tensor(
        [
            [20.0 + offset_x, 40.0],
            [55.0 + offset_x, 20.0],
            [90.0 + offset_x, 60.0],
            [130.0 + offset_x, 40.0],
            [155.0 + offset_x, 30.0],
            [180.0 + offset_x, 80.0],
            [215.0 + offset_x, 55.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    path = pydiffvg.Path(
        num_control_points=torch.tensor([2, 2], dtype=torch.int32, device=device),
        points=points,
        stroke_width=torch.tensor(1.25, dtype=torch.float32, device=device),
        is_closed=False,
    )
    group = pydiffvg.ShapeGroup(
        shape_ids=torch.tensor([0], dtype=torch.int32, device=device),
        fill_color=None,
        stroke_color=torch.tensor(color, dtype=torch.float32, device=device),
    )
    return path, group


def main() -> None:
    pydiffvg.set_backend("bezier_gsplat")
    device = pydiffvg.get_device()
    if device.type != "cuda":
        print("skip: stroke-first smoke requires CUDA for bezier_gsplat")
        return

    width = 256
    height = 192
    renderer = pydiffvg.Renderer(backend="bezier_gsplat")

    path0, group0 = _make_path(device, 0.0, (0.15, 0.20, 0.90, 1.0))
    path1, group1 = _make_path(device, 12.0, (0.90, 0.20, 0.15, 1.0))
    group0.shape_ids = torch.tensor([0], dtype=torch.int32, device=device)
    group1.shape_ids = torch.tensor([1], dtype=torch.int32, device=device)
    shapes = [path0, path1]
    shape_groups = [group0, group1]

    scene_args = renderer.serialize_scene(
        width,
        height,
        shapes,
        shape_groups,
        device=device,
        cache_key="smoke",
        invalidate_cache=True,
    )
    compiled = scene_args[0]
    assert compiled.stroke_count == 2, compiled.stroke_count

    before = path0.points.detach().clone()
    with torch.no_grad():
        compiled.point_bank[0, 0] += 3.0
    assert torch.isclose(path0.points[0, 0], before[0, 0] + 3.0), "frontend points are not bound to point_bank"

    image = renderer.apply(width, height, 2, 2, 0, None, *scene_args)
    assert tuple(image.shape) == (height, width, 4), tuple(image.shape)
    assert torch.isfinite(image).all().item(), "render produced non-finite values"

    with tempfile.TemporaryDirectory(prefix="stroke-first-smoke-") as tmpdir:
        svg_path = Path(tmpdir) / "scene.svg"
        pydiffvg.save_svg(
            str(svg_path),
            width,
            height,
            shapes,
            shape_groups,
            use_gamma=False,
            background_rgb=(1.0, 1.0, 1.0),
        )
        assert svg_path.is_file(), "save_svg did not create output"
        rt_width, rt_height, rt_shapes, rt_groups = pydiffvg.svg_to_scene(str(svg_path))
        assert (rt_width, rt_height) == (width, height)
        assert len(rt_shapes) == 2
        assert len(rt_groups) == 2
        for group in rt_groups:
            assert getattr(group, "fill_color", None) is None
            stroke = getattr(group, "stroke_color", None)
            assert isinstance(stroke, torch.Tensor) and stroke.numel() == 4
        rt_args = renderer.serialize_scene(
            rt_width,
            rt_height,
            rt_shapes,
            rt_groups,
            device=device,
            cache_key="roundtrip",
            invalidate_cache=True,
        )
        rt_image = renderer.apply(rt_width, rt_height, 2, 2, 0, None, *rt_args)
        assert tuple(rt_image.shape) == (rt_height, rt_width, 4)
        assert torch.isfinite(rt_image).all().item(), "round-trip render produced non-finite values"

    print("stroke-first runtime smoke: ok")


if __name__ == "__main__":
    main()
