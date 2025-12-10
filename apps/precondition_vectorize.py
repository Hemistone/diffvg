"""Run raster preconditioning and dump debug artifacts.

This script is intentionally limited to the preconditioning stage: it produces
edge/skeleton masks and an initial render of the generated paths, but does not
run any diffvg optimization. Use painterly_rendering.py with --precondition for
full optimization.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import pydiffvg


def _render(renderer: pydiffvg.Renderer, width: int, height: int, shapes, groups, device, cache_key="main", invalidate=False):
    scene_args = renderer.serialize_scene(
        width,
        height,
        shapes,
        groups,
        device=device,
        cache_key=cache_key,
        invalidate_cache=invalidate,
    )
    # num_samples_x/y kept small to keep turnaround fast
    return renderer.apply(width, height, 2, 2, 0, None, *scene_args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Precondition raster -> diffvg paths (no optimization).")
    parser.add_argument("image", type=Path, help="Input raster image")
    parser.add_argument("--backend", default="splat", choices=["baseline", "splat"], help="Render backend to use")
    parser.add_argument("--out-dir", type=Path, default=Path("results/precondition"), help="Where to write renders/debug outputs")
    args = parser.parse_args()

    pydiffvg.set_backend(args.backend)
    device = pydiffvg.get_device()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = pydiffvg.PreconditionConfig()
    scene = pydiffvg.build_preconditioned_scene(args.image, cfg=cfg, backend=args.backend, device=device)

    # Debug outputs for the preconditioning stage
    pydiffvg.imwrite(scene.edge_mask.astype(np.float32), str(out_dir / "edge_mask.png"), gamma=1.0)
    pydiffvg.imwrite(scene.skeleton.astype(np.float32), str(out_dir / "skeleton.png"), gamma=1.0)

    init_img = _render(scene.renderer, scene.width, scene.height, scene.shapes, scene.shape_groups, device, invalidate=True)
    pydiffvg.imwrite(init_img, str(out_dir / "init_render.png"))
    print(f"Preconditioning complete: {len(scene.shapes)} paths generated. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
