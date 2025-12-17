"""Vectorize line-art style rasters into stroke-based SVG using preconditioning."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import pydiffvg


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        rgb = np.array(im.convert("RGB"), dtype=np.float32)
    if rgb.max() > 1.5:
        rgb = rgb / 255.0
    return np.clip(rgb, 0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Line-art vectorization using preconditioning (no painterly noise).")
    parser.add_argument("image", type=Path, help="Input line-art raster")
    parser.add_argument("--backend", default="splat", choices=["baseline", "splat"], help="Render backend")
    parser.add_argument("--num-colors", type=int, default=1, help="Number of inks/colors to extract")
    parser.add_argument("--merge-distance", type=float, default=3.0, help="Endpoint merge distance in pixels")
    parser.add_argument("--merge-angle", type=float, default=18.0, help="Max angle (deg) between tangents for merging")
    parser.add_argument("--simplify-eps", type=float, default=1.1, help="RDP epsilon for path simplification")
    parser.add_argument("--smooth-window", type=int, default=5, help="Moving-average window for smoothing")
    parser.add_argument("--refine-iters", type=int, default=0, help="Optional short diffvg refinement steps")
    parser.add_argument("--out-dir", type=Path, default=Path("results/lineart_vectorize"), help="Output directory")
    args = parser.parse_args()

    pydiffvg.set_backend(args.backend)
    device = pydiffvg.get_device()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rgb = _load_rgb(args.image)
    cfg = pydiffvg.PreconditionConfig(
        mode="lineart",
        num_colors=args.num_colors,
        merge_distance=args.merge_distance,
        merge_angle_deg=args.merge_angle,
        simplify_epsilon=args.simplify_eps,
        smooth_window=args.smooth_window,
        curve_mode="bezier",
    )
    scene = pydiffvg.build_preconditioned_scene(args.image, cfg=cfg, backend=args.backend, device=device)

    # Render initial vectorization
    renderer = scene.renderer
    scene_args = renderer.serialize_scene(
        scene.width,
        scene.height,
        scene.shapes,
        scene.shape_groups,
        device=device,
    )
    init_img = renderer.apply(scene.width, scene.height, 2, 2, 0, None, *scene_args)
    pydiffvg.imwrite(init_img, str(out_dir / "init_render.png"))

    pydiffvg.imwrite(scene.edge_mask.astype(np.float32), str(out_dir / "edge_mask.png"), gamma=1.0)
    pydiffvg.imwrite(scene.skeleton.astype(np.float32), str(out_dir / "skeleton.png"), gamma=1.0)

    if args.refine_iters > 0:
        params = []
        for path in scene.shapes:
            path.points.requires_grad_(True)
            params.append(path.points)
            path.stroke_width.requires_grad_(True)
            params.append(path.stroke_width)
        for group in scene.shape_groups:
            if getattr(group, "stroke_color", None) is not None:
                group.stroke_color.requires_grad_(True)
                params.append(group.stroke_color)
        opt = torch.optim.Adam(params, lr=1e-1)
        target = torch.tensor(np.concatenate([rgb, np.ones((rgb.shape[0], rgb.shape[1], 1), dtype=np.float32)], axis=2), device=device)
        for t in range(args.refine_iters):
            opt.zero_grad()
            scene_args = renderer.serialize_scene(
                scene.width,
                scene.height,
                scene.shapes,
                scene.shape_groups,
                device=device,
                cache_key="main",
                invalidate_cache=True,
            )
            img = renderer.apply(scene.width, scene.height, 2, 2, t, None, *scene_args)
            loss = ((img[..., :3] - target[..., :3]) ** 2).mean()
            loss.backward()
            opt.step()
            if t % max(1, args.refine_iters // 10) == 0 or t == args.refine_iters - 1:
                print(f"[{t}/{args.refine_iters}] refine loss={loss.item():.5f}")
        refined = renderer.apply(scene.width, scene.height, 2, 2, 0, None, *scene_args)
        pydiffvg.imwrite(refined, str(out_dir / "refined_render.png"))

    # Save SVG
    pydiffvg.save_svg(
        str(out_dir / "vectorized.svg"),
        scene.width,
        scene.height,
        scene.shapes,
        scene.shape_groups,
        use_gamma=False,
        background_rgb=None,
    )
    print(f"Wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
