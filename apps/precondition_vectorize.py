"""Run raster preconditioning and dump debug artifacts.

This script is intentionally limited to the preconditioning stage: it produces
edge/skeleton masks and an initial render of the generated paths, but does not
run any diffvg optimization. Use painterly_rendering.py with --precondition for
full optimization.
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import pydiffvg
from pydiffvg.precondition.cli import (
    apply_fixed_stroke_config,
    apply_pen_widths,
    apply_precondition_cleanup,
    apply_stroke_widths,
    apply_teed_settings,
)


def _load_config_defaults(config_path: str, parser: argparse.ArgumentParser) -> dict:
    with open(config_path, "rb") as handle:
        data = tomllib.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Config must be a TOML table (key/value mapping) at top level.")
    valid = {action.dest for action in parser._actions if action.dest != "config"}
    unknown = sorted(key for key in data.keys() if key not in valid)
    if unknown:
        raise ValueError(f"Unknown config keys in {config_path}: {', '.join(unknown)}")
    return data


def _rgba_over_white(img_rgba: torch.Tensor) -> torch.Tensor:
    """Composite RGBA to RGB over white (matches painterly_rendering.py)."""
    backend = pydiffvg.get_backend()
    a = img_rgba[:, :, 3:4].clamp(0.0, 1.0)
    rgb = img_rgba[:, :, :3]
    if backend == "splat":
        premul = rgb
    else:
        premul = rgb * a
    return (premul + (1.0 - a)).clamp(0.0, 1.0)


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
    parser.add_argument("--config", type=str, default=None, help="TOML config file for default arguments")
    parser.add_argument("image", type=Path, help="Input raster image")
    parser.add_argument("--backend", default="splat", choices=["baseline", "splat"], help="Render backend to use")
    parser.add_argument("--edge-backend", type=str, default="xdog", choices=["xdog", "teed"], help="Edge backend for preconditioning")
    parser.add_argument("--teed-weights", type=str, default=None, help="Path to TEED/MTEED weights (.pth) for --edge-backend teed")
    parser.add_argument("--teed-detect-res", type=int, default=512, help="TEED detect resolution (min side) before rounding to 64-multiple")
    parser.add_argument("--teed-threshold", type=float, default=0.5, help="Threshold on TEED edge strength (0..1) to form a boolean edge mask")
    parser.add_argument("--teed-safe-steps", type=int, default=2, help="Quantization steps (controlnet-aux safe_step); 0 disables")
    parser.add_argument("--teed-threshold-mode", type=str, default=None, choices=["fixed", "hysteresis"], help="TEED thresholding mode")
    parser.add_argument("--teed-hysteresis-low-ratio", type=float, default=None, help="Hysteresis low/high ratio (default=0.5)")
    parser.add_argument("--stroke-width-mode", type=str, default=None, choices=["absolute", "a4_pen"], help="Stroke width mode for preconditioning")
    parser.add_argument("--pen-width-min-mm", type=float, default=None, help="A4 pen min width in mm (default=0.35)")
    parser.add_argument("--pen-width-max-mm", type=float, default=None, help="A4 pen max width in mm (default=0.8)")
    parser.add_argument("--max-paths", type=int, default=None, help="Cap number of generated paths (default: config default)")
    parser.add_argument("--min-path-length", type=int, default=None, help="Minimum skeleton polyline length in pixels")
    parser.add_argument("--max-path-length", type=int, default=None, help="Maximum skeleton polyline length in pixels (smaller splits long paths)")
    parser.add_argument("--min-component-area", type=int, default=None, help="Remove edge components smaller than this area (pixels)")
    parser.add_argument("--morph-open-radius", type=int, default=None, help="Binary opening radius applied to edge mask")
    parser.add_argument("--morph-close-radius", type=int, default=None, help="Binary closing radius applied to edge mask")
    parser.add_argument("--base-stroke-width", type=float, default=None, help="Base stroke width for preconditioned paths")
    parser.add_argument("--max-stroke-width", type=float, default=None, help="Max stroke width for preconditioned paths")
    parser.add_argument("--fixed-stroke", action="store_true", help="Force all stroke colors to opaque-ish black ink")
    parser.add_argument("--fixed-stroke-alpha", type=float, default=0.9, help="Alpha used with --fixed-stroke (0..1)")
    parser.add_argument("--out-dir", type=Path, default=Path("results/precondition"), help="Where to write renders/debug outputs")

    argv = sys.argv[1:]
    known, _ = parser.parse_known_args(argv)
    if known.config:
        defaults = _load_config_defaults(known.config, parser)
        parser.set_defaults(**defaults)
    args = parser.parse_args(argv)

    pydiffvg.set_backend(args.backend)
    device = pydiffvg.get_device()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = pydiffvg.PreconditionConfig()
    apply_pen_widths(
        cfg,
        stroke_width_mode=getattr(args, "stroke_width_mode", None),
        pen_min_mm=getattr(args, "pen_width_min_mm", None),
        pen_max_mm=getattr(args, "pen_width_max_mm", None),
    )
    apply_precondition_cleanup(
        cfg,
        max_paths=args.max_paths,
        min_path_length=args.min_path_length,
        max_path_length=args.max_path_length,
        min_component_area=args.min_component_area,
        morph_open_radius=args.morph_open_radius,
        morph_close_radius=args.morph_close_radius,
    )
    apply_stroke_widths(
        cfg,
        base_stroke_width=args.base_stroke_width,
        max_stroke_width=args.max_stroke_width,
    )
    apply_fixed_stroke_config(
        cfg,
        enabled=getattr(args, "fixed_stroke", False),
        alpha=getattr(args, "fixed_stroke_alpha", None),
    )

    cfg.edge_backend = args.edge_backend
    if args.edge_backend == "teed":
        apply_teed_settings(
            cfg,
            weights_path=args.teed_weights,
            detect_res=args.teed_detect_res,
            threshold=args.teed_threshold,
            safe_steps=args.teed_safe_steps,
            threshold_mode=getattr(args, "teed_threshold_mode", None),
            hysteresis_low_ratio=getattr(args, "teed_hysteresis_low_ratio", None),
            require_weights=True,
            missing_weights_message="--edge-backend teed requires --teed-weights PATH",
        )
    scene = pydiffvg.build_preconditioned_scene(args.image, cfg=cfg, backend=args.backend, device=device)

    # Debug outputs for the preconditioning stage
    pydiffvg.imwrite(scene.edge_mask.astype(np.float32), str(out_dir / "edge_mask.png"), gamma=1.0)
    pydiffvg.imwrite(scene.skeleton.astype(np.float32), str(out_dir / "skeleton.png"), gamma=1.0)

    if len(scene.shapes) == 0 or len(scene.shape_groups) == 0:
        print(
            f"Preconditioning complete but produced 0 paths. "
            f"Try lowering thresholds (e.g., --teed-threshold) or relaxing cleanup. Outputs in {out_dir}"
        )
        return

    init_img = _render(scene.renderer, scene.width, scene.height, scene.shapes, scene.shape_groups, device, invalidate=True)
    pydiffvg.imwrite(init_img, str(out_dir / "init_render.png"))
    init_rgb = _rgba_over_white(init_img)
    pydiffvg.imwrite(init_rgb, str(out_dir / "init_render_over_white.png"), gamma=1.0)
    print(f"Preconditioning complete: {len(scene.shapes)} paths generated. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
