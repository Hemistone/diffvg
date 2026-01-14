"""Run raster preconditioning and dump debug artifacts.

This script is intentionally limited to the preconditioning stage: it produces
edge/skeleton masks and an initial render of the generated paths, but does not
run any diffvg optimization. Use painterly_rendering.py with --precondition for
full optimization.
"""

from __future__ import annotations

import os
import argparse
import sys
import tomllib
from pathlib import Path

# Precondition-only debugging should avoid CUDA init for faster, safer runs.
os.environ["DIFFVG_DEVICE"] = "cpu"
os.environ["DIFFVG_FORCE_CPU"] = "1"

import numpy as np
import torch
from PIL import Image

import pydiffvg
from pydiffvg.precondition.cli import build_precondition_config


def _default_teed_weights_path() -> str | None:
    candidate = Path("weights/teed/5_model.pth")
    if candidate.is_file():
        return str(candidate)
    return None


_CONFIG_KEY_ALIASES = {
    "edge_backend": "precond_mode",
    "num_paths": "max_paths",
    "precond_max_paths": "max_paths",
    "precond_min_path_length": "min_path_length",
    "precond_max_path_length": "max_path_length",
    "precond_min_component_area": "min_component_area",
    "precond_morph_open_radius": "morph_open_radius",
    "precond_morph_close_radius": "morph_close_radius",
    "precond_merge_polylines": "merge_polylines",
    "precond_merge_distance": "merge_distance",
    "precond_merge_angle_deg": "merge_angle_deg",
    "precond_force_open_paths": "force_open_paths",
    "precond_base_stroke_width": "base_stroke_width",
    "precond_max_stroke_width": "max_stroke_width",
}


def _load_config_defaults(config_path: str, parser: argparse.ArgumentParser) -> dict:
    with open(config_path, "rb") as handle:
        data = tomllib.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Config must be a TOML table (key/value mapping) at top level.")
    normalized: dict = {}
    for key, value in data.items():
        mapped = _CONFIG_KEY_ALIASES.get(key, key)
        normalized[mapped] = value
    valid = {action.dest for action in parser._actions if action.dest != "config"}
    unknown = sorted(key for key in normalized.keys() if key not in valid)
    if unknown:
        raise ValueError(f"Unknown config keys in {config_path}: {', '.join(unknown)}")
    return normalized


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
    default_config = "configs/precondition/teed_detail_quantile.toml"
    parser.add_argument("--config", type=str, default=default_config, help="TOML config file for default arguments")
    parser.add_argument("image", type=Path, help="Input raster image")
    parser.add_argument("--backend", default="splat", choices=["baseline", "splat"], help="Render backend to use")
    parser.add_argument("--precondition", action="store_true", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--max_width", dest="max_width", type=float, default=None, help="Clamp precondition stroke widths (matches painterly --max_width)")
    parser.add_argument("--max-width", dest="max_width", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--precond-mode",
        type=str,
        default="xdog",
        choices=["xdog", "teed", "lineart", "flowline"],
        help="Preconditioning mode",
    )
    parser.add_argument("--lineart-threshold-mode", type=str, default=None, choices=["quantile", "otsu", "fixed"], help="Lineart threshold mode")
    parser.add_argument("--lineart-threshold-quantile", type=float, default=None, help="Quantile used for lineart threshold (0..1)")
    parser.add_argument("--lineart-threshold", type=float, default=None, help="Fixed threshold for lineart mode (0..1)")
    parser.add_argument("--teed-weights", type=str, default=None, help="Path to TEED/MTEED weights (.pth) for precond_mode=teed")
    parser.add_argument("--teed-detect-res", type=int, default=512, help="TEED detect resolution (min side) before rounding to 64-multiple")
    parser.add_argument("--teed-threshold", type=float, default=0.5, help="Threshold on TEED edge strength (0..1) to form a boolean edge mask")
    parser.add_argument("--teed-safe-steps", type=int, default=2, help="Quantization steps (controlnet-aux safe_step); 0 disables")
    parser.add_argument("--teed-threshold-mode", type=str, default=None, choices=["fixed", "hysteresis", "quantile", "otsu"], help="TEED thresholding mode")
    parser.add_argument("--teed-hysteresis-low-ratio", type=float, default=None, help="Hysteresis low/high ratio (default=0.5)")
    parser.add_argument("--teed-threshold-quantile", type=float, default=None, help="Quantile used for teed threshold mode=quantile (0..1)")
    parser.add_argument("--teed-lineart", action="store_true", default=None, help="Apply lineart intensity boost (Anyline-style)")
    parser.add_argument("--teed-lineart-blur-sigma", type=float, default=None, help="Gaussian sigma for lineart intensity (default from config)")
    parser.add_argument("--teed-lineart-strength", type=float, default=None, help="Strength for lineart intensity boost (default from config)")
    parser.add_argument("--teed-lineart-combine", type=str, default=None, choices=["screen", "max", "add"], help="Combine TEED + lineart intensity")
    parser.add_argument(
        "--flow-edge-backend",
        type=str,
        default=None,
        choices=["teed", "xdog"],
        help="Edge backend for flowline mode",
    )
    parser.add_argument(
        "--flow-seed-mode",
        type=str,
        default=None,
        choices=["quantile", "fixed"],
        help="Seed selection mode for flowline",
    )
    parser.add_argument("--flow-seed-quantile", type=float, default=None, help="Quantile for flowline seeds (0..1)")
    parser.add_argument("--flow-seed-threshold", type=float, default=None, help="Fixed threshold for flowline seeds (0..1)")
    parser.add_argument("--flow-min-strength", type=float, default=None, help="Minimum edge strength to keep tracing (0..1)")
    parser.add_argument("--flow-step-px", type=float, default=None, help="Step size in pixels when tracing flowlines")
    parser.add_argument("--flow-max-len", type=int, default=None, help="Maximum steps per flowline")
    parser.add_argument("--flow-min-len", type=int, default=None, help="Minimum steps per flowline")
    parser.add_argument("--flow-min-seed-dist", type=int, default=None, help="Minimum seed distance (pixels)")
    parser.add_argument("--flow-curvature-deg", type=float, default=None, help="Max curvature between steps (degrees)")
    parser.add_argument("--flow-field-sigma", type=float, default=None, help="Gaussian sigma for flow field smoothing")
    parser.add_argument("--flow-field-iters", type=int, default=None, help="Gaussian smoothing iterations for flow field")
    parser.add_argument("--flow-coverage-decay", type=float, default=None, help="Strength decay around traced paths (0..1)")
    parser.add_argument("--flow-coverage-radius", type=int, default=None, help="Radius in pixels for coverage decay")
    parser.add_argument("--stroke-width-mode", type=str, default=None, choices=["absolute", "a4_pen"], help="Stroke width mode for preconditioning")
    parser.add_argument("--pen-width-min-mm", type=float, default=None, help="A4 pen min width in mm (default=0.35)")
    parser.add_argument("--pen-width-max-mm", type=float, default=None, help="A4 pen max width in mm (default=0.8)")
    parser.add_argument("--max-paths", dest="max_paths", type=int, default=None, help="Cap number of generated paths (default: config default)")
    parser.add_argument("--precond-max-paths", dest="max_paths", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--num-paths", dest="max_paths", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--num_paths", dest="max_paths", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--min-path-length", type=int, default=None, help="Minimum skeleton polyline length in pixels")
    parser.add_argument("--max-path-length", type=int, default=None, help="Maximum skeleton polyline length in pixels (smaller splits long paths)")
    parser.add_argument("--min-component-area", type=int, default=None, help="Remove edge components smaller than this area (pixels)")
    parser.add_argument("--morph-open-radius", type=int, default=None, help="Binary opening radius applied to edge mask")
    parser.add_argument("--morph-close-radius", type=int, default=None, help="Binary closing radius applied to edge mask")
    parser.add_argument("--merge-polylines", action="store_true", default=None, help="Merge polylines that align and are close")
    parser.add_argument("--merge-distance", type=float, default=None, help="Merge distance in pixels (default from config)")
    parser.add_argument("--merge-angle-deg", type=float, default=None, help="Merge angle threshold in degrees (default from config)")
    parser.add_argument("--force-open-paths", action="store_true", default=None, help="Force open stroke paths (avoid closed loops)")
    parser.add_argument("--precond-min-path-length", dest="min_path_length", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--precond-max-path-length", dest="max_path_length", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--precond-min-component-area", dest="min_component_area", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--precond-morph-open-radius", dest="morph_open_radius", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--precond-morph-close-radius", dest="morph_close_radius", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--precond-merge-polylines", dest="merge_polylines", action="store_true", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--precond-merge-distance", dest="merge_distance", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--precond-merge-angle-deg", dest="merge_angle_deg", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--precond-force-open-paths", dest="force_open_paths", action="store_true", default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--precond-target-paths-min",
        type=int,
        default=None,
        help="Target minimum path count for auto-scaling (default: PRECONDITION_TARGET_PATHS_MIN_DEFAULT)",
    )
    parser.add_argument(
        "--precond-target-paths-max",
        type=int,
        default=None,
        help="Target maximum path count for auto-scaling (default: PRECONDITION_TARGET_PATHS_MAX_DEFAULT)",
    )
    parser.add_argument("--base-stroke-width", dest="base_stroke_width", type=float, default=None, help="Base stroke width for preconditioned paths")
    parser.add_argument("--max-stroke-width", dest="max_stroke_width", type=float, default=None, help="Max stroke width for preconditioned paths")
    parser.add_argument("--precond-base-stroke-width", dest="base_stroke_width", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--precond-max-stroke-width", dest="max_stroke_width", type=float, default=None, help=argparse.SUPPRESS)
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

    cfg = build_precondition_config(
        args,
        default_teed_weights_path=_default_teed_weights_path(),
        require_teed_weights=True,
        missing_weights_message="precond_mode=teed requires --teed-weights PATH",
        clamp_max_width=getattr(args, "max_width", None),
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
