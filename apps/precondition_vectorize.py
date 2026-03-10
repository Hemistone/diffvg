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

import numpy as np
import torch
from PIL import Image

def _default_teed_weights_path() -> str | None:
    candidate = Path("weights/teed/5_model.pth")
    if candidate.is_file():
        return str(candidate)
    return None


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


def _rgba_over_white(img_rgba: torch.Tensor, backend: str) -> torch.Tensor:
    """Composite RGBA to RGB over white (matches painterly_rendering.py)."""
    a = img_rgba[:, :, 3:4].clamp(0.0, 1.0)
    rgb = img_rgba[:, :, :3]
    if backend in {"splat", "bezier_gsplat"}:
        premul = rgb
    else:
        premul = rgb * a
    return (premul + (1.0 - a)).clamp(0.0, 1.0)


def _render(renderer, width: int, height: int, shapes, groups, device, cache_key="main", invalidate=False):
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
    parser.add_argument("--palette", type=str, default=None, help="Palette name or path (configs/palette/...)")
    parser.add_argument("image", type=Path, help="Input raster image")
    parser.add_argument("--backend", default="splat", choices=["baseline", "splat", "bezier_gsplat"], help="Render backend to use")
    parser.add_argument("--precondition", action="store_true", default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--precond-mode",
        type=str,
        default="xdog",
        choices=["xdog", "teed", "lineart", "flowline"],
        help="Preconditioning mode",
    )
    parser.add_argument("--precond-lineart-threshold-mode", dest="precond_lineart_threshold_mode", type=str, default=None, choices=["quantile", "otsu", "fixed"], help="Lineart threshold mode")
    parser.add_argument("--precond-lineart-threshold-quantile", dest="precond_lineart_threshold_quantile", type=float, default=None, help="Quantile used for lineart threshold (0..1)")
    parser.add_argument("--precond-lineart-threshold", dest="precond_lineart_threshold", type=float, default=None, help="Fixed threshold for lineart mode (0..1)")
    parser.add_argument("--precond-lineart-mask-count", dest="precond_lineart_mask_count", type=int, default=None, help="Lineart mask color count (1 disables palette masking)")
    parser.add_argument("--precond-lineart-mask-mode", dest="precond_lineart_mask_mode", type=str, default=None, choices=["auto", "fixed"], help="Lineart mask palette mode")
    parser.add_argument("--precond-lineart-mask-colors", dest="precond_lineart_mask_colors", default=None, help="Lineart mask colors (use config for lists)")
    parser.add_argument("--precond-teed-weights", dest="precond_teed_weights", type=str, default=None, help="Path to TEED/MTEED weights (.pth) for precond_mode=teed")
    parser.add_argument("--precond-teed-detect-res", dest="precond_teed_detect_res", type=int, default=512, help="TEED detect resolution (min side) before rounding to 64-multiple")
    parser.add_argument("--precond-teed-threshold", dest="precond_teed_threshold", type=float, default=0.5, help="Threshold on TEED edge strength (0..1) to form a boolean edge mask")
    parser.add_argument("--precond-teed-safe-steps", dest="precond_teed_safe_steps", type=int, default=2, help="Quantization steps (controlnet-aux safe_step); 0 disables")
    parser.add_argument("--precond-teed-threshold-mode", dest="precond_teed_threshold_mode", type=str, default=None, choices=["fixed", "hysteresis", "quantile", "otsu"], help="TEED thresholding mode")
    parser.add_argument("--precond-teed-hysteresis-low-ratio", dest="precond_teed_hysteresis_low_ratio", type=float, default=None, help="Hysteresis low/high ratio (default=0.5)")
    parser.add_argument("--precond-teed-threshold-quantile", dest="precond_teed_threshold_quantile", type=float, default=None, help="Quantile used for teed threshold mode=quantile (0..1)")
    parser.add_argument("--precond-teed-lineart", dest="precond_teed_lineart", action="store_true", default=None, help="Apply lineart intensity boost (Anyline-style)")
    parser.add_argument("--precond-teed-lineart-blur-sigma", dest="precond_teed_lineart_blur_sigma", type=float, default=None, help="Gaussian sigma for lineart intensity (default from config)")
    parser.add_argument("--precond-teed-lineart-strength", dest="precond_teed_lineart_strength", type=float, default=None, help="Strength for lineart intensity boost (default from config)")
    parser.add_argument("--precond-teed-lineart-combine", dest="precond_teed_lineart_combine", type=str, default=None, choices=["screen", "max", "add"], help="Combine TEED + lineart intensity")
    parser.add_argument(
        "--precond-flow-edge-backend",
        dest="precond_flow_edge_backend",
        type=str,
        default=None,
        choices=["teed", "xdog"],
        help="Edge backend for flowline mode",
    )
    parser.add_argument(
        "--precond-flow-seed-mode",
        dest="precond_flow_seed_mode",
        type=str,
        default=None,
        choices=["quantile", "fixed"],
        help="Seed selection mode for flowline",
    )
    parser.add_argument("--precond-flow-seed-quantile", dest="precond_flow_seed_quantile", type=float, default=None, help="Quantile for flowline seeds (0..1)")
    parser.add_argument("--precond-flow-seed-threshold", dest="precond_flow_seed_threshold", type=float, default=None, help="Fixed threshold for flowline seeds (0..1)")
    parser.add_argument("--precond-flow-min-strength", dest="precond_flow_min_strength", type=float, default=None, help="Minimum edge strength to keep tracing (0..1)")
    parser.add_argument("--precond-flow-step-px", dest="precond_flow_step_px", type=float, default=None, help="Step size in pixels when tracing flowlines")
    parser.add_argument("--precond-flow-max-len", dest="precond_flow_max_len", type=int, default=None, help="Maximum steps per flowline")
    parser.add_argument("--precond-flow-min-len", dest="precond_flow_min_len", type=int, default=None, help="Minimum steps per flowline")
    parser.add_argument("--precond-flow-min-seed-dist", dest="precond_flow_min_seed_dist", type=int, default=None, help="Minimum seed distance (pixels)")
    parser.add_argument("--precond-flow-curvature-deg", dest="precond_flow_curvature_deg", type=float, default=None, help="Max curvature between steps (degrees)")
    parser.add_argument("--precond-flow-field-sigma", dest="precond_flow_field_sigma", type=float, default=None, help="Gaussian sigma for flow field smoothing")
    parser.add_argument("--precond-flow-field-iters", dest="precond_flow_field_iters", type=int, default=None, help="Gaussian smoothing iterations for flow field")
    parser.add_argument("--precond-flow-coverage-decay", dest="precond_flow_coverage_decay", type=float, default=None, help="Strength decay around traced paths (0..1)")
    parser.add_argument("--precond-flow-coverage-radius", dest="precond_flow_coverage_radius", type=int, default=None, help="Radius in pixels for coverage decay")
    parser.add_argument("--precond-width-mode", dest="precond_width_mode", type=str, default=None, choices=["absolute", "a4_pen"], help="Width mode for preconditioning")
    parser.add_argument("--precond-width-min-mm", dest="precond_width_min_mm", type=float, default=None, help="A4 width minimum in mm (default=0.35)")
    parser.add_argument("--precond-width-max-mm", dest="precond_width_max_mm", type=float, default=None, help="A4 width maximum in mm (default=0.8)")
    parser.add_argument("--precond-max-paths", dest="precond_max_paths", type=int, default=None, help="Cap number of generated paths (default: config default)")
    parser.add_argument("--precond-min-path-length", dest="precond_min_path_length", type=int, default=None, help="Minimum skeleton polyline length in pixels")
    parser.add_argument("--precond-max-path-length", dest="precond_max_path_length", type=int, default=None, help="Maximum skeleton polyline length in pixels (smaller splits long paths)")
    parser.add_argument("--precond-min-component-area", dest="precond_min_component_area", type=int, default=None, help="Remove edge components smaller than this area (pixels)")
    parser.add_argument("--precond-morph-open-radius", dest="precond_morph_open_radius", type=int, default=None, help="Binary opening radius applied to edge mask")
    parser.add_argument("--precond-morph-close-radius", dest="precond_morph_close_radius", type=int, default=None, help="Binary closing radius applied to edge mask")
    parser.add_argument("--precond-merge-polylines", dest="precond_merge_polylines", action="store_true", default=None, help="Merge polylines that align and are close")
    parser.add_argument("--precond-merge-distance", dest="precond_merge_distance", type=float, default=None, help="Merge distance in pixels (default from config)")
    parser.add_argument("--precond-merge-angle-deg", dest="precond_merge_angle_deg", type=float, default=None, help="Merge angle threshold in degrees (default from config)")
    parser.add_argument("--precond-force-open-paths", dest="precond_force_open_paths", action="store_true", default=None, help="Force open stroke paths (avoid closed loops)")
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
    parser.add_argument("--precond-base-width", dest="precond_base_width", type=float, default=None, help="Base width for preconditioned paths")
    parser.add_argument("--precond-max-width", dest="precond_max_width", type=float, default=None, help="Max width for preconditioned paths")
    parser.add_argument("--out-dir", type=Path, default=Path("results/precondition"), help="Where to write renders/debug outputs")

    argv = sys.argv[1:]
    known, _ = parser.parse_known_args(argv)
    if known.config:
        defaults = _load_config_defaults(known.config, parser)
        parser.set_defaults(**defaults)
    args = parser.parse_args(argv)

    if args.backend == "bezier_gsplat":
        os.environ.pop("DIFFVG_DEVICE", None)
        os.environ.pop("DIFFVG_FORCE_CPU", None)
    else:
        # Precondition-only debugging stays on CPU unless the user explicitly
        # asks for the CUDA-only gsplat backend.
        os.environ["DIFFVG_DEVICE"] = "cpu"
        os.environ["DIFFVG_FORCE_CPU"] = "1"

    import pydiffvg
    from pydiffvg.palette import load_palette
    from pydiffvg.precondition.cli import build_precondition_config

    pydiffvg.set_backend(args.backend)
    device = pydiffvg.get_device()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    palette = None
    if args.palette:
        palette = load_palette(args.palette)

    cfg = build_precondition_config(
        args,
        default_teed_weights_path=_default_teed_weights_path(),
        require_teed_weights=True,
        missing_weights_message="precond_mode=teed requires --precond-teed-weights PATH",
    )
    if palette is not None:
        cfg.palette = palette
    scene = pydiffvg.build_preconditioned_scene(args.image, cfg=cfg, backend=args.backend, device=device)

    # Debug outputs for the preconditioning stage
    pydiffvg.imwrite(scene.edge_mask.astype(np.float32), str(out_dir / "edge_mask.png"), gamma=1.0)
    pydiffvg.imwrite(scene.skeleton.astype(np.float32), str(out_dir / "skeleton.png"), gamma=1.0)

    if len(scene.shapes) == 0 or len(scene.shape_groups) == 0:
        print(
            f"Preconditioning complete but produced 0 paths. "
            f"Try lowering thresholds (e.g., --precond-teed-threshold) or relaxing cleanup. Outputs in {out_dir}"
        )
        return

    init_img = _render(scene.renderer, scene.width, scene.height, scene.shapes, scene.shape_groups, device, invalidate=True)
    pydiffvg.imwrite(init_img, str(out_dir / "init_render.png"))
    init_rgb = _rgba_over_white(init_img, args.backend)
    pydiffvg.imwrite(init_rgb, str(out_dir / "init_render_over_white.png"), gamma=1.0)
    print(f"Preconditioning complete: {len(scene.shapes)} paths generated. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
