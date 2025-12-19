"""
Scream: python painterly_rendering.py imgs/scream.jpg --num_paths 2048 --max_width 4.0
Fallingwater: python painterly_rendering.py imgs/fallingwater.jpg --num_paths 2048 --max_width 4.0
Fallingwater LPIPS: python painterly_rendering.py imgs/fallingwater.jpg --num_paths 2048 --max_width 4.0 --loss lpips
Baboon: python painterly_rendering.py imgs/baboon.png --num_paths 1024 --max_width 4.0 --num_iter 250
Baboon perceptual: python painterly_rendering.py imgs/baboon.png --num_paths 1024 --max_width 4.0 --num_iter 500 --loss perceptual-balanced
Kitty: python painterly_rendering.py imgs/kitty.jpg --num_paths 1024 --use_blob
"""
import argparse
import math
import os
import random
import sys
import tomllib
from pathlib import Path

import pydiffvg
import numpy as np
import skimage
import skimage.io
import torch

from single_utils import create_run_context, log_run_configuration


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


def _rgba_over_white(img_rgba: torch.Tensor) -> torch.Tensor:
    """Composite RGBA to RGB over white.

    Handles premultiplied vs straight-alpha differences between backends:
    - baseline returns straight RGB (needs multiply by A)
    - splat returns premultiplied RGB (do NOT multiply by A again)
    """
    backend = pydiffvg.get_backend()
    a = img_rgba[:, :, 3:4].clamp(0.0, 1.0)
    rgb = img_rgba[:, :, :3]
    if backend == "splat":
        premul = rgb  # RGB is already premultiplied in splat backend
    else:
        premul = rgb * a  # baseline provides straight (non-premultiplied) RGB
    return (premul + (1.0 - a)).clamp(0.0, 1.0)


gamma = 1.0


def main(args):
    backend = getattr(args, "backend", "baseline").strip().lower()
    pydiffvg.set_backend(backend)
    use_gpu = torch.cuda.is_available()
    pydiffvg.set_use_gpu(use_gpu)

    precondition = bool(getattr(args, "precondition", False))
    lineart_precondition = bool(getattr(args, "lineart_precondition", False))
    edge_backend = getattr(args, "edge_backend", "xdog").strip().lower()
    fixed_stroke = bool(getattr(args, "fixed_stroke", False))
    fixed_stroke_alpha = float(getattr(args, "fixed_stroke_alpha", 0.9))

    # Build loss function
    device = pydiffvg.get_device()
    loss_name = getattr(args, "loss", "mse").strip().lower()
    if loss_name == "mse":
        def _loss_fn(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
            return (pred - tgt).pow(2).mean()
    elif loss_name == "l1":
        def _loss_fn(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
            return (pred - tgt).abs().mean()
    elif loss_name in ("lpips", "msssim", "dists", "perceptual-balanced"):
        try:
            import piq  # type: ignore
        except Exception as e:
            raise RuntimeError(f"loss '{loss_name}' requested but 'piq' is not available. Install with 'pip install piq'.") from e
        if loss_name == "lpips":
            lpips = piq.LPIPS(reduction="mean").to(device)
            def _loss_fn(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
                return lpips(pred * 2.0 - 1.0, tgt * 2.0 - 1.0)
        elif loss_name == "msssim":
            msssim = piq.MultiScaleSSIMLoss(data_range=1.0).to(device)
            def _loss_fn(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
                return msssim(pred, tgt)
        elif loss_name == "dists":
            dists = piq.DISTS(reduction="mean").to(device)
            def _loss_fn(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
                return dists(pred * 2.0 - 1.0, tgt * 2.0 - 1.0)
        else:  # perceptual-balanced preset
            lpips = piq.LPIPS(reduction="mean").to(device)
            def _loss_fn(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
                l_perc = lpips(pred * 2.0 - 1.0, tgt * 2.0 - 1.0)
                l_mse = (pred - tgt).pow(2).mean()
                return l_perc + 0.02 * l_mse
    else:
        raise ValueError(f"Unsupported --loss '{loss_name}'. Choose from: mse, l1, lpips, msssim, dists, perceptual-balanced")

    target_path = Path(args.target)
    # Robust image load: prefer imageio or PIL; fallback to skimage, then sanitize dtype
    img_np: np.ndarray
    try:
        import imageio.v3 as iio  # type: ignore
        img_np = iio.imread(str(target_path))
    except Exception:
        try:
            from PIL import Image  # type: ignore

            with Image.open(str(target_path)) as _im:
                if _im.mode not in ("RGB", "RGBA"):
                    # Convert paletted/LA/others to RGB
                    _im = _im.convert("RGB")
                img_np = np.array(_im)
        except Exception:
            img_np = skimage.io.imread(str(target_path))
    # Sanitize dtype/channels
    if img_np.dtype == object:
        # Some backends can return object arrays for odd encodings; coerce via PIL
        try:
            from PIL import Image  # type: ignore

            with Image.open(str(target_path)) as _im:
                if _im.mode not in ("RGB", "RGBA"):
                    _im = _im.convert("RGB")
                img_np = np.array(_im)
        except Exception:
            img_np = np.array(img_np, dtype=np.uint8)
    if img_np.ndim == 2:
        img_np = np.repeat(img_np[:, :, None], 3, axis=2)
    if img_np.shape[-1] == 4:
        img_np = img_np[:, :, :3]
    if img_np.dtype != np.uint8:
        img_np = np.clip(img_np, 0, 255).astype(np.uint8, copy=False)
    target = torch.from_numpy(img_np).to(torch.float32) / 255.0
    target = target.pow(gamma)
    target = target.to(pydiffvg.get_device())
    target = target.unsqueeze(0)
    target = target.permute(0, 3, 1, 2)  # NHWC -> NCHW
    canvas_width, canvas_height = target.shape[3], target.shape[2]
    num_paths = args.num_paths
    max_width = args.max_width

    def _sanitize_component(value: str) -> str:
        sanitized = [ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value]
        result = "".join(sanitized).strip("_")
        return result or "target"

    run_label = f"{_sanitize_component(target_path.stem)}_paths{num_paths}_"
    run_label += "blob" if args.use_blob else "strokes"
    if precondition:
        run_label += f"_precondition_{edge_backend}"
    elif lineart_precondition:
        run_label += "_lineart_precondition"

    mode_dir = "blob" if args.use_blob else ("strokes_precondition" if (precondition or lineart_precondition) else "strokes")
    image_dir = _sanitize_component(target_path.stem)
    variant_parts = [backend, loss_name]
    if precondition:
        variant_parts.append(edge_backend)
    elif lineart_precondition:
        variant_parts.append("lineart")
    if fixed_stroke and not args.use_blob:
        variant_parts.append("fixed_ink")
    variant_dir = "_".join(variant_parts)

    task_subdir = Path(mode_dir) / str(num_paths) / image_dir / str(args.num_iter) / variant_dir

    run = create_run_context(
        str(task_subdir),
        args.num_iter,
        video_fps=24,
        video_bitrate="20M",
        results_root=Path("results") / "painterly_rendering",
        label=run_label,
    )
    device_str = str(pydiffvg.get_device())
    config_items = {
        "device": device_str,
        "backend": backend,
        "precondition": precondition or lineart_precondition,
        "lineart_precondition": lineart_precondition,
        "edge_backend": edge_backend if precondition else None,
        "fixed_stroke": fixed_stroke if not args.use_blob else None,
        "fixed_stroke_alpha": fixed_stroke_alpha if (fixed_stroke and not args.use_blob) else None,
        "target": str(target_path),
        "canvas": f"{canvas_width}x{canvas_height}",
        "paths": num_paths,
        "iterations": args.num_iter,
        "max_width": max_width,
        "loss": loss_name,
        "blob_mode": args.use_blob,
        "run_dir": run.results_dir,
    }
    if precondition and edge_backend == "teed":
        config_items["teed_weights"] = getattr(args, "teed_weights", None) or _default_teed_weights_path()
        config_items["teed_detect_res"] = getattr(args, "teed_detect_res", None)
        config_items["teed_threshold"] = getattr(args, "teed_threshold", None)
        config_items["teed_safe_steps"] = getattr(args, "teed_safe_steps", None)
        config_items["precond_min_path_length"] = getattr(args, "precond_min_path_length", None)
        config_items["precond_max_path_length"] = getattr(args, "precond_max_path_length", None)
        config_items["precond_min_component_area"] = getattr(args, "precond_min_component_area", None)
        config_items["precond_morph_open_radius"] = getattr(args, "precond_morph_open_radius", None)
        config_items["precond_morph_close_radius"] = getattr(args, "precond_morph_close_radius", None)
        config_items["precond_base_stroke_width"] = getattr(args, "precond_base_stroke_width", None)
        config_items["precond_max_stroke_width"] = getattr(args, "precond_max_stroke_width", None)
    if pydiffvg.get_backend() == "splat":
        raw_thresh = os.environ.get("DIFFVG_SPLAT_TILE_THRESH")
        if raw_thresh is None or raw_thresh.strip() == "":
            tile_thresh_display = "1e-4 (default)"
        else:
            tile_thresh_display = raw_thresh.strip()
        config_items["splat_tile_thresh"] = tile_thresh_display

    log_run_configuration(
        "painterly_rendering",
        config_items,
    )

    random.seed(1234)
    torch.manual_seed(1234)

    renderer = pydiffvg.Renderer(backend=backend)
    shapes = []
    shape_groups = []

    if getattr(args, "precondition", False) or getattr(args, "lineart_precondition", False):
        if args.use_blob:
            raise ValueError("Preconditioning is incompatible with --use_blob (preconditioning generates stroke paths).")
        cfg = pydiffvg.PreconditionConfig()
        cfg.max_paths = num_paths
        if fixed_stroke:
            cfg.fixed_stroke_rgba = (0.0, 0.0, 0.0, max(0.0, min(1.0, fixed_stroke_alpha)))
        if precondition:
            cfg.edge_backend = edge_backend
            if edge_backend == "teed":
                cfg.teed_weights_path = getattr(args, "teed_weights", None) or _default_teed_weights_path()
                if cfg.teed_weights_path is None or str(cfg.teed_weights_path).strip() == "":
                    raise ValueError(
                        "--edge-backend teed requires weights. "
                        "Put them at weights/teed/5_model.pth or pass --teed-weights PATH."
                    )

                cfg.teed_detect_resolution = int(getattr(args, "teed_detect_res", None) or 512)
                cfg.teed_safe_steps = int(getattr(args, "teed_safe_steps", None) if getattr(args, "teed_safe_steps", None) is not None else 0)
                cfg.teed_threshold = float(getattr(args, "teed_threshold", None) if getattr(args, "teed_threshold", None) is not None else 0.30)

                cfg.min_path_length = int(getattr(args, "precond_min_path_length", None) or 4)
                cfg.max_path_length = int(getattr(args, "precond_max_path_length", None) or 64)
                cfg.min_component_area = int(getattr(args, "precond_min_component_area", None) or 0)
                cfg.morph_open_radius = int(getattr(args, "precond_morph_open_radius", None) or 0)
                cfg.morph_close_radius = int(getattr(args, "precond_morph_close_radius", None) or 0)

                base_w = float(getattr(args, "precond_base_stroke_width", None) or 0.5)
                max_w = float(getattr(args, "precond_max_stroke_width", None) or 1.0)
                max_w = min(max_w, max_width)
                cfg.max_stroke_width = max_w
                cfg.base_stroke_width = min(base_w, max_w)
            else:
                # For apples-to-apples comparisons with TEED, keep XDoG defaults
                # aligned unless the user explicitly overrides them.
                base_w = float(getattr(args, "precond_base_stroke_width", None) or 0.5)
                max_w = float(getattr(args, "precond_max_stroke_width", None) or 1.0)
                max_w = min(max_w, max_width)
                cfg.max_stroke_width = max_w
                cfg.base_stroke_width = min(base_w, max_w)
        else:
            cfg.max_stroke_width = max_width
            cfg.base_stroke_width = min(cfg.base_stroke_width, max_width)
        if lineart_precondition:
            if precondition and edge_backend != "xdog":
                print("NOTE: --edge-backend is ignored when using --lineart-precondition.")
            cfg.mode = "lineart"
            cfg.num_colors = 1
        pre = pydiffvg.build_preconditioned_scene(
            args.target,
            cfg=cfg,
            backend=backend,
            device=device,
        )
        if len(pre.shapes) == 0 or len(pre.shape_groups) == 0:
            raise ValueError(
                "Preconditioning produced 0 paths. "
                "Try lowering thresholds (e.g., --teed-threshold) or relaxing cleanup parameters."
            )
        shapes = pre.shapes
        shape_groups = pre.shape_groups
        print(f"[precond] generated {len(shapes)}/{num_paths} paths (edge_backend={edge_backend})")
        if (pre.width, pre.height) != (canvas_width, canvas_height):
            canvas_width, canvas_height = pre.width, pre.height
        # Save debug masks alongside run artifacts
        pydiffvg.imwrite(pre.edge_mask.astype(np.float32), str(run.results_dir / "precond_edge.png"), gamma=1.0)
        pydiffvg.imwrite(pre.skeleton.astype(np.float32), str(run.results_dir / "precond_skeleton.png"), gamma=1.0)
    else:
        for i in range(num_paths):
            if args.use_blob:
                num_segments = random.randint(3, 5)
            else:
                num_segments = random.randint(1, 3)
            num_control_points = torch.zeros(num_segments, dtype=torch.int32, device=device) + 2
            points = []
            p0 = (random.random(), random.random())
            points.append(p0)
            for j in range(num_segments):
                radius = 0.05
                p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
                p2 = (p1[0] + radius * (random.random() - 0.5), p1[1] + radius * (random.random() - 0.5))
                p3 = (p2[0] + radius * (random.random() - 0.5), p2[1] + radius * (random.random() - 0.5))
                points.append(p1)
                points.append(p2)
                if args.use_blob:
                    if j < num_segments - 1:
                        points.append(p3)
                        p0 = p3
                else:
                    points.append(p3)
                    p0 = p3
            points = torch.tensor(points, device=device)
            points[:, 0] *= canvas_width
            points[:, 1] *= canvas_height
            path = pydiffvg.Path(
                num_control_points=num_control_points,
                points=points,
                stroke_width=torch.tensor(1.0, device=device),
                is_closed=args.use_blob,
            )
            shapes.append(path)
            if args.use_blob:
                path_group = pydiffvg.ShapeGroup(
                    shape_ids=torch.tensor([len(shapes) - 1], device=device),
                    fill_color=torch.tensor(
                        [random.random(), random.random(), random.random(), random.random()],
                        device=device,
                    ),
                )
            else:
                if fixed_stroke:
                    stroke_color = torch.tensor([0.0, 0.0, 0.0, max(0.0, min(1.0, fixed_stroke_alpha))], device=device)
                else:
                    stroke_color = torch.tensor(
                        [random.random(), random.random(), random.random(), random.random()],
                        device=device,
                    )
                path_group = pydiffvg.ShapeGroup(
                    shape_ids=torch.tensor([len(shapes) - 1], device=device),
                    fill_color=None,
                    stroke_color=stroke_color,
                )
            shape_groups.append(path_group)

    def _render(seed: int, invalidate: bool = True):
        scene_args = renderer.serialize_scene(
            canvas_width,
            canvas_height,
            shapes,
            shape_groups,
            device=device,
            cache_key="main",
            invalidate_cache=invalidate,
        )
        return renderer.apply(
            canvas_width,
            canvas_height,
            2,
            2,
            seed,
            None,
            *scene_args,
        )

    img = _render(0, invalidate=False)
    # Save initial frame composited over white for visualization
    init_rgb = _rgba_over_white(img)
    pydiffvg.imwrite(init_rgb.cpu(), str(run.results_dir / "init.png"), gamma=gamma)

    points_vars = []
    stroke_width_vars = []
    color_vars = []
    for path in shapes:
        path.points.requires_grad = True
        points_vars.append(path.points)
    if not args.use_blob:
        for path in shapes:
            path.stroke_width.requires_grad = True
            stroke_width_vars.append(path.stroke_width)
    if args.use_blob:
        for group in shape_groups:
            group.fill_color.requires_grad = True
            color_vars.append(group.fill_color)
    else:
        if not fixed_stroke:
            for group in shape_groups:
                group.stroke_color.requires_grad = True
                color_vars.append(group.stroke_color)
    
    # Optimize
    points_optim = torch.optim.Adam(points_vars, lr=1.0)
    width_optim = torch.optim.Adam(stroke_width_vars, lr=0.1) if stroke_width_vars else None
    color_optim = torch.optim.Adam(color_vars, lr=0.01) if color_vars else None

    progress = run.progress
    t = -1
    try:
        for t in range(args.num_iter):
            points_optim.zero_grad()
            if width_optim is not None:
                width_optim.zero_grad()
            if color_optim is not None:
                color_optim.zero_grad()
            img = _render(t)
            # Compose to RGB (over white) and save if requested
            img_rgb = _rgba_over_white(img)
            save_every = getattr(args, "save_every", 1)
            if save_every and save_every > 0 and (t % save_every == 0):
                pydiffvg.imwrite(img_rgb.cpu(), str(run.iter_path(t)), gamma=gamma)
            img = img_rgb
            img = img.unsqueeze(0)
            img = img.permute(0, 3, 1, 2)
            loss = _loss_fn(img, target)

            loss.backward()

            points_optim.step()
            if width_optim is not None:
                width_optim.step()
            if color_optim is not None:
                color_optim.step()
            if width_optim is not None:
                for path in shapes:
                    path.stroke_width.data.clamp_(1.0, max_width)
            if args.use_blob:
                for group in shape_groups:
                    group.fill_color.data.clamp_(0.0, 1.0)
            else:
                if not fixed_stroke:
                    for group in shape_groups:
                        group.stroke_color.data.clamp_(0.0, 1.0)

            svg_every = getattr(args, "save_svg_every", 0)
            if (svg_every and svg_every > 0 and (t % svg_every == 0)) or (t == args.num_iter - 1 and svg_every and svg_every > 0):
                pydiffvg.save_svg(
                    str(run.iter_dir / f"iter_{t:04d}.svg"),
                    canvas_width,
                    canvas_height,
                    shapes,
                    shape_groups,
                    use_gamma=False,
                    background_rgb=(1.0, 1.0, 1.0),
                )

            progress.log(t, loss=loss.item())
    except KeyboardInterrupt:
        progress.interrupt(t)
    finally:
        progress.close()

    # Render the final result.
    img = _render(0)
    final_rgb = _rgba_over_white(img)
    pydiffvg.imwrite(final_rgb.cpu(), str(run.results_dir / "final.png"), gamma=gamma)
    # Also emit final SVG with white background if requested
    if getattr(args, "save_svg_every", 0) and getattr(args, "save_svg_every", 0) > 0:
        pydiffvg.save_svg(
            str(run.results_dir / "final.svg"),
            canvas_width,
            canvas_height,
            shapes,
            shape_groups,
            use_gamma=False,
            background_rgb=(1.0, 1.0, 1.0),
        )

    run.make_video()

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="TOML config file for default arguments")
    parser.add_argument("target", help="target image path")
    parser.add_argument("--num_paths", type=int, default=512)
    parser.add_argument("--max_width", type=float, default=2.0)
    parser.add_argument("--backend", type=str, default="baseline", choices=["baseline", "splat"], help="Render backend")
    parser.add_argument("--precondition", action="store_true", help="Seed paths via preconditioning instead of random init (see --edge-backend)")
    parser.add_argument("--lineart-precondition", action="store_true", help="Seed paths via line-art preconditioning (palette + skeleton).")
    parser.add_argument("--edge-backend", type=str, default="xdog", choices=["xdog", "teed"], help="Edge backend for --precondition")
    parser.add_argument("--teed-weights", type=str, default=None, help="Path to TEED weights (.pth). Default: weights/teed/5_model.pth if present")
    parser.add_argument("--teed-detect-res", type=int, default=None, help="TEED detect resolution (min side); default=512 for TEED runs")
    parser.add_argument("--teed-threshold", type=float, default=None, help="TEED threshold (0..1); default=0.30 for TEED runs")
    parser.add_argument("--teed-safe-steps", type=int, default=None, help="Quantization steps; default=0 for TEED runs")
    parser.add_argument("--precond-min-path-length", type=int, default=None, help="Preconditioning min skeleton polyline length; default=4 for TEED")
    parser.add_argument("--precond-max-path-length", type=int, default=None, help="Preconditioning max skeleton polyline length; default=64 for TEED")
    parser.add_argument("--precond-min-component-area", type=int, default=None, help="Remove edge components smaller than this area; default=0 for TEED")
    parser.add_argument("--precond-morph-open-radius", type=int, default=None, help="Edge mask binary opening radius; default=0 for TEED")
    parser.add_argument("--precond-morph-close-radius", type=int, default=None, help="Edge mask binary closing radius; default=0 for TEED")
    parser.add_argument("--precond-base-stroke-width", type=float, default=None, help="Precondition base stroke width; default=0.5 for TEED")
    parser.add_argument("--precond-max-stroke-width", type=float, default=None, help="Precondition max stroke width; default=1.0 for TEED (clamped to --max_width)")
    parser.add_argument("--fixed-stroke", action="store_true", help="Fix all stroke colors to black ink (disables color optimization)")
    parser.add_argument("--fixed-stroke-alpha", type=float, default=0.9, help="Alpha for --fixed-stroke (0..1)")
    parser.add_argument("--loss", type=str, default="mse", help="Loss: mse|l1|lpips|msssim|dists|perceptual-balanced")
    parser.add_argument("--num_iter", type=int, default=500)
    parser.add_argument("--save_svg_every", type=int, default=0, help="Save SVG every N iters (0 disables)")
    parser.add_argument(
        "--save_every",
        type=int,
        default=1,
        help="Save PNG every N iters (1 saves every iter, 0 disables)",
    )
    parser.add_argument("--use_blob", dest="use_blob", action="store_true")

    argv = sys.argv[1:]
    known, _ = parser.parse_known_args(argv)
    if known.config:
        defaults = _load_config_defaults(known.config, parser)
        parser.set_defaults(**defaults)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    main(args)
