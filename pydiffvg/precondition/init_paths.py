"""High-level helper to turn a raster image into preconditioned diffvg shapes."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import pydiffvg
from .config import PreconditionConfig
from .edge import compute_edge_mask
from .skeleton import skeletonize_edges, skeleton_to_polylines
from .vectorize import merge_polylines, polylines_to_paths
from .lineart import build_lineart_scene
from .flowline import flowline_polylines
from .teed import teed_edge_strength, teed_mask_from_strength

_AUTO_MIN_SIDE_FLOOR = 128  # Prevent overly tiny edge maps.
_AUTO_MIN_SIDE_CAP = 2048  # Avoid runaway auto-upscaling cost.

def _apply_stroke_width_mode(cfg: PreconditionConfig, width: int, height: int) -> None:
    mode = (cfg.stroke_width_mode or "absolute").strip().lower()
    if mode == "absolute":
        return
    if mode != "a4_pen":
        raise ValueError(f"Unsupported stroke_width_mode '{cfg.stroke_width_mode}'. Choose from: absolute, a4_pen")

    if width <= 0 or height <= 0:
        return

    if width >= height:
        a4_w_mm, a4_h_mm = 297.0, 210.0
    else:
        a4_w_mm, a4_h_mm = 210.0, 297.0

    px_per_mm = min(width / a4_w_mm, height / a4_h_mm)
    min_mm = max(0.0, float(cfg.stroke_width_pen_min_mm))
    max_mm = max(0.0, float(cfg.stroke_width_pen_max_mm))
    if max_mm < min_mm:
        max_mm = min_mm

    cfg.base_stroke_width = min_mm * px_per_mm
    cfg.max_stroke_width = max_mm * px_per_mm


def _resize_to_min_side(
    rgb: np.ndarray,
    min_side: int,
    *,
    allow_downscale: bool,
) -> tuple[np.ndarray, float]:
    min_side = int(min_side)
    if min_side <= 0:
        return rgb, 1.0
    height, width = rgb.shape[0], rgb.shape[1]
    current_min = min(height, width)
    if current_min == min_side:
        return rgb, 1.0
    if current_min > min_side and not allow_downscale:
        return rgb, 1.0
    scale = float(min_side) / float(current_min)
    new_w = int(round(width * scale))
    new_h = int(round(height * scale))
    img = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)
    pil = Image.fromarray(img, mode="RGB")
    pil = pil.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
    resized = np.asarray(pil, dtype=np.float32) / 255.0
    return resized, scale


def _resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    if mask.shape == (height, width):
        return mask
    img = (mask.astype(np.uint8) * 255)
    pil = Image.fromarray(img, mode="L")
    pil = pil.resize((width, height), resample=Image.Resampling.NEAREST)
    return np.asarray(pil, dtype=np.uint8) > 127


def _scale_polylines(
    polylines: list[list[tuple[int, int]]],
    scale: float,
) -> list[list[tuple[int, int]]]:
    if scale == 1.0:
        return polylines
    inv = 1.0 / float(scale)
    out: list[list[tuple[int, int]]] = []
    for poly in polylines:
        scaled: list[tuple[int, int]] = []
        last = None
        for y, x in poly:
            ny = int(round(y * inv))
            nx = int(round(x * inv))
            if last == (ny, nx):
                continue
            scaled.append((ny, nx))
            last = (ny, nx)
        if scaled:
            out.append(scaled)
    return out


def _edges_and_polylines(
    rgb: np.ndarray,
    cfg: PreconditionConfig,
    *,
    device: torch.device,
    preserve_threshold: bool = False,
    disable_auto_tune: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[list[tuple[int, int]]]]:
    orig_thr = cfg.teed_threshold
    if (
        (cfg.mode or "xdog").strip().lower() == "teed"
        and cfg.teed_auto_tune_threshold
        and not disable_auto_tune
        and (cfg.teed_threshold_mode or "fixed").strip().lower() == "fixed"
    ):
        edge_mask, skeleton, polylines = _teed_autotuned_edges_and_polylines(rgb, cfg, device=device)
    else:
        edge_mask = compute_edge_mask(rgb, cfg, device=device)
        skeleton = skeletonize_edges(edge_mask)
        polylines = skeleton_to_polylines(skeleton, cfg)
        polylines = merge_polylines(polylines, cfg, enabled=cfg.merge_polylines)
    if preserve_threshold:
        cfg.teed_threshold = orig_thr
    return edge_mask, skeleton, polylines


def _precondition_raw_count(
    rgb: np.ndarray,
    cfg: PreconditionConfig,
    *,
    device: torch.device,
) -> int:
    if cfg.mode == "lineart":
        shapes, _, _, _ = build_lineart_scene(rgb, cfg, device=device)
        return len(shapes)
    if cfg.mode == "flowline":
        _, polylines = flowline_polylines(rgb, cfg, device=device)
        return len(polylines)
    # Disable TEED auto-tune here so auto-scaling doesn't fight threshold tuning.
    _, _, polylines = _edges_and_polylines(
        rgb,
        cfg,
        device=device,
        preserve_threshold=True,
        disable_auto_tune=True,
    )
    return len(polylines)


def _auto_select_min_side(
    rgb: np.ndarray,
    cfg: PreconditionConfig,
    *,
    device: torch.device,
) -> int:
    """Pick a resize min-side so the preconditioning path count lands in target range."""
    height, width = rgb.shape[0], rgb.shape[1]
    orig_min = min(height, width)
    min_side = max(orig_min, _AUTO_MIN_SIDE_FLOOR)
    max_side = max(min_side, _AUTO_MIN_SIDE_CAP)

    target_min = max(1, int(cfg.precondition_target_paths_min))
    target_max = max(target_min, int(cfg.precondition_target_paths_max))
    if cfg.max_paths is not None:
        target_max = min(target_max, int(cfg.max_paths))
        target_min = min(target_min, target_max)
    target_mid = 0.5 * (target_min + target_max)

    trials = max(1, int(cfg.precondition_auto_trials))
    best_side = min_side
    best_score = float("inf")

    # We approximate that path count grows ~area, so we scale by sqrt(target/count).
    for _ in range(trials):
        rgb_proc, _ = _resize_to_min_side(rgb, min_side, allow_downscale=True)
        count = _precondition_raw_count(rgb_proc, cfg, device=device)
        score = abs(count - target_mid)
        if score < best_score:
            best_score = score
            best_side = min_side
        if target_min <= count <= target_max:
            return min_side
        if count <= 0:
            next_side = int(round(min_side * 1.5))
        else:
            scale_factor = math.sqrt(target_mid / float(count))
            next_side = int(round(min_side * scale_factor))
        next_side = max(_AUTO_MIN_SIDE_FLOOR, min(max_side, next_side))
        if next_side == min_side:
            break
        min_side = next_side

    return best_side


def _teed_autotuned_edges_and_polylines(
    rgb: np.ndarray,
    cfg: PreconditionConfig,
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[list[tuple[int, int]]]]:
    strength = teed_edge_strength(rgb, cfg, device=device)

    target = int(cfg.max_paths) if cfg.max_paths is not None else 0
    if target <= 0:
        edges = teed_mask_from_strength(strength, cfg)
        skel = skeletonize_edges(edges)
        polys = skeleton_to_polylines(skel, cfg)
        polys = merge_polylines(polys, cfg, enabled=cfg.merge_polylines)
        return edges, skel, polys

    thr = float(cfg.teed_threshold)
    thr_min = float(cfg.teed_threshold_min)
    decay = float(cfg.teed_threshold_decay)
    trials = int(max(1, cfg.teed_threshold_trials))

    best_edges: np.ndarray | None = None
    best_skel: np.ndarray | None = None
    best_polys: list[list[tuple[int, int]]] = []

    for _ in range(trials):
        thr = max(thr, thr_min)
        edges = teed_mask_from_strength(strength, cfg, threshold=thr)
        skel = skeletonize_edges(edges)
        polys = skeleton_to_polylines(skel, cfg)
        polys = merge_polylines(polys, cfg, enabled=cfg.merge_polylines)
        if len(polys) > len(best_polys):
            best_edges, best_skel, best_polys = edges, skel, polys
        if len(polys) >= target:
            cfg.teed_threshold = thr
            return edges, skel, polys
        if thr <= thr_min + 1e-6:
            break
        thr *= decay

    if best_edges is None or best_skel is None:
        best_edges = teed_mask_from_strength(strength, cfg, threshold=thr_min)
        best_skel = skeletonize_edges(best_edges)
        best_polys = skeleton_to_polylines(best_skel, cfg)
        best_polys = merge_polylines(best_polys, cfg, enabled=cfg.merge_polylines)
    cfg.teed_threshold = thr_min
    return best_edges, best_skel, best_polys


def _load_image(image: str | Path | np.ndarray) -> np.ndarray:
    if isinstance(image, np.ndarray):
        arr = image
    else:
        with Image.open(image) as im:
            arr = np.array(im.convert("RGB"))
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=2)
    if arr.shape[2] == 4:
        arr = arr[..., :3]
    arr = np.asarray(arr, dtype=np.float32)
    if arr.max() > 1.5:  # assume 0-255
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


@dataclass
class PreconditionedScene:
    width: int
    height: int
    shapes: list[pydiffvg.Path]
    shape_groups: list[pydiffvg.ShapeGroup]
    renderer: pydiffvg.Renderer
    scene_args: tuple
    edge_mask: np.ndarray
    skeleton: np.ndarray
    polylines: list[list[tuple[int, int]]]


def build_preconditioned_scene(
    image: str | Path | np.ndarray,
    cfg: PreconditionConfig | None = None,
    *,
    backend: str = "splat",
    device: torch.device | None = None,
) -> PreconditionedScene:
    """Generate initial diffvg paths from a raster image."""
    cfg = cfg or PreconditionConfig()
    device = device if device is not None else pydiffvg.get_device()

    rgb = _load_image(image)
    height, width = rgb.shape[0], rgb.shape[1]
    _apply_stroke_width_mode(cfg, width, height)
    if (backend or "").strip().lower() == "splat":
        cfg.force_open_paths = True
    # Auto-scale (up or down) to keep preconditioning path count in a stable range.
    if cfg.max_paths is None:
        cfg.max_paths = int(cfg.precondition_target_paths_max)
    auto_min_side = _auto_select_min_side(rgb, cfg, device=device)
    rgb_proc, scale = _resize_to_min_side(rgb, auto_min_side, allow_downscale=True)
    proc_h, proc_w = rgb_proc.shape[0], rgb_proc.shape[1]
    if cfg.mode == "lineart":
        shapes, groups, edge_mask, skeleton = build_lineart_scene(rgb_proc, cfg, device=device)
        polylines: list[list[tuple[int, int]]] = []
    elif cfg.mode == "flowline":
        edge_strength, polylines = flowline_polylines(rgb_proc, cfg, device=device)
        edge_mask = edge_strength.astype(np.float32, copy=False)
        skeleton = np.zeros_like(edge_mask, dtype=np.float32)
        shapes, groups = polylines_to_paths(
            polylines,
            rgb_proc,
            cfg,
            canvas_w=proc_w,
            canvas_h=proc_h,
            device=device,
        )
    else:
        edge_mask, skeleton, polylines = _edges_and_polylines(rgb_proc, cfg, device=device)
        shapes, groups = polylines_to_paths(
            polylines,
            rgb_proc,
            cfg,
            canvas_w=proc_w,
            canvas_h=proc_h,
            device=device,
        )
    if scale != 1.0:
        inv = 1.0 / scale
        for path in shapes:
            path.points = path.points * inv
        edge_mask = _resize_mask(edge_mask, width, height)
        skeleton = _resize_mask(skeleton, width, height)
        polylines = _scale_polylines(polylines, scale)

    renderer = pydiffvg.Renderer(backend=backend)
    scene_args = renderer.serialize_scene(
        width,
        height,
        shapes,
        groups,
        device=device,
    )

    return PreconditionedScene(
        width=width,
        height=height,
        shapes=shapes,
        shape_groups=groups,
        renderer=renderer,
        scene_args=scene_args,
        edge_mask=edge_mask,
        skeleton=skeleton,
        polylines=polylines,
    )


__all__ = ["PreconditionedScene", "build_preconditioned_scene"]
