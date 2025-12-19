"""High-level helper to turn a raster image into preconditioned diffvg shapes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import pydiffvg
from .config import PreconditionConfig
from .edge import compute_edge_mask
from .skeleton import skeletonize_edges, skeleton_to_polylines
from .vectorize import polylines_to_paths
from .lineart import build_lineart_scene
from .teed import teed_edge_strength, teed_mask_from_strength


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
    if cfg.mode == "lineart":
        shapes, groups, edge_mask, skeleton = build_lineart_scene(rgb, cfg, device=device)
        polylines: list[list[tuple[int, int]]] = []
    else:
        if (cfg.edge_backend or "xdog").strip().lower() == "teed" and cfg.teed_auto_tune_threshold:
            edge_mask, skeleton, polylines = _teed_autotuned_edges_and_polylines(rgb, cfg, device=device)
        else:
            edge_mask = compute_edge_mask(rgb, cfg, device=device)
            skeleton = skeletonize_edges(edge_mask)
            polylines = skeleton_to_polylines(skeleton, cfg)
        shapes, groups = polylines_to_paths(
            polylines,
            rgb,
            cfg,
            canvas_w=width,
            canvas_h=height,
            device=device,
        )

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
