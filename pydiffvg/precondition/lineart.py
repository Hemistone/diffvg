"""Preconditioning pipeline for already-sketchy line-art images.

This skips edge detectors and instead:
1) Quantizes to a small palette (auto or fixed).
2) Binarizes per color mask, denoises.
3) Skeletonizes and traces polylines.
4) Merges fragments based on proximity and tangent alignment.
5) Smooths/simplifies and fits cubic Béziers.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Tuple

import numpy as np
import torch
from skimage import filters, morphology

import pydiffvg
from .config import PreconditionConfig
from .skeleton import skeletonize_edges, skeleton_to_polylines
from .vectorize import (
    _catmull_rom_to_beziers,
    _rdp,
    _smooth_polyline,
)


def _simple_kmeans(flat: np.ndarray, k: int, iters: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """Tiny KMeans fallback to avoid heavy dependencies."""
    n = flat.shape[0]
    # pick initial centers uniformly
    idx = np.linspace(0, n - 1, num=k, dtype=int)
    centers = flat[idx].copy()
    labels = np.zeros(n, dtype=np.int32)
    for _ in range(max(1, iters)):
        dists = ((flat[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = np.argmin(dists, axis=1)
        for c in range(k):
            mask = labels == c
            if not np.any(mask):
                continue
            centers[c] = flat[mask].mean(axis=0)
    return centers, labels


def _quantize_palette(image: np.ndarray, cfg: PreconditionConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Return (palette, labels) where palette is Kx3 in [0,1], labels is HxW ints."""
    h, w, _ = image.shape
    flat = image.reshape(-1, 3)
    if cfg.num_colors <= 1:
        # single-color: use otsu to binarize; palette derived from mean
        gray = image.mean(axis=2)
        thresh = filters.threshold_otsu(gray)
        labels = (gray >= thresh).astype(np.int32).reshape(h, w)
        palette = np.array([[gray.mean()], [gray.mean()]], dtype=np.float32)
        palette = np.clip(np.repeat(palette, 3, axis=1), 0.0, 1.0)
        return palette, labels

    if cfg.palette_mode == "fixed" and cfg.palette_colors:
        palette = np.array(cfg.palette_colors, dtype=np.float32)
        dists = ((flat[:, None, :] - palette[None, :, :]) ** 2).sum(axis=2)
        labels = np.argmin(dists, axis=1)
    else:
        k = max(1, cfg.num_colors)
        palette, labels = _simple_kmeans(flat, k)
    return palette.astype(np.float32), labels.reshape(h, w)


def _merge_polylines(polys: List[List[Tuple[int, int]]], cfg: PreconditionConfig) -> List[List[Tuple[int, int]]]:
    """Greedy merge of polylines when endpoints are close and tangents align."""
    if not polys:
        return []

    def endpoint(p: List[Tuple[int, int]], head: bool):
        return np.array(p[0 if head else -1], dtype=np.float32)

    def tangent(p: List[Tuple[int, int]], head: bool):
        pts = np.array(p, dtype=np.float32)
        if head:
            v = pts[min(len(p) - 1, 1)] - pts[0]
        else:
            v = pts[-1] - pts[-2 if len(p) > 1 else -1]
        n = np.linalg.norm(v) + 1e-8
        return v / n

    used = [False] * len(polys)
    merged: List[List[Tuple[int, int]]] = []
    max_d2 = cfg.merge_distance * cfg.merge_distance
    cos_thresh = math.cos(math.radians(cfg.merge_angle_deg))

    for i, p in enumerate(polys):
        if used[i]:
            continue
        chain = list(p)
        used[i] = True
        changed = True
        while changed:
            changed = False
            head_pt = endpoint(chain, head=True)
            tail_pt = endpoint(chain, head=False)
            head_tan = tangent(chain, head=True)
            tail_tan = tangent(chain, head=False)
            best = None
            for j, q in enumerate(polys):
                if used[j] or i == j or len(q) < 2:
                    continue
                q_head = endpoint(q, True)
                q_tail = endpoint(q, False)
                # Try connecting tail->head
                for tail_first, reverse_q in ((True, False), (True, True), (False, False), (False, True)):
                    a_pt = tail_pt if tail_first else head_pt
                    a_tan = tail_tan if tail_first else head_tan
                    q_start = q_tail if reverse_q else q_head
                    q_tan = tangent(q[::-1] if reverse_q else q, head=True)
                    d2 = float(np.sum((a_pt - q_start) ** 2))
                    if d2 > max_d2:
                        continue
                    cosang = float(np.dot(a_tan, q_tan))
                    if cosang < cos_thresh:
                        continue
                    score = d2 - cosang * cfg.merge_distance
                    best = (score, j, tail_first, reverse_q)
            if best is not None:
                _, j, tail_first, reverse_q = best
                q = polys[j][::-1] if reverse_q else polys[j]
                if tail_first:
                    chain.extend(q)
                else:
                    chain = q + chain
                used[j] = True
                changed = True
        merged.append(chain)
    return merged


def _paths_from_polylines(polys: List[List[Tuple[int, int]]], image_rgb: np.ndarray, cfg: PreconditionConfig, device: torch.device) -> Tuple[List[pydiffvg.Path], List[pydiffvg.ShapeGroup]]:
    shapes: List[pydiffvg.Path] = []
    groups: List[pydiffvg.ShapeGroup] = []
    gray = image_rgb[..., :3].mean(axis=2)
    for idx, poly in enumerate(polys):
        pts = np.array([[x, y] for y, x in poly], dtype=np.float32)
        if cfg.smooth_window and cfg.smooth_window > 1:
            pts = _smooth_polyline(pts, cfg.smooth_window)
        pts = _rdp(pts, cfg.simplify_epsilon)
        if pts.shape[0] < 2:
            continue
        is_closed = np.linalg.norm(pts[0] - pts[-1]) < 1.0
        if is_closed:
            pts[-1] = pts[0]
        if cfg.curve_mode.lower() == "bezier":
            pts, cp_arr = _catmull_rom_to_beziers(pts, cfg.catmull_rom_tension, is_closed)
            num_control_points = torch.tensor(cp_arr, dtype=torch.int32, device=device)
        else:
            num_control_points = torch.zeros(pts.shape[0] - 1, dtype=torch.int32, device=device)
        points = torch.tensor(pts, dtype=torch.float32, device=device)
        stroke_width = torch.tensor(cfg.base_stroke_width, dtype=torch.float32, device=device)
        path = pydiffvg.Path(
            num_control_points=num_control_points,
            points=points,
            is_closed=bool(is_closed),
            stroke_width=stroke_width,
            id=f"lineart_path_{idx}",
            use_distance_approx=False,
        )
        shapes.append(path)
        # Color picked upstream; groups will be created by caller
    return shapes, groups


def build_lineart_scene(
    image_rgb: np.ndarray,
    cfg: PreconditionConfig,
    *,
    device: torch.device,
) -> Tuple[List[pydiffvg.Path], List[pydiffvg.ShapeGroup], np.ndarray, np.ndarray]:
    palette, labels = _quantize_palette(image_rgb, cfg)
    all_shapes: List[pydiffvg.Path] = []
    all_groups: List[pydiffvg.ShapeGroup] = []
    next_id = 0
    edge_union = np.zeros(labels.shape, dtype=bool)
    skel_union = np.zeros_like(edge_union)

    for color_idx, color in enumerate(palette):
        mask = labels == color_idx
        if cfg.min_component_area > 0:
            mask = morphology.remove_small_objects(mask, cfg.min_component_area)
        if cfg.morph_open_radius > 0:
            mask = morphology.binary_opening(mask, morphology.disk(cfg.morph_open_radius))
        if cfg.morph_close_radius > 0:
            mask = morphology.binary_closing(mask, morphology.disk(cfg.morph_close_radius))
        edge_union |= mask
        skel = skeletonize_edges(mask)
        skel_union |= skel
        polylines = skeleton_to_polylines(skel, cfg)
        polylines = _merge_polylines(polylines, cfg)
        shapes, _ = _paths_from_polylines(polylines, image_rgb, cfg, device)
        if not shapes:
            continue
        # Assign color and stroke group per color
        if cfg.fixed_stroke_rgba is not None:
            rgba = np.array(cfg.fixed_stroke_rgba, dtype=np.float32)
            rgba = np.clip(rgba, 0.0, 1.0)
            stroke_color = torch.tensor([rgba[0], rgba[1], rgba[2], rgba[3]], dtype=torch.float32, device=device)
        else:
            stroke_color = torch.tensor([color[0], color[1], color[2], 1.0], dtype=torch.float32, device=device)
        for s in shapes:
            s.stroke_width = torch.tensor(cfg.base_stroke_width, dtype=torch.float32, device=device)
            all_shapes.append(s)
            sg = pydiffvg.ShapeGroup(
                shape_ids=torch.tensor([next_id], dtype=torch.int32, device=device),
                fill_color=None,
                stroke_color=stroke_color,
                shape_to_canvas=torch.eye(3, device=device),
                id=f"lineart_group_{next_id}",
            )
            all_groups.append(sg)
            next_id += 1
    return all_shapes, all_groups, edge_union.astype(np.float32), skel_union.astype(np.float32)


__all__ = ["build_lineart_scene"]
