"""Convert skeleton polylines into diffvg Path/ShapeGroup pairs."""

from __future__ import annotations

import math
import numpy as np
import torch

import pydiffvg
from .config import PreconditionConfig


def _rdp(points: np.ndarray, epsilon: float) -> np.ndarray:
    """Ramer–Douglas–Peucker simplification."""
    if points.shape[0] < 3:
        return points

    start, end = points[0], points[-1]
    vec = end - start
    norm = np.linalg.norm(vec) + 1e-8
    distances = np.abs(np.cross(vec, points - start) / norm)

    idx = int(np.argmax(distances))
    dmax = distances[idx]
    if dmax < epsilon:
        return np.vstack([start, end])
    left = _rdp(points[: idx + 1], epsilon)
    right = _rdp(points[idx:], epsilon)
    return np.vstack([left[:-1], right])


def _smooth_polyline(points: np.ndarray, window: int) -> np.ndarray:
    """Simple moving-average smoothing over coordinates."""
    if window <= 1 or points.shape[0] < 3:
        return points
    w = int(max(2, window))
    w = min(w, points.shape[0])
    half = w // 2
    padded = np.pad(points, ((half, half), (0, 0)), mode="edge")
    smoothed = np.empty_like(points)
    for i in range(points.shape[0]):
        smoothed[i] = padded[i : i + w].mean(axis=0)
    return smoothed


def merge_polylines(
    polys: list[list[tuple[int, int]]],
    cfg: PreconditionConfig,
    *,
    enabled: bool = True,
) -> list[list[tuple[int, int]]]:
    """Greedy merge of polylines when endpoints are close and tangents align."""
    if not enabled or not polys or cfg.merge_distance <= 0:
        return polys

    def endpoint(p: list[tuple[int, int]], head: bool) -> np.ndarray:
        return np.array(p[0 if head else -1], dtype=np.float32)

    def tangent(p: list[tuple[int, int]], head: bool) -> np.ndarray:
        pts = np.array(p, dtype=np.float32)
        if head:
            v = pts[min(len(p) - 1, 1)] - pts[0]
        else:
            v = pts[-1] - pts[-2 if len(p) > 1 else -1]
        n = np.linalg.norm(v) + 1e-8
        return v / n

    used = [False] * len(polys)
    merged: list[list[tuple[int, int]]] = []
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
                # Try connecting tail->head (and variations)
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


def _catmull_rom_to_beziers(points: np.ndarray, tension: float, is_closed: bool) -> tuple[np.ndarray, np.ndarray]:
    """Convert a polyline to cubic Bézier control points via Catmull-Rom tangents."""
    n = points.shape[0]
    if n < 2:
        return points, np.array([], dtype=np.int32)

    def _p(idx: int) -> np.ndarray:
        if is_closed:
            return points[idx % n]
        return points[min(max(idx, 0), n - 1)]

    out: list[np.ndarray] = [points[0]]
    cp_counts: list[int] = []
    num_segments = n if is_closed else n - 1
    for i in range(num_segments):
        p0, p1, p2, p3 = _p(i - 1), _p(i), _p(i + 1), _p(i + 2)
        c1 = p1 + (p2 - p0) * (tension / 6.0)
        c2 = p2 - (p3 - p1) * (tension / 6.0)
        out.extend([c1, c2, p2])
        cp_counts.append(2)
    return np.stack(out, axis=0), np.array(cp_counts, dtype=np.int32)


def _score_polyline(poly: list[tuple[int, int]], gray: np.ndarray, mode: str) -> float:
    pts = np.array(poly, dtype=np.int32)
    ys = np.clip(pts[:, 0], 0, gray.shape[0] - 1)
    xs = np.clip(pts[:, 1], 0, gray.shape[1] - 1)
    luminance = gray[ys, xs]
    darkness = 1.0 - float(luminance.mean())
    length = float(len(poly))
    if mode == "length":
        return length
    return darkness * length


def polylines_to_paths(
    polylines: list[list[tuple[int, int]]],
    image_rgb: np.ndarray,
    cfg: PreconditionConfig,
    canvas_w: int,
    canvas_h: int,
    device: torch.device,
    *,
    palette_canvas_w: int | None = None,
    palette_canvas_h: int | None = None,
) -> tuple[list[pydiffvg.Path], list[pydiffvg.ShapeGroup]]:
    """Convert traced polylines to diffvg shapes."""
    _ = (canvas_w, canvas_h)  # reserved for future scaling/normalization
    palette_canvas_w = palette_canvas_w or canvas_w
    palette_canvas_h = palette_canvas_h or canvas_h
    palette = cfg.palette
    if image_rgb.ndim != 3 or image_rgb.shape[2] < 3:
        raise ValueError("image_rgb must be HxWx3")

    gray = image_rgb[..., :3].mean(axis=2)

    scored = [
        (_score_polyline(poly, gray, cfg.sort_by), poly)
        for poly in polylines
    ]
    scored.sort(key=lambda t: t[0], reverse=True)
    if cfg.max_paths is not None:
        scored = scored[: cfg.max_paths]

    shapes: list[pydiffvg.Path] = []
    groups: list[pydiffvg.ShapeGroup] = []

    for idx, (_, poly) in enumerate(scored):
        pts = np.array([[x, y] for y, x in poly], dtype=np.float32)
        pts = _rdp(pts, cfg.simplify_epsilon)
        if cfg.smooth_window and cfg.smooth_window > 1:
            pts = _smooth_polyline(pts, cfg.smooth_window)
        if pts.shape[0] < 2:
            continue

        is_closed = np.linalg.norm(pts[0] - pts[-1]) < 1.0
        if cfg.force_open_paths:
            is_closed = False
        elif is_closed:
            pts[-1] = pts[0]

        if cfg.curve_mode.lower() == "bezier":
            pts, cp_arr = _catmull_rom_to_beziers(pts, cfg.catmull_rom_tension, is_closed)
            num_control_points = torch.tensor(cp_arr, dtype=torch.int32, device=device)
        else:
            num_segments = pts.shape[0] - 1
            num_control_points = torch.zeros(num_segments, dtype=torch.int32, device=device)
        points = torch.tensor(pts, dtype=torch.float32, device=device)

        ys = np.clip(pts[:, 1].astype(int), 0, image_rgb.shape[0] - 1)
        xs = np.clip(pts[:, 0].astype(int), 0, image_rgb.shape[1] - 1)
        luminance = gray[ys, xs]
        darkness = 1.0 - float(luminance.mean())
        width = cfg.base_width + (cfg.max_width - cfg.base_width) * (darkness ** cfg.width_gamma)
        if (cfg.mode or "xdog").strip().lower() == "xdog":
            width *= float(cfg.xdog_width_scale)
        entry = None
        if palette is not None:
            entry, width, _ = palette.entry_for_width(width, palette_canvas_w, palette_canvas_h)
        stroke_width = torch.tensor(width, dtype=torch.float32, device=device)

        path = pydiffvg.Path(
            num_control_points=num_control_points,
            points=points,
            is_closed=bool(is_closed),
            stroke_width=stroke_width,
            id=f"pre_path_{idx}",
            use_distance_approx=False,
        )
        shapes.append(path)

        if entry is not None:
            rgba = np.array(entry.color_rgba, dtype=np.float32)
            stroke_color = torch.tensor([rgba[0], rgba[1], rgba[2], rgba[3]], dtype=torch.float32, device=device)
        else:
            if cfg.sample_color:
                rgb = image_rgb[ys, xs].mean(axis=0).astype(np.float32)
                rgb = np.clip(rgb, 0.0, 1.0)
            else:
                rgb = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            stroke_color = torch.tensor([rgb[0], rgb[1], rgb[2], 1.0], dtype=torch.float32, device=device)

        group = pydiffvg.ShapeGroup(
            shape_ids=torch.tensor([len(shapes) - 1], dtype=torch.int32, device=device),
            fill_color=None,
            stroke_color=stroke_color,
            shape_to_canvas=torch.eye(3, device=device),
            id=f"pre_group_{idx}",
        )
        groups.append(group)

    return shapes, groups


__all__ = ["merge_polylines", "polylines_to_paths"]
