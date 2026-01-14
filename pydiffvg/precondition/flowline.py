"""Flow-guided path growing for preconditioning."""

from __future__ import annotations

import math
from typing import Iterable, List, Tuple

import numpy as np
import torch
from skimage import filters

from .config import PreconditionConfig
from .teed import teed_edge_strength
from .xdog import xdog_edges


def _safe_quantile(values: np.ndarray, q: float) -> float:
    q = max(0.0, min(1.0, float(q)))
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return 0.0
    try:
        t = torch.from_numpy(arr).reshape(-1)
        return float(torch.quantile(t, q).item())
    except Exception:
        flat = np.sort(arr.reshape(-1))
        if flat.size == 0:
            return 0.0
        idx = int(round((flat.size - 1) * q))
        idx = max(0, min(flat.size - 1, idx))
        return float(flat[idx])


def _to_gray(image_rgb: np.ndarray) -> np.ndarray:
    if image_rgb.ndim != 3 or image_rgb.shape[2] < 3:
        raise ValueError("image_rgb must be HxWx3")
    rgb = image_rgb[..., :3].astype(np.float32)
    if rgb.max() > 1.5:
        rgb = rgb / 255.0
    return rgb.mean(axis=2)


def _compute_flow_field(gray: np.ndarray, cfg: PreconditionConfig) -> tuple[np.ndarray, np.ndarray]:
    gx = filters.sobel_h(gray)
    gy = filters.sobel_v(gray)
    jxx = gx * gx
    jxy = gx * gy
    jyy = gy * gy

    sigma = max(0.0, float(cfg.flow_field_sigma))
    iters = max(1, int(cfg.flow_field_iters))
    if sigma > 0.0:
        for _ in range(iters):
            jxx = filters.gaussian(jxx, sigma=sigma, mode="nearest")
            jxy = filters.gaussian(jxy, sigma=sigma, mode="nearest")
            jyy = filters.gaussian(jyy, sigma=sigma, mode="nearest")

    theta = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)
    tx = -np.sin(theta)
    ty = np.cos(theta)
    mag = np.sqrt(tx * tx + ty * ty) + 1e-6
    tx /= mag
    ty /= mag
    return tx, ty


def _bilinear_sample(arr: np.ndarray, x: float, y: float) -> float:
    h, w = arr.shape
    if x < 0.0 or y < 0.0 or x > w - 1 or y > h - 1:
        return 0.0
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    dx = x - x0
    dy = y - y0
    v00 = float(arr[y0, x0])
    v10 = float(arr[y0, x1])
    v01 = float(arr[y1, x0])
    v11 = float(arr[y1, x1])
    v0 = v00 * (1.0 - dx) + v10 * dx
    v1 = v01 * (1.0 - dx) + v11 * dx
    return v0 * (1.0 - dy) + v1 * dy


def _trace_single(
    seed: tuple[int, int],
    strength: np.ndarray,
    tx: np.ndarray,
    ty: np.ndarray,
    cfg: PreconditionConfig,
    *,
    direction: float,
) -> list[tuple[int, int]]:
    h, w = strength.shape
    max_len = max(1, int(cfg.flow_max_len))
    min_strength = float(cfg.flow_min_strength)
    step = float(cfg.flow_step_px)
    curvature = float(cfg.flow_curvature_deg)
    curvature_rad = math.radians(curvature) if curvature > 0.0 else None

    x = float(seed[1])
    y = float(seed[0])
    prev_dir: tuple[float, float] | None = None
    path: list[tuple[int, int]] = []
    visited = set()

    for _ in range(max_len):
        if x < 0.0 or y < 0.0 or x > w - 1 or y > h - 1:
            break
        s = _bilinear_sample(strength, x, y)
        if s < min_strength:
            break
        vx = _bilinear_sample(tx, x, y)
        vy = _bilinear_sample(ty, x, y)
        norm = math.hypot(vx, vy)
        if norm < 1e-6:
            break
        vx /= norm
        vy /= norm
        if prev_dir is not None:
            if vx * prev_dir[0] + vy * prev_dir[1] < 0.0:
                vx = -vx
                vy = -vy
            if curvature_rad is not None:
                dot = max(-1.0, min(1.0, vx * prev_dir[0] + vy * prev_dir[1]))
                angle = math.acos(dot)
                if angle > curvature_rad:
                    break
        iy = int(round(y))
        ix = int(round(x))
        if iy < 0 or ix < 0 or iy >= h or ix >= w:
            break
        if (iy, ix) in visited:
            break
        visited.add((iy, ix))
        path.append((iy, ix))
        prev_dir = (vx, vy)
        x += direction * step * vx
        y += direction * step * vy

    return path


def _seed_threshold(strength: np.ndarray, cfg: PreconditionConfig) -> float:
    mode = (cfg.flow_seed_mode or "quantile").strip().lower()
    if mode == "fixed":
        return float(cfg.flow_seed_threshold)
    if mode != "quantile":
        raise ValueError(f"Unsupported flow_seed_mode '{cfg.flow_seed_mode}'. Choose from: quantile, fixed")
    return _safe_quantile(strength, cfg.flow_seed_quantile)


def _disk_offsets(radius: int) -> list[tuple[int, int]]:
    if radius <= 0:
        return [(0, 0)]
    offsets: list[tuple[int, int]] = []
    r2 = radius * radius
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy * dy + dx * dx <= r2:
                offsets.append((dy, dx))
    return offsets


def _edge_strength(image_rgb: np.ndarray, cfg: PreconditionConfig, *, device: torch.device) -> np.ndarray:
    backend = (cfg.flow_edge_backend or "teed").strip().lower()
    if backend == "teed":
        return teed_edge_strength(image_rgb, cfg, device=device)
    if backend == "xdog":
        edges = xdog_edges(image_rgb, cfg)
        return edges.astype(np.float32)
    raise ValueError(f"Unsupported flow_edge_backend '{cfg.flow_edge_backend}'. Choose from: teed, xdog")


def flowline_polylines(
    image_rgb: np.ndarray,
    cfg: PreconditionConfig,
    *,
    device: torch.device,
) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    """Return (edge_strength, polylines) using flow-guided tracing."""
    strength = _edge_strength(image_rgb, cfg, device=device)
    gray = _to_gray(image_rgb)
    tx, ty = _compute_flow_field(gray, cfg)

    thr = _seed_threshold(strength, cfg)
    ys, xs = np.nonzero(strength >= thr)
    if ys.size == 0:
        return strength, []

    vals = strength[ys, xs]
    order = np.argsort(vals)[::-1]
    seeds = [(int(ys[i]), int(xs[i])) for i in order]

    max_paths = cfg.max_paths if cfg.max_paths is not None else None
    min_len = max(1, int(cfg.flow_min_len))
    min_strength = float(cfg.flow_min_strength)
    coverage_decay = float(cfg.flow_coverage_decay)
    coverage_radius = max(int(cfg.flow_coverage_radius), int(cfg.flow_min_seed_dist))
    offsets = _disk_offsets(coverage_radius)
    strength_work = strength.copy()

    polylines: list[list[tuple[int, int]]] = []
    for seed in seeds:
        y, x = seed
        if y < 0 or x < 0 or y >= strength_work.shape[0] or x >= strength_work.shape[1]:
            continue
        if strength_work[y, x] < max(min_strength, thr):
            continue

        fwd = _trace_single(seed, strength_work, tx, ty, cfg, direction=1.0)
        bwd = _trace_single(seed, strength_work, tx, ty, cfg, direction=-1.0)
        if bwd:
            bwd = list(reversed(bwd))
        if fwd and bwd:
            poly = bwd + fwd[1:]
        else:
            poly = fwd or bwd
        if len(poly) < min_len:
            continue

        polylines.append(poly)
        for py, px in poly:
            for dy, dx in offsets:
                yy = py + dy
                xx = px + dx
                if 0 <= yy < strength_work.shape[0] and 0 <= xx < strength_work.shape[1]:
                    strength_work[yy, xx] *= coverage_decay
        if max_paths is not None and len(polylines) >= max_paths:
            break

    return strength, polylines


__all__ = ["flowline_polylines"]
