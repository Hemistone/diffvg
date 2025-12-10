"""Skeletonization and polyline extraction from binary edges."""

from __future__ import annotations

import numpy as np
from skimage import morphology

from .config import PreconditionConfig

# 8-connected neighbors
_NEIGHBORS = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)


def skeletonize_edges(edge_mask: np.ndarray) -> np.ndarray:
    """Return a 1-pixel-wide skeleton from a boolean edge mask."""
    binary = edge_mask.astype(bool)
    return morphology.skeletonize(binary)


def _neighbors(skel: np.ndarray, y: int, x: int):
    h, w = skel.shape
    for dy, dx in _NEIGHBORS:
        ny, nx = y + dy, x + dx
        if 0 <= ny < h and 0 <= nx < w and skel[ny, nx]:
            yield ny, nx


def skeleton_to_polylines(skel: np.ndarray, cfg: PreconditionConfig) -> list[list[tuple[int, int]]]:
    """Trace polylines along a skeleton graph."""
    skel = skel.astype(bool)
    visited = np.zeros_like(skel, dtype=bool)

    ys, xs = np.nonzero(skel)
    # int16 avoids overflow when sorting by negative degree values
    degree = np.zeros_like(skel, dtype=np.int16)
    for y, x in zip(ys, xs):
        degree[y, x] = sum(1 for _ in _neighbors(skel, y, x))

    endpoints = [(y, x) for y, x in zip(ys, xs) if degree[y, x] <= 1]
    polylines: list[list[tuple[int, int]]] = []

    def trace(start: tuple[int, int]) -> list[tuple[int, int]]:
        path = [start]
        visited[start] = True
        current = start
        while len(path) < cfg.max_path_length:
            neigh = [n for n in _neighbors(skel, *current) if not visited[n]]
            if not neigh:
                break
            # Prefer continuing through straight sections (degree==2) to avoid
            # branching too early.
            neigh.sort(key=lambda n: (degree[n] != 2, -degree[n]))
            nxt = neigh[0]
            path.append(nxt)
            visited[nxt] = True
            current = nxt
        return path

    for start in endpoints:
        if visited[start]:
            continue
        path = trace(start)
        if len(path) >= cfg.min_path_length:
            polylines.append(path)

    # Cover loops or leftover components with no endpoints
    for y, x in zip(ys, xs):
        if visited[y, x]:
            continue
        path = trace((y, x))
        if len(path) >= cfg.min_path_length:
            polylines.append(path)

    return polylines


__all__ = ["skeletonize_edges", "skeleton_to_polylines"]
