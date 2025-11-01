"""Residual map utilities used to progressively cover the raster image."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np


@dataclass(slots=True)
class ResidualKernelCache:
    """Cache reusable circular kernels keyed by radius."""

    kernels: dict[int, np.ndarray]

    def __init__(self) -> None:
        self.kernels = {}

    def get(self, radius: int) -> np.ndarray:
        if radius not in self.kernels:
            self.kernels[radius] = _circular_kernel(radius)
        return self.kernels[radius]


def update_residual_map(residual: np.ndarray, points: Iterable[Tuple[int, int]], *, radius: int = 3, cache: ResidualKernelCache | None = None) -> np.ndarray:
    """Return a copy of *residual* with a circular region erased around *points*."""

    if radius < 1:
        return residual.copy()
    if cache is None:
        cache = ResidualKernelCache()
    kernel = cache.get(radius)
    h, w = residual.shape
    updated = residual.copy()
    kr = kernel.shape[0] // 2
    for y, x in points:
        y0 = max(y - kr, 0)
        x0 = max(x - kr, 0)
        y1 = min(y + kr + 1, h)
        x1 = min(x + kr + 1, w)
        ky0 = kr - (y - y0)
        kx0 = kr - (x - x0)
        ky1 = ky0 + (y1 - y0)
        kx1 = kx0 + (x1 - x0)
        updated[y0:y1, x0:x1] *= 1.0 - kernel[ky0:ky1, kx0:kx1]
    return updated


def reuse_sequence(sequence: List[Tuple[int, int]], new_points: Iterable[Tuple[int, int]], *, max_length: int) -> List[Tuple[int, int]]:
    """Reuse the prefix of *sequence* and append *new_points* bounded by *max_length*."""

    reused = list(sequence)
    for pt in new_points:
        if len(reused) >= max_length:
            break
        reused.append(pt)
    return reused


def _circular_kernel(radius: int) -> np.ndarray:
    size = radius * 2 + 1
    kernel = np.zeros((size, size), dtype=np.float32)
    cy = cx = radius
    for y in range(size):
        for x in range(size):
            if (y - cy) ** 2 + (x - cx) ** 2 <= radius * radius:
                kernel[y, x] = 1.0
    return kernel
