"""Stroke construction helpers used by :func:`pydiffvg.vectorizer.api.vectorize`."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from .tone import integral_image

Point = Tuple[int, int]


@dataclass(slots=True)
class Seed:
    position: Point
    score: float


def select_seeds(prob_map: np.ndarray, *, count: int, method: str = "nms", radius: int = 5) -> List[Seed]:
    """Pick *count* seeds using non-max suppression or Poisson disc sampling."""

    flat_indices = np.argsort(prob_map.ravel())[::-1]
    selected: List[Seed] = []
    taken = np.zeros_like(prob_map, dtype=bool)
    h, w = prob_map.shape
    for idx in flat_indices:
        if len(selected) >= count:
            break
        y = idx // w
        x = idx % w
        if taken[y, x]:
            continue
        selected.append(Seed((y, x), float(prob_map[y, x])))
        if method == "nms":
            y0 = max(y - radius, 0)
            y1 = min(y + radius + 1, h)
            x0 = max(x - radius, 0)
            x1 = min(x + radius + 1, w)
            taken[y0:y1, x0:x1] = True
        elif method == "poisson":
            _mark_poisson(taken, (y, x), radius)
        else:  # pragma: no cover - defensive programming
            raise ValueError(f"Unknown seed selection method: {method}")
    return selected


def bresenham_line(p0: Point, p1: Point) -> List[Point]:
    """Return integer coordinates along the Bresenham line between *p0* and *p1*."""

    (y0, x0), (y1, x1) = p0, p1
    dy = abs(y1 - y0)
    dx = abs(x1 - x0)
    sy = 1 if y0 < y1 else -1
    sx = 1 if x0 < x1 else -1
    err = dx - dy
    points: List[Point] = []
    while True:
        points.append((y0, x0))
        if y0 == y1 and x0 == x1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return points


def score_line(prob_integral: np.ndarray, p0: Point, p1: Point) -> float:
    """Score a line by summing probabilities along its path using an integral image."""

    total = 0.0
    for y, x in bresenham_line(p0, p1):
        total += _box_sum(prob_integral, y, x, 1, 1)
    return total


def expand_squiggle(seed: Seed, prob_map: np.ndarray, *, max_length: int = 20, step: int = 3) -> List[Point]:
    """Greedily extend from *seed* following the strongest integral score."""

    integral = integral_image(prob_map)
    points = [seed.position]
    current = seed.position
    for _ in range(max_length - 1):
        candidates = _neighbors(current, step, prob_map.shape)
        if not candidates:
            break
        best = max(candidates, key=lambda pt: score_line(integral, current, pt))
        if best == current:
            break
        points.append(best)
        current = best
    return points


def control_points(points: Sequence[Point], *, mode: str = "lines") -> List[Tuple[Tuple[float, float], ...]]:
    """Return control points for ``lines``, ``quad`` or ``cubic`` modes."""

    if mode == "lines":
        return [((float(x0), float(y0)), (float(x1), float(y1))) for (y0, x0), (y1, x1) in zip(points[:-1], points[1:])]
    if mode == "quad":
        result = []
        for (y0, x0), (y1, x1) in zip(points[:-1], points[1:]):
            mid = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
            result.append(((float(x0), float(y0)), mid, (float(x1), float(y1))))
        return result
    if mode == "cubic":
        result = []
        for idx in range(len(points) - 1):
            y0, x0 = points[idx]
            y1, x1 = points[idx + 1]
            c1 = (float(x0 + (x1 - x0) * 0.33), float(y0 + (y1 - y0) * 0.33))
            c2 = (float(x0 + (x1 - x0) * 0.66), float(y0 + (y1 - y0) * 0.66))
            result.append(((float(x0), float(y0)), c1, c2, (float(x1), float(y1))))
        return result
    raise ValueError(f"Unsupported mode: {mode}")


def _neighbors(point: Point, step: int, shape: Tuple[int, int]) -> List[Point]:
    y, x = point
    h, w = shape
    neighbors: List[Point] = []
    for dy in range(-step, step + 1):
        for dx in range(-step, step + 1):
            if dy == 0 and dx == 0:
                continue
            ny = int(np.clip(y + dy, 0, h - 1))
            nx = int(np.clip(x + dx, 0, w - 1))
            candidate = (ny, nx)
            if candidate not in neighbors:
                neighbors.append(candidate)
    return neighbors


def _box_sum(integral: np.ndarray, y: int, x: int, height: int, width: int) -> float:
    y0 = max(y, 0)
    x0 = max(x, 0)
    y1 = min(y + height - 1, integral.shape[0] - 1)
    x1 = min(x + width - 1, integral.shape[1] - 1)
    a = integral[y1, x1]
    b = integral[y0 - 1, x1] if y0 > 0 else 0.0
    c = integral[y1, x0 - 1] if x0 > 0 else 0.0
    d = integral[y0 - 1, x0 - 1] if y0 > 0 and x0 > 0 else 0.0
    return float(a - b - c + d)


def _mark_poisson(mask: np.ndarray, seed: Point, radius: int) -> None:
    y, x = seed
    h, w = mask.shape
    for j in range(-radius, radius + 1):
        for i in range(-radius, radius + 1):
            if math.hypot(j, i) <= radius:
                ny = int(np.clip(y + j, 0, h - 1))
                nx = int(np.clip(x + i, 0, w - 1))
                mask[ny, nx] = True
