"""Simple least squares fitting of point sequences to Bézier curves."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

Point = Tuple[float, float]


def douglas_peucker(points: Sequence[Point], epsilon: float) -> List[Point]:
    """Return a simplified version of *points* using Douglas–Peucker."""

    if len(points) < 3:
        return list(points)

    start, end = np.array(points[0]), np.array(points[-1])
    distances = []
    for idx, point in enumerate(points[1:-1], start=1):
        p = np.array(point)
        if np.allclose(start, end):
            dist = np.linalg.norm(p - start)
        else:
            segment = end - start
            offset = start - p
            cross_mag = abs(segment[0] * offset[1] - segment[1] * offset[0])
            dist = cross_mag / np.linalg.norm(segment)
        distances.append((dist, idx))
    max_dist, max_idx = max(distances, key=lambda item: item[0])
    if max_dist > epsilon:
        left = douglas_peucker(points[: max_idx + 1], epsilon)
        right = douglas_peucker(points[max_idx:], epsilon)
        return left[:-1] + right
    return [points[0], points[-1]]


def fit_quadratic(points: Sequence[Point]) -> Tuple[Point, Point, Point]:
    """Fit *points* to a quadratic Bézier curve returning its control points."""

    return _fit_bezier(points, degree=2)


def fit_cubic(points: Sequence[Point]) -> Tuple[Point, Point, Point, Point]:
    """Fit *points* to a cubic Bézier curve returning its control points."""

    return _fit_bezier(points, degree=3)


def _fit_bezier(points: Sequence[Point], *, degree: int) -> Tuple[Point, ...]:
    if len(points) < degree + 1:
        raise ValueError("Not enough points to fit Bézier curve")

    pts = np.array(points, dtype=np.float64)
    t = np.linspace(0.0, 1.0, len(points), dtype=np.float64)
    basis = _bezier_basis(t, degree)
    # Solve for control points: basis @ ctrl = pts => ctrl = lstsq(basis, pts)
    ctrl, *_ = np.linalg.lstsq(basis, pts, rcond=None)
    return tuple(map(tuple, ctrl.astype(np.float32)))  # type: ignore[return-value]


def _bezier_basis(t: np.ndarray, degree: int) -> np.ndarray:
    from math import comb

    basis = np.empty((t.size, degree + 1), dtype=np.float64)
    for i in range(degree + 1):
        basis[:, i] = comb(degree, i) * (t ** i) * ((1 - t) ** (degree - i))
    return basis
