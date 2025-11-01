"""Tests for :mod:`pydiffvg.vectorizer.bezier_fit`."""

from __future__ import annotations

import numpy as np
import pytest

from .utils import load_vectorizer_module

bezier_fit = load_vectorizer_module("bezier_fit")


douglas_peucker = bezier_fit.douglas_peucker
fit_quadratic = bezier_fit.fit_quadratic
fit_cubic = bezier_fit.fit_cubic


def test_douglas_peucker_reduces_points_on_straight_line() -> None:
    points = [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)]

    simplified = douglas_peucker(points, epsilon=1e-3)

    assert simplified == [(0.0, 0.0), (1.0, 1.0)]


def test_fit_quadratic_returns_endpoints() -> None:
    points = [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)]

    ctrl = fit_quadratic(points)

    assert np.allclose(ctrl[0], points[0])
    assert np.allclose(ctrl[-1], points[-1])


def test_fit_cubic_handles_simple_curve() -> None:
    # Sample points from y = x^2 between 0 and 1
    xs = np.linspace(0.0, 1.0, 5)
    points = list(zip(xs, xs ** 2))

    ctrl = fit_cubic(points)

    assert np.allclose(ctrl[0], points[0])
    assert np.allclose(ctrl[-1], points[-1])
    # Control points should reflect convex curve (positive y)
    assert ctrl[1][1] > 0.0
    assert ctrl[2][1] > 0.0
