"""Tests for :mod:`pydiffvg.vectorizer.strokes`."""

from __future__ import annotations

import numpy as np
import pytest

from .utils import load_vectorizer_module

strokes = load_vectorizer_module("strokes")

Seed = strokes.Seed
select_seeds = strokes.select_seeds
bresenham_line = strokes.bresenham_line
score_line = strokes.score_line
expand_squiggle = strokes.expand_squiggle
control_points = strokes.control_points


@pytest.mark.parametrize("method", ["nms", "poisson"])
def test_select_seeds_picks_highest_probabilities(method: str) -> None:
    prob_map = np.zeros((5, 5), dtype=np.float32)
    prob_map[1, 1] = 0.9
    prob_map[3, 3] = 0.8

    seeds = select_seeds(prob_map, count=2, method=method, radius=1)

    assert [seed.position for seed in seeds] == [(1, 1), (3, 3)]


def test_bresenham_line_covers_endpoints() -> None:
    points = bresenham_line((0, 0), (0, 3))

    assert points[0] == (0, 0)
    assert points[-1] == (0, 3)
    assert len(points) == 4


def test_score_line_uses_integral_image() -> None:
    field = np.ones((3, 3), dtype=np.float32)
    integral = strokes.integral_image(field)

    score = score_line(integral, (1, 0), (1, 2))

    assert score == pytest.approx(3.0)


def test_expand_squiggle_follows_high_prob_path() -> None:
    prob_map = np.zeros((3, 4), dtype=np.float32)
    prob_map[1] = np.array([0.1, 0.5, 1.0, 2.0], dtype=np.float32)
    seed = Seed((1, 0), 0.1)

    points = expand_squiggle(seed, prob_map, max_length=4, step=1)

    assert points[0] == (1, 0)
    assert points[-1] == (1, 3)
    assert len(points) == 4


def test_control_points_support_multiple_modes() -> None:
    path = [(0, 0), (1, 1), (2, 2)]

    line_ctrl = control_points(path, mode="lines")
    quad_ctrl = control_points(path, mode="quad")
    cubic_ctrl = control_points(path, mode="cubic")

    assert len(line_ctrl) == 2
    assert len(quad_ctrl) == 2
    assert len(cubic_ctrl) == 2
    assert line_ctrl[0][0][0] == pytest.approx(0.0)
    assert quad_ctrl[0][1][0] == pytest.approx(0.5)
    assert cubic_ctrl[0][1][0] == pytest.approx(0.33, rel=1e-2)
