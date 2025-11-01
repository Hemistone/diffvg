"""Tests for :mod:`pydiffvg.vectorizer.tone`."""

from __future__ import annotations

import numpy as np
import pytest

from .utils import load_vectorizer_module

tone = load_vectorizer_module("tone")

brightness_map = tone.brightness_map
combine_brightness_and_edges = tone.combine_brightness_and_edges
integral_image = tone.integral_image
seed_probability_map = tone.seed_probability_map


def test_brightness_map_inverts_normalized_input() -> None:
    image = np.array(
        [
            [0.0, 0.5],
            [0.75, 1.0],
        ],
        dtype=np.float32,
    )

    bright = brightness_map(image)

    expected = np.array([[1.0, 0.5], [0.25, 0.0]], dtype=np.float32)
    np.testing.assert_allclose(bright, expected)


def test_combine_brightness_and_edges_normalizes_result() -> None:
    bright = np.array([[0.0, 0.2], [0.4, 0.6]], dtype=np.float32)
    edges = np.array([[0.6, 0.4], [0.2, 0.0]], dtype=np.float32)

    combined = combine_brightness_and_edges(bright, edges, edge_weight=0.75)

    assert combined.dtype == np.float32
    assert np.isclose(combined.max(), 1.0)
    assert np.isclose(combined.min(), 0.0)


def test_integral_image_matches_manual_sum() -> None:
    field = np.arange(1, 10, dtype=np.float32).reshape(3, 3)

    integral = integral_image(field)

    assert integral[1, 1] == pytest.approx(field[:2, :2].sum())
    assert integral[-1, -1] == pytest.approx(field.sum())


def test_seed_probability_map_combines_residual_and_edges() -> None:
    image = np.zeros((3, 3), dtype=np.float32)
    image[:, 1:] = 1.0
    residual = np.eye(3, dtype=np.float32)

    prob = seed_probability_map(image, edge_weight=0.5, residual=residual)

    assert prob.shape == image.shape
    assert np.isclose(prob.sum(), 1.0)
    assert (prob > 0).all()
