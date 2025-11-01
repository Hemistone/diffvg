"""Tests for :mod:`pydiffvg.vectorizer.edges`."""

from __future__ import annotations

import numpy as np
import pytest

from .utils import load_vectorizer_module

edges = load_vectorizer_module("edges")


sobel_edge_map = edges.sobel_edge_map
canny_edge_map = edges.canny_edge_map


@pytest.fixture(autouse=True)
def _ensure_no_cv2(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the fallback path so tests do not require OpenCV."""

    monkeypatch.setattr(edges, "cv2", None, raising=False)


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((4, 4), dtype=np.float32),
        np.zeros((4, 4, 3), dtype=np.float32),
    ],
)
def test_sobel_edge_map_returns_normalized_float32(image: np.ndarray) -> None:
    edge_map = sobel_edge_map(image)

    assert edge_map.dtype == np.float32
    assert edge_map.shape == image.shape[:2]
    assert edge_map.max() <= 1.0
    assert edge_map.min() >= 0.0


def test_sobel_edge_map_detects_vertical_contrast() -> None:
    image = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    edges = sobel_edge_map(image)

    assert np.isclose(edges[:, 1], edges.max()).all()


def test_canny_edge_map_fallback_produces_binary_like_response() -> None:
    image = np.zeros((5, 5), dtype=np.float32)
    image[:, 2:] = 1.0

    edge_map = canny_edge_map(image, low_threshold=0.05, high_threshold=0.2)

    assert edge_map.dtype == np.float32
    assert edge_map.shape == image.shape
    assert np.isin(np.unique(edge_map), [0.0, 1.0]).all()
