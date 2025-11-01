"""Tests for :mod:`pydiffvg.vectorizer.api`."""

from __future__ import annotations

import numpy as np
import pytest

from .utils import load_vectorizer_module

vectorizer_api = load_vectorizer_module("api")

VectorDoc = vectorizer_api.VectorDoc
vectorize = vectorizer_api.vectorize
_is_numpy_image = vectorizer_api._is_numpy_image
_is_torch_image = vectorizer_api._is_torch_image


def test_vectorize_numpy_array_returns_document_with_canvas_size() -> None:
    image = np.zeros((10, 20, 3), dtype=np.float32)

    doc = vectorize(image, num_layers=2)

    assert isinstance(doc, VectorDoc)
    assert doc.canvas_size == (20, 10)
    assert len(doc.layers) == 2


def test_vectorize_torch_tensor_returns_document_with_canvas_size() -> None:
    torch = pytest.importorskip("torch")

    image = torch.zeros((1, 3, 32, 16), dtype=torch.float32)

    doc = vectorize(image)

    assert isinstance(doc, VectorDoc)
    assert doc.canvas_size == (16, 32)
    assert len(doc.layers) == 1


def test_vectorize_rejects_invalid_num_layers() -> None:
    image = np.zeros((4, 4), dtype=np.float32)

    with pytest.raises(ValueError):
        vectorize(image, num_layers=0)


def test_vectorize_requires_image_like_input() -> None:
    with pytest.raises(TypeError):
        vectorize(object())


@pytest.mark.parametrize(
    "image_factory, guard",
    [
        (lambda: np.zeros((2, 2), dtype=np.float32), _is_numpy_image),
        (lambda: pytest.importorskip("torch").zeros((1, 1, 2, 2)), _is_torch_image),
    ],
)
def test_image_guards_only_accept_supported_types(image_factory, guard) -> None:
    valid_image = image_factory()
    assert guard(valid_image)

    assert not guard(object())


def test_vectorize_requires_two_dimensional_numpy_images() -> None:
    image = np.zeros((5,), dtype=np.float32)

    with pytest.raises(ValueError):
        vectorize(image)
