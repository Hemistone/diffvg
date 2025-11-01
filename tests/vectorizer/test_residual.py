"""Tests for :mod:`pydiffvg.vectorizer.residual`."""

from __future__ import annotations

import numpy as np
import pytest

from .utils import load_vectorizer_module

residual_mod = load_vectorizer_module("residual")

ResidualKernelCache = residual_mod.ResidualKernelCache
update_residual_map = residual_mod.update_residual_map
reuse_sequence = residual_mod.reuse_sequence


def test_update_residual_map_erases_neighbourhood() -> None:
    residual = np.ones((7, 7), dtype=np.float32)
    points = [(3, 3)]

    updated = update_residual_map(residual, points, radius=2, cache=ResidualKernelCache())

    assert updated[3, 3] == pytest.approx(0.0)
    assert updated.sum() < residual.sum()
    assert np.all(updated >= 0.0)


def test_reuse_sequence_truncates_to_max_length() -> None:
    sequence = [(0, 0), (1, 1)]
    new_points = [(2, 2), (3, 3)]

    result = reuse_sequence(sequence, new_points, max_length=3)

    assert result == [(0, 0), (1, 1), (2, 2)]
