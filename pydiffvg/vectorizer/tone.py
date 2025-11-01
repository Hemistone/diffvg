"""Tone management helpers for the vectorizer pipeline."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .edges import sobel_edge_map


def brightness_map(image: np.ndarray) -> np.ndarray:
    """Return a normalized brightness map in ``[0, 1]`` for *image*."""

    if image.ndim == 2:
        gray = image.astype(np.float32, copy=False)
    else:
        gray = image[..., :3].astype(np.float32, copy=False).mean(axis=-1)
    if gray.max(initial=0.0) > 1.0:
        gray = gray / 255.0
    return 1.0 - gray


def combine_brightness_and_edges(brightness: np.ndarray, edges: np.ndarray, *, edge_weight: float = 0.5) -> np.ndarray:
    """Blend *brightness* and *edges* into a single saliency map."""

    edge_weight = float(np.clip(edge_weight, 0.0, 1.0))
    brightness = _normalize(brightness)
    edges = _normalize(edges)
    combined = (1.0 - edge_weight) * brightness + edge_weight * edges
    return _normalize(combined)


def integral_image(field: np.ndarray) -> np.ndarray:
    """Compute the summed area table of *field* using ``float32`` precision."""

    return np.cumsum(np.cumsum(field.astype(np.float32), axis=0), axis=1)


def seed_probability_map(image: np.ndarray, *, edge_weight: float = 0.5, residual: Optional[np.ndarray] = None) -> np.ndarray:
    """Return a probability distribution highlighting good stroke seeds."""

    bright = brightness_map(image)
    edges = sobel_edge_map(image)
    saliency = combine_brightness_and_edges(bright, edges, edge_weight=edge_weight)
    if residual is not None:
        residual = _normalize(residual)
        saliency = _normalize(0.5 * saliency + 0.5 * residual)
    # Avoid degenerate zero maps.
    saliency = saliency + 1e-6
    return saliency / saliency.sum()


def _normalize(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float32)
    values = values.astype(np.float32, copy=False)
    vmin = float(values.min())
    vmax = float(values.max())
    if vmax - vmin < 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return (values - vmin) / (vmax - vmin)
