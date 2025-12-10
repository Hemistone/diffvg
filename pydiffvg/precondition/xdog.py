"""Edge extraction using a lightweight XDoG variant.

This keeps the math intentionally simple (two Gaussians + tanh shaping) to
preserve speed while exposing the familiar XDoG knobs in PreconditionConfig.
"""

from __future__ import annotations

import numpy as np
from skimage import filters, morphology

from .config import PreconditionConfig


def _as_gray(image: np.ndarray) -> np.ndarray:
    """Return a float32 grayscale image in [0, 1]."""
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3:
        gray = image[..., :3].mean(axis=2)
    else:
        raise ValueError(f"expected HxW or HxWxC image, got shape {image.shape}")
    gray = np.asarray(gray, dtype=np.float32)
    return np.clip(gray, 0.0, 1.0)


def xdog_edges(image: np.ndarray, cfg: PreconditionConfig) -> np.ndarray:
    """Compute a binary edge mask from an RGB/gray image.

    Returns a boolean array with True where edges are detected.
    """
    gray = _as_gray(image)

    g1 = filters.gaussian(gray, sigma=cfg.sigma, mode="reflect")
    g2 = filters.gaussian(gray, sigma=cfg.sigma * cfg.k, mode="reflect")
    dog = g1 - cfg.gamma * g2

    # Winnemöller-style shaping; stronger edges become more negative after the
    # offset and tanh, making thresholding stable across scales.
    shaped = np.tanh(cfg.phi * (dog - cfg.epsilon))
    edges = shaped < cfg.edge_threshold

    if cfg.min_component_area > 0:
        edges = morphology.remove_small_objects(edges, cfg.min_component_area)
    if cfg.morph_open_radius > 0:
        edges = morphology.binary_opening(edges, morphology.disk(cfg.morph_open_radius))
    if cfg.morph_close_radius > 0:
        edges = morphology.binary_closing(edges, morphology.disk(cfg.morph_close_radius))

    return edges


__all__ = ["xdog_edges"]
