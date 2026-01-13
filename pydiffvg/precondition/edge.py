"""Edge selection helper for preconditioning."""

from __future__ import annotations

import numpy as np
import torch

from .config import PreconditionConfig
from .xdog import xdog_edges


def compute_edge_mask(
    image_rgb: np.ndarray,
    cfg: PreconditionConfig,
    *,
    device: torch.device,
) -> np.ndarray:
    mode = (cfg.mode or "xdog").strip().lower()
    if mode == "xdog":
        return xdog_edges(image_rgb, cfg)
    if mode == "teed":
        from .teed import teed_edges

        return teed_edges(image_rgb, cfg, device=device)
    raise ValueError(f"Unsupported precond mode '{cfg.mode}'. Choose from: xdog, teed")


__all__ = ["compute_edge_mask"]
