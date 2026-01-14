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
    if mode == "flowline":
        backend = (cfg.flow_edge_backend or "teed").strip().lower()
        if backend == "xdog":
            return xdog_edges(image_rgb, cfg)
        if backend == "teed":
            from .teed import teed_edges

            return teed_edges(image_rgb, cfg, device=device)
        raise ValueError(
            f"Unsupported flow_edge_backend '{cfg.flow_edge_backend}'. Choose from: xdog, teed"
        )
    raise ValueError(
        f"Unsupported precond mode '{cfg.mode}' for edge mask. Choose from: xdog, teed, flowline"
    )


__all__ = ["compute_edge_mask"]
