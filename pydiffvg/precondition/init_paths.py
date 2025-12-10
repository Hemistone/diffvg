"""High-level helper to turn a raster image into preconditioned diffvg shapes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import pydiffvg
from .config import PreconditionConfig
from .skeleton import skeletonize_edges, skeleton_to_polylines
from .vectorize import polylines_to_paths
from .xdog import xdog_edges


def _load_image(image: str | Path | np.ndarray) -> np.ndarray:
    if isinstance(image, np.ndarray):
        arr = image
    else:
        with Image.open(image) as im:
            arr = np.array(im.convert("RGB"))
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=2)
    if arr.shape[2] == 4:
        arr = arr[..., :3]
    arr = np.asarray(arr, dtype=np.float32)
    if arr.max() > 1.5:  # assume 0-255
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


@dataclass
class PreconditionedScene:
    width: int
    height: int
    shapes: list[pydiffvg.Path]
    shape_groups: list[pydiffvg.ShapeGroup]
    renderer: pydiffvg.Renderer
    scene_args: tuple
    edge_mask: np.ndarray
    skeleton: np.ndarray
    polylines: list[list[tuple[int, int]]]


def build_preconditioned_scene(
    image: str | Path | np.ndarray,
    cfg: PreconditionConfig | None = None,
    *,
    backend: str = "splat",
    device: torch.device | None = None,
) -> PreconditionedScene:
    """Generate initial diffvg paths from a raster image."""
    cfg = cfg or PreconditionConfig()
    device = device if device is not None else pydiffvg.get_device()

    rgb = _load_image(image)
    edge_mask = xdog_edges(rgb, cfg)
    skeleton = skeletonize_edges(edge_mask)
    polylines = skeleton_to_polylines(skeleton, cfg)

    height, width = rgb.shape[0], rgb.shape[1]
    shapes, groups = polylines_to_paths(
        polylines,
        rgb,
        cfg,
        canvas_w=width,
        canvas_h=height,
        device=device,
    )

    renderer = pydiffvg.Renderer(backend=backend)
    scene_args = renderer.serialize_scene(
        width,
        height,
        shapes,
        groups,
        device=device,
    )

    return PreconditionedScene(
        width=width,
        height=height,
        shapes=shapes,
        shape_groups=groups,
        renderer=renderer,
        scene_args=scene_args,
        edge_mask=edge_mask,
        skeleton=skeleton,
        polylines=polylines,
    )


__all__ = ["PreconditionedScene", "build_preconditioned_scene"]
