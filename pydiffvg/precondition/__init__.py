"""Raster preconditioning utilities for faster diffvg optimization loops."""

from .config import PreconditionConfig
from .init_paths import PreconditionedScene, build_preconditioned_scene
from .xdog import xdog_edges
from .skeleton import skeletonize_edges, skeleton_to_polylines
from .vectorize import polylines_to_paths

__all__ = [
    "PreconditionConfig",
    "PreconditionedScene",
    "build_preconditioned_scene",
    "xdog_edges",
    "skeletonize_edges",
    "skeleton_to_polylines",
    "polylines_to_paths",
]
