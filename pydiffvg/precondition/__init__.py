"""Raster preconditioning utilities for faster diffvg optimization loops."""

from .config import PreconditionConfig
from .edge import compute_edge_mask
from .init_paths import PreconditionedScene, build_preconditioned_scene
from .teed import teed_edge_strength, teed_edges
from .xdog import xdog_edges
from .skeleton import skeletonize_edges, skeleton_to_polylines
from .vectorize import polylines_to_paths
from .lineart import build_lineart_scene
from .flowline import flowline_polylines

__all__ = [
    "PreconditionConfig",
    "PreconditionedScene",
    "build_preconditioned_scene",
    "build_lineart_scene",
    "flowline_polylines",
    "compute_edge_mask",
    "teed_edge_strength",
    "teed_edges",
    "xdog_edges",
    "skeletonize_edges",
    "skeleton_to_polylines",
    "polylines_to_paths",
]
