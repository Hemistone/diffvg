from __future__ import annotations

from .runtime import HAS_TRITON, is_available, tl, triton  # re-export for callers needing the runtime handles
from .forward import (
    env_wants_triton,
    env_forces_python,
    composite_gaussians_full_triton,
    composite_gaussians_tiled_triton,
    _build_tile_csr,
)
from .backward import (
    backward_tiled_color_triton,
    backward_tiled_full_triton,
    fused_spec_reduce_triton,
    fused_points_scatter_triton,
    get_last_backward_capture,
)

__all__ = [
    "HAS_TRITON",
    "is_available",
    "triton",
    "tl",
    "env_wants_triton",
    "env_forces_python",
    "composite_gaussians_full_triton",
    "composite_gaussians_tiled_triton",
    "backward_tiled_color_triton",
    "backward_tiled_full_triton",
    "fused_spec_reduce_triton",
    "fused_points_scatter_triton",
    "get_last_backward_capture",
    "_build_tile_csr",
]
