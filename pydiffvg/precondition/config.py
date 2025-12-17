"""Configuration for raster-to-stroke preconditioning.

The dataclass below keeps the knobs small and explicit so callers can tweak
edge extraction, skeleton tracing, and path shaping without digging through
implementation details.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PreconditionConfig:
    # XDoG parameters
    sigma: float = 1.2
    k: float = 1.6
    gamma: float = 0.98
    epsilon: float = 0.01
    phi: float = 10.0
    edge_threshold: float = -0.015  # post-nonlinearity threshold; more negative = fewer edges

    # Edge cleanup
    min_component_area: int = 24
    morph_open_radius: int = 1
    morph_close_radius: int = 1

    # Skeleton / polyline extraction
    max_path_length: int = 1024
    min_path_length: int = 12
    simplify_epsilon: float = 1.25
    max_paths: int | None = 2000
    smooth_window: int = 5

    # Stroke shaping
    base_stroke_width: float = 1.6
    max_stroke_width: float = 3.2
    stroke_width_gamma: float = 1.5
    sample_color: bool = True
    curve_mode: str = "bezier"  # "polyline" or "bezier"
    catmull_rom_tension: float = 0.5

    # Scoring / ordering
    sort_by: str = "darkness_length"  # or "length"

    # Line-art mode toggles
    mode: str = "xdog"  # "xdog" or "lineart"
    num_colors: int = 1
    palette_mode: str = "auto"  # "auto" or "fixed"
    palette_colors: list[tuple[float, float, float]] | None = None
    merge_distance: float = 3.0
    merge_angle_deg: float = 18.0
    clamp_stroke_width_per_color: bool = True


__all__ = ["PreconditionConfig"]
