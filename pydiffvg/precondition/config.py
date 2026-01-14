"""Configuration for raster-to-stroke preconditioning.

The dataclass below keeps the knobs small and explicit so callers can tweak
edge extraction, skeleton tracing, and path shaping without digging through
implementation details.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..palette import Palette

PRECONDITION_TARGET_PATHS_MIN_DEFAULT = 512
PRECONDITION_TARGET_PATHS_MAX_DEFAULT = 1024
PRECONDITION_AUTO_TRIALS_DEFAULT = 4


@dataclass
class PreconditionConfig:
    # Preconditioning resolution controls (auto-scaled to hit a target path range).
    precondition_target_paths_min: int = PRECONDITION_TARGET_PATHS_MIN_DEFAULT
    precondition_target_paths_max: int = PRECONDITION_TARGET_PATHS_MAX_DEFAULT
    precondition_auto_trials: int = PRECONDITION_AUTO_TRIALS_DEFAULT

    # XDoG parameters
    sigma: float = 1.2
    k: float = 1.6
    gamma: float = 0.98
    epsilon: float = 0.01
    phi: float = 10.0
    edge_threshold: float = -0.015  # post-nonlinearity threshold; more negative = fewer edges

    # TEED parameters (used when mode="teed")
    teed_weights_path: str | None = None
    teed_detect_resolution: int = 512
    teed_safe_steps: int = 2
    teed_threshold: float = 0.5
    teed_threshold_mode: str = "fixed"  # "fixed", "hysteresis", "quantile", "otsu"
    teed_hysteresis_low_ratio: float = 0.5
    teed_auto_tune_threshold: bool = True
    teed_threshold_min: float = 0.05
    teed_threshold_decay: float = 0.85
    teed_threshold_trials: int = 8
    teed_threshold_quantile: float = 0.85
    teed_lineart: bool = False
    teed_lineart_blur_sigma: float = 2.0
    teed_lineart_strength: float = 1.5
    teed_lineart_combine: str = "screen"  # "screen", "max", "add"

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
    merge_polylines: bool = False

    # Stroke shaping
    force_open_paths: bool = False
    stroke_width_mode: str = "a4_pen"  # "absolute" or "a4_pen"
    stroke_width_pen_min_mm: float = 0.35
    stroke_width_pen_max_mm: float = 0.8
    base_stroke_width: float = 1.6
    max_stroke_width: float = 3.2
    stroke_width_gamma: float = 1.5
    xdog_stroke_width_scale: float = 0.9
    sample_color: bool = True
    fixed_stroke_rgba: tuple[float, float, float, float] | None = None
    palette: "Palette | None" = None
    curve_mode: str = "bezier"  # "polyline" or "bezier"
    catmull_rom_tension: float = 0.5

    # Scoring / ordering
    sort_by: str = "darkness_length"  # or "length"

    # Preconditioning mode
    mode: str = "xdog"  # "xdog", "teed", "lineart", or "flowline"
    num_colors: int = 1
    palette_mode: str = "auto"  # "auto" or "fixed"
    palette_colors: list[tuple[float, float, float]] | None = None
    merge_distance: float = 3.0
    merge_angle_deg: float = 18.0
    clamp_stroke_width_per_color: bool = True
    lineart_threshold_mode: str = "quantile"  # "quantile", "otsu", or "fixed"
    lineart_threshold_quantile: float = 0.25
    lineart_threshold: float = 0.5

    # Flowline (path growing) parameters
    flow_edge_backend: str = "teed"  # "teed" or "xdog"
    flow_seed_mode: str = "quantile"  # "quantile" or "fixed"
    flow_seed_quantile: float = 0.85
    flow_seed_threshold: float = 0.2
    flow_min_strength: float = 0.2
    flow_step_px: float = 1.0
    flow_max_len: int = 256
    flow_min_len: int = 8
    flow_min_seed_dist: int = 6
    flow_curvature_deg: float = 60.0
    flow_field_sigma: float = 2.0
    flow_field_iters: int = 1
    flow_coverage_decay: float = 0.85
    flow_coverage_radius: int = 2


__all__ = [
    "PRECONDITION_AUTO_TRIALS_DEFAULT",
    "PRECONDITION_TARGET_PATHS_MAX_DEFAULT",
    "PRECONDITION_TARGET_PATHS_MIN_DEFAULT",
    "PreconditionConfig",
]
