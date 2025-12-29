"""Helpers to map CLI-style inputs into PreconditionConfig."""

from __future__ import annotations

from .config import PreconditionConfig


def resolve_teed_weights_path(weights_path: str | None, default_weights_path: str | None = None) -> str | None:
    candidate = weights_path if weights_path is not None else default_weights_path
    if candidate is None:
        return None
    candidate = str(candidate).strip()
    if candidate == "":
        return None
    return candidate


def apply_fixed_stroke_config(
    cfg: PreconditionConfig,
    *,
    enabled: bool | None,
    alpha: float | None = None,
) -> None:
    if not enabled:
        return
    a = 0.9 if alpha is None else float(alpha)
    a = max(0.0, min(1.0, a))
    cfg.fixed_stroke_rgba = (0.0, 0.0, 0.0, a)


def apply_precondition_cleanup(
    cfg: PreconditionConfig,
    *,
    max_paths: int | None = None,
    min_path_length: int | None = None,
    max_path_length: int | None = None,
    min_component_area: int | None = None,
    morph_open_radius: int | None = None,
    morph_close_radius: int | None = None,
) -> None:
    if max_paths is not None:
        cfg.max_paths = int(max_paths)
    if min_path_length is not None:
        cfg.min_path_length = int(min_path_length)
    if max_path_length is not None:
        cfg.max_path_length = int(max_path_length)
    if min_component_area is not None:
        cfg.min_component_area = int(min_component_area)
    if morph_open_radius is not None:
        cfg.morph_open_radius = int(morph_open_radius)
    if morph_close_radius is not None:
        cfg.morph_close_radius = int(morph_close_radius)


def apply_precondition_scaling(
    cfg: PreconditionConfig,
    *,
    target_paths_min: int | None = None,
    target_paths_max: int | None = None,
) -> None:
    # Keep CLI surface minimal: only target path range is user-tunable.
    if target_paths_min is not None:
        cfg.precondition_target_paths_min = int(target_paths_min)
    if target_paths_max is not None:
        cfg.precondition_target_paths_max = int(target_paths_max)


def apply_polyline_settings(
    cfg: PreconditionConfig,
    *,
    merge_polylines: bool | None = None,
    merge_distance: float | None = None,
    merge_angle_deg: float | None = None,
    force_open_paths: bool | None = None,
) -> None:
    if merge_polylines is not None:
        cfg.merge_polylines = bool(merge_polylines)
    if merge_distance is not None:
        cfg.merge_distance = float(merge_distance)
    if merge_angle_deg is not None:
        cfg.merge_angle_deg = float(merge_angle_deg)
    if force_open_paths is not None:
        cfg.force_open_paths = bool(force_open_paths)


def apply_stroke_widths(
    cfg: PreconditionConfig,
    *,
    base_stroke_width: float | None = None,
    max_stroke_width: float | None = None,
    clamp_max_width: float | None = None,
) -> None:
    if base_stroke_width is None and max_stroke_width is None and clamp_max_width is None:
        return
    if base_stroke_width is not None or max_stroke_width is not None:
        cfg.stroke_width_mode = "absolute"
    base = cfg.base_stroke_width if base_stroke_width is None else float(base_stroke_width)
    max_w = cfg.max_stroke_width if max_stroke_width is None else float(max_stroke_width)
    if clamp_max_width is not None:
        max_w = min(max_w, float(clamp_max_width))
    cfg.max_stroke_width = max_w
    cfg.base_stroke_width = min(base, max_w)


def apply_pen_widths(
    cfg: PreconditionConfig,
    *,
    stroke_width_mode: str | None = None,
    pen_min_mm: float | None = None,
    pen_max_mm: float | None = None,
) -> None:
    if stroke_width_mode is not None:
        cfg.stroke_width_mode = str(stroke_width_mode)
    if pen_min_mm is not None:
        cfg.stroke_width_pen_min_mm = float(pen_min_mm)
    if pen_max_mm is not None:
        cfg.stroke_width_pen_max_mm = float(pen_max_mm)


def apply_teed_settings(
    cfg: PreconditionConfig,
    *,
    weights_path: str | None,
    default_weights_path: str | None = None,
    detect_res: int | None = None,
    detect_res_default: int | None = None,
    threshold: float | None = None,
    threshold_default: float | None = None,
    safe_steps: int | None = None,
    safe_steps_default: int | None = None,
    threshold_mode: str | None = None,
    threshold_mode_default: str | None = None,
    hysteresis_low_ratio: float | None = None,
    hysteresis_low_ratio_default: float | None = None,
    threshold_quantile: float | None = None,
    threshold_quantile_default: float | None = None,
    lineart_enabled: bool | None = None,
    lineart_enabled_default: bool | None = None,
    lineart_blur_sigma: float | None = None,
    lineart_blur_sigma_default: float | None = None,
    lineart_strength: float | None = None,
    lineart_strength_default: float | None = None,
    lineart_combine: str | None = None,
    lineart_combine_default: str | None = None,
    require_weights: bool = False,
    missing_weights_message: str | None = None,
) -> None:
    resolved = resolve_teed_weights_path(weights_path, default_weights_path)
    if resolved is not None:
        cfg.teed_weights_path = resolved

    if detect_res is None:
        detect_res = detect_res_default
    if detect_res is not None:
        cfg.teed_detect_resolution = int(detect_res)

    if threshold is None:
        threshold = threshold_default
    if threshold is not None:
        cfg.teed_threshold = float(threshold)

    if safe_steps is None:
        safe_steps = safe_steps_default
    if safe_steps is not None:
        cfg.teed_safe_steps = int(safe_steps)

    if threshold_mode is None:
        threshold_mode = threshold_mode_default
    if threshold_mode is not None:
        cfg.teed_threshold_mode = str(threshold_mode)

    if hysteresis_low_ratio is None:
        hysteresis_low_ratio = hysteresis_low_ratio_default
    if hysteresis_low_ratio is not None:
        cfg.teed_hysteresis_low_ratio = float(hysteresis_low_ratio)

    if threshold_quantile is None:
        threshold_quantile = threshold_quantile_default
    if threshold_quantile is not None:
        cfg.teed_threshold_quantile = float(threshold_quantile)

    if lineart_enabled is None:
        lineart_enabled = lineart_enabled_default
    if lineart_enabled is not None:
        cfg.teed_lineart = bool(lineart_enabled)

    if lineart_blur_sigma is None:
        lineart_blur_sigma = lineart_blur_sigma_default
    if lineart_blur_sigma is not None:
        cfg.teed_lineart_blur_sigma = float(lineart_blur_sigma)

    if lineart_strength is None:
        lineart_strength = lineart_strength_default
    if lineart_strength is not None:
        cfg.teed_lineart_strength = float(lineart_strength)

    if lineart_combine is None:
        lineart_combine = lineart_combine_default
    if lineart_combine is not None:
        cfg.teed_lineart_combine = str(lineart_combine)

    if require_weights and (cfg.teed_weights_path is None or str(cfg.teed_weights_path).strip() == ""):
        message = missing_weights_message or "TEED edge backend requires weights. Provide a .pth path."
        raise ValueError(message)


__all__ = [
    "apply_fixed_stroke_config",
    "apply_precondition_scaling",
    "apply_polyline_settings",
    "apply_pen_widths",
    "apply_precondition_cleanup",
    "apply_stroke_widths",
    "apply_teed_settings",
    "resolve_teed_weights_path",
]
