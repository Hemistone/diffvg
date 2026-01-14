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
        message = missing_weights_message or "TEED preconditioning requires weights. Provide a .pth path."
        raise ValueError(message)


def apply_flowline_settings(
    cfg: PreconditionConfig,
    *,
    edge_backend: str | None = None,
    seed_mode: str | None = None,
    seed_quantile: float | None = None,
    seed_threshold: float | None = None,
    min_strength: float | None = None,
    step_px: float | None = None,
    max_len: int | None = None,
    min_len: int | None = None,
    min_seed_dist: int | None = None,
    curvature_deg: float | None = None,
    field_sigma: float | None = None,
    field_iters: int | None = None,
    coverage_decay: float | None = None,
    coverage_radius: int | None = None,
) -> None:
    if edge_backend is not None:
        cfg.flow_edge_backend = str(edge_backend)
    if seed_mode is not None:
        cfg.flow_seed_mode = str(seed_mode)
    if seed_quantile is not None:
        cfg.flow_seed_quantile = float(seed_quantile)
    if seed_threshold is not None:
        cfg.flow_seed_threshold = float(seed_threshold)
    if min_strength is not None:
        cfg.flow_min_strength = float(min_strength)
    if step_px is not None:
        cfg.flow_step_px = float(step_px)
    if max_len is not None:
        cfg.flow_max_len = int(max_len)
    if min_len is not None:
        cfg.flow_min_len = int(min_len)
    if min_seed_dist is not None:
        cfg.flow_min_seed_dist = int(min_seed_dist)
    if curvature_deg is not None:
        cfg.flow_curvature_deg = float(curvature_deg)
    if field_sigma is not None:
        cfg.flow_field_sigma = float(field_sigma)
    if field_iters is not None:
        cfg.flow_field_iters = int(field_iters)
    if coverage_decay is not None:
        cfg.flow_coverage_decay = float(coverage_decay)
    if coverage_radius is not None:
        cfg.flow_coverage_radius = int(coverage_radius)


def build_precondition_config(
    args,
    *,
    default_teed_weights_path: str | None = None,
    require_teed_weights: bool = False,
    missing_weights_message: str | None = None,
    max_paths_fallback: int | None = None,
    clamp_max_width: float | None = None,
) -> PreconditionConfig:
    cfg = PreconditionConfig()
    apply_pen_widths(
        cfg,
        stroke_width_mode=getattr(args, "stroke_width_mode", None),
        pen_min_mm=getattr(args, "pen_width_min_mm", None),
        pen_max_mm=getattr(args, "pen_width_max_mm", None),
    )
    apply_fixed_stroke_config(
        cfg,
        enabled=getattr(args, "fixed_stroke", False),
        alpha=getattr(args, "fixed_stroke_alpha", None),
    )
    apply_lineart_settings(
        cfg,
        threshold_mode=getattr(args, "lineart_threshold_mode", None),
        threshold_quantile=getattr(args, "lineart_threshold_quantile", None),
        threshold=getattr(args, "lineart_threshold", None),
    )
    apply_flowline_settings(
        cfg,
        edge_backend=getattr(args, "flow_edge_backend", None),
        seed_mode=getattr(args, "flow_seed_mode", None),
        seed_quantile=getattr(args, "flow_seed_quantile", None),
        seed_threshold=getattr(args, "flow_seed_threshold", None),
        min_strength=getattr(args, "flow_min_strength", None),
        step_px=getattr(args, "flow_step_px", None),
        max_len=getattr(args, "flow_max_len", None),
        min_len=getattr(args, "flow_min_len", None),
        min_seed_dist=getattr(args, "flow_min_seed_dist", None),
        curvature_deg=getattr(args, "flow_curvature_deg", None),
        field_sigma=getattr(args, "flow_field_sigma", None),
        field_iters=getattr(args, "flow_field_iters", None),
        coverage_decay=getattr(args, "flow_coverage_decay", None),
        coverage_radius=getattr(args, "flow_coverage_radius", None),
    )
    arg_max_paths = getattr(args, "max_paths", None)
    apply_precondition_cleanup(
        cfg,
        max_paths=arg_max_paths,
        min_path_length=getattr(args, "min_path_length", None),
        max_path_length=getattr(args, "max_path_length", None),
        min_component_area=getattr(args, "min_component_area", None),
        morph_open_radius=getattr(args, "morph_open_radius", None),
        morph_close_radius=getattr(args, "morph_close_radius", None),
    )
    apply_precondition_scaling(
        cfg,
        target_paths_min=getattr(args, "precond_target_paths_min", None),
        target_paths_max=getattr(args, "precond_target_paths_max", None),
    )
    apply_stroke_widths(
        cfg,
        base_stroke_width=getattr(args, "base_stroke_width", None),
        max_stroke_width=getattr(args, "max_stroke_width", None),
        clamp_max_width=clamp_max_width,
    )
    apply_polyline_settings(
        cfg,
        merge_polylines=getattr(args, "merge_polylines", None),
        merge_distance=getattr(args, "merge_distance", None),
        merge_angle_deg=getattr(args, "merge_angle_deg", None),
        force_open_paths=getattr(args, "force_open_paths", None),
    )
    cfg.mode = (getattr(args, "precond_mode", "xdog") or "xdog").strip().lower()
    if cfg.mode not in ("xdog", "teed", "lineart", "flowline"):
        raise ValueError(
            f"Unsupported --precond-mode '{cfg.mode}'. Choose from: xdog, teed, lineart, flowline"
        )
    needs_teed = cfg.mode == "teed" or (
        cfg.mode == "flowline" and (cfg.flow_edge_backend or "teed").strip().lower() == "teed"
    )
    if needs_teed:
        apply_teed_settings(
            cfg,
            weights_path=getattr(args, "teed_weights", None),
            default_weights_path=default_teed_weights_path,
            detect_res=getattr(args, "teed_detect_res", None),
            threshold=getattr(args, "teed_threshold", None),
            safe_steps=getattr(args, "teed_safe_steps", None),
            threshold_mode=getattr(args, "teed_threshold_mode", None),
            hysteresis_low_ratio=getattr(args, "teed_hysteresis_low_ratio", None),
            threshold_quantile=getattr(args, "teed_threshold_quantile", None),
            lineart_enabled=getattr(args, "teed_lineart", None),
            lineart_blur_sigma=getattr(args, "teed_lineart_blur_sigma", None),
            lineart_strength=getattr(args, "teed_lineart_strength", None),
            lineart_combine=getattr(args, "teed_lineart_combine", None),
            require_weights=require_teed_weights,
            missing_weights_message=missing_weights_message,
        )
    if max_paths_fallback is not None and arg_max_paths is None:
        cfg.max_paths = int(max_paths_fallback)
    return cfg


def apply_lineart_settings(
    cfg: PreconditionConfig,
    *,
    threshold_mode: str | None = None,
    threshold_quantile: float | None = None,
    threshold: float | None = None,
) -> None:
    if threshold_mode is not None:
        cfg.lineart_threshold_mode = str(threshold_mode)
    if threshold_quantile is not None:
        cfg.lineart_threshold_quantile = float(threshold_quantile)
    if threshold is not None:
        cfg.lineart_threshold = float(threshold)


__all__ = [
    "build_precondition_config",
    "apply_fixed_stroke_config",
    "apply_precondition_scaling",
    "apply_polyline_settings",
    "apply_pen_widths",
    "apply_precondition_cleanup",
    "apply_stroke_widths",
    "apply_teed_settings",
    "apply_flowline_settings",
    "apply_lineart_settings",
    "resolve_teed_weights_path",
]
