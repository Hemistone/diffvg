"""Plotter-oriented metrics and cleanup helpers."""

from .cleanup import (
    PlotterCleanupConfig,
    PlotterCleanupResult,
    PlotterCleanupStats,
    cleanup_scene,
    cleanup_svg,
)
from .metrics import (
    PlotterMetrics,
    PlotterMetricsRow,
    PlotterStroke,
    analyze_scene,
    analyze_svg,
    analyze_svgs,
    format_metrics_table,
    iter_scene_strokes,
    metrics_rows_to_json,
    run_vpype_pipeline,
    summarize_strokes,
)

__all__ = [
    "PlotterCleanupConfig",
    "PlotterCleanupResult",
    "PlotterCleanupStats",
    "PlotterMetrics",
    "PlotterMetricsRow",
    "PlotterStroke",
    "analyze_scene",
    "analyze_svg",
    "analyze_svgs",
    "cleanup_scene",
    "cleanup_svg",
    "format_metrics_table",
    "iter_scene_strokes",
    "metrics_rows_to_json",
    "run_vpype_pipeline",
    "summarize_strokes",
]
