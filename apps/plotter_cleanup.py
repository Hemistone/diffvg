#!/usr/bin/env python3
"""Conservative plotter-aware cleanup for diffvg SVG output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pydiffvg


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_svg", help="input SVG file")
    parser.add_argument("output_svg", help="output cleaned SVG file")
    parser.add_argument("--report-json", type=str, default=None, help="optional JSON report output path")
    parser.add_argument("--min-stroke-length-px", type=float, default=10.0, help="prune strokes shorter than this length")
    parser.add_argument("--merge-distance-px", type=float, default=3.0, help="maximum endpoint gap allowed for stitching")
    parser.add_argument("--merge-angle-deg", type=float, default=18.0, help="maximum tangent angle difference for stitching")
    parser.add_argument("--width-tolerance-px", type=float, default=0.25, help="maximum SVG stroke-width delta for stitching")
    parser.add_argument("--disable-reorder", action="store_true", help="disable nearest-neighbor reordering inside strict color buckets")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_svg = Path(args.input_svg).resolve()
    output_svg = Path(args.output_svg).resolve()
    if not input_svg.is_file():
        raise FileNotFoundError(f"SVG not found: {input_svg}")

    config = pydiffvg.PlotterCleanupConfig(
        min_stroke_length_px=args.min_stroke_length_px,
        merge_distance_px=args.merge_distance_px,
        merge_angle_deg=args.merge_angle_deg,
        width_tolerance_px=args.width_tolerance_px,
        reorder=not args.disable_reorder,
    )
    result = pydiffvg.cleanup_svg(input_svg, output_svg, config=config)

    rows = [
        pydiffvg.PlotterMetricsRow(source=str(input_svg), mode="before", metrics=result.before_metrics),
        pydiffvg.PlotterMetricsRow(source=str(output_svg), mode="after", metrics=result.after_metrics),
    ]
    print(pydiffvg.format_metrics_table(rows))
    print(json.dumps(result.to_report(), indent=2))

    if args.report_json:
        Path(args.report_json).resolve().write_text(
            json.dumps(result.to_report(), indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
