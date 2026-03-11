#!/usr/bin/env python3
"""Compute plotter-oriented metrics for one or more SVG files."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pydiffvg


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("svg", nargs="+", help="one or more SVG files to analyze")
    parser.add_argument(
        "--vpype-pipeline",
        type=str,
        default=None,
        help="optional vpype pipeline, e.g. 'linemerge linesimplify linesort'",
    )
    parser.add_argument(
        "--vpype-output-dir",
        type=str,
        default=None,
        help="directory for vpype-processed SVGs; defaults to a temporary directory",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="optional path to write metrics as JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    svg_paths = [Path(item).resolve() for item in args.svg]
    for path in svg_paths:
        if not path.is_file():
            raise FileNotFoundError(f"SVG not found: {path}")

    rows = pydiffvg.analyze_svgs(
        svg_paths,
        vpype_pipeline=args.vpype_pipeline,
        vpype_output_dir=args.vpype_output_dir,
    )
    print(pydiffvg.format_metrics_table(rows))
    if args.json_out:
        Path(args.json_out).resolve().write_text(
            pydiffvg.metrics_rows_to_json(rows),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
