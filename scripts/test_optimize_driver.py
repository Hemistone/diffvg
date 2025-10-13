#!/usr/bin/env python3
"""Smoke test for the high-level SVG optimization driver."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

import pydiffvg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--svg",
        type=Path,
        default=Path("apps/imgs/note_small.svg"),
        help="Path to an SVG input. Defaults to apps/imgs/note_small.svg.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of optimization iterations to execute.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.svg.exists():
        raise SystemExit(f"SVG file not found: {args.svg}")

    pydiffvg.set_use_gpu(False)
    torch.manual_seed(0)

    settings = pydiffvg.SvgOptimizationSettings()
    driver = pydiffvg.SvgOptimizationDriver(
        str(args.svg),
        settings=settings,
        optimize_background=False,
        verbose=False,
        device=torch.device("cpu"),
    )

    canvas = driver.document.canvas
    target = torch.ones((canvas[1], canvas[0], 4), dtype=torch.float32)
    target[..., 3] = 1.0

    def loss_fn(image: torch.Tensor, iteration: int, drv: pydiffvg.SvgOptimizationDriver) -> torch.Tensor:
        return torch.nn.functional.mse_loss(image, target)

    history = driver.optimize(
        loss_fn,
        args.iterations,
        seed_schedule=lambda t: t,
    )

    if history[0] - history[-1] < 0.005:
        raise SystemExit(
            f"Driver smoke test failed: loss did not decrease enough ({history[0]:.6f} -> {history[-1]:.6f})"
        )

    print(
        f"ok: driver loss {history[0]:.6f} -> {history[-1]:.6f} over {args.iterations} iterations using {args.svg}"
    )


if __name__ == "__main__":
    main()
