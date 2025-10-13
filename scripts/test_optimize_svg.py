#!/usr/bin/env python3
"""CPU-only smoke test for the SVG optimization pipeline.

This verifies that the monolithic `pydiffvg.optimize_svg` module can still
parse an SVG, render it, backpropagate a simple loss, and perform at least a
small loss reduction using its built-in optimizers.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

import pydiffvg


DEFAULT_SVG = Path("apps/imgs/note_small.svg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--svg",
        type=Path,
        default=DEFAULT_SVG,
        help="Path to an SVG input. Defaults to apps/imgs/note_small.svg.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of gradient steps to run (default: 5).",
    )
    parser.add_argument(
        "--expect_drop",
        type=float,
        default=0.005,
        help="Required absolute drop in MSE loss (default: 0.005).",
    )
    return parser.parse_args()


def run(svg_path: Path, iterations: int, expect_drop: float) -> None:
    if not svg_path.exists():
        raise SystemExit(f"SVG file not found: {svg_path}")

    pydiffvg.set_use_gpu(False)
    torch.manual_seed(0)

    settings = pydiffvg.SvgOptimizationSettings()
    optimizer = pydiffvg.OptimizableSvg(
        str(svg_path),
        settings=settings,
        optimize_background=False,
        verbose=False,
        device=torch.device("cpu"),
    )

    target = torch.ones(
        (optimizer.canvas[1], optimizer.canvas[0], 4),
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    target[..., 3] = 1.0

    losses = []
    for iteration in range(iterations):
        optimizer.zero_grad()
        img = optimizer.render(seed=iteration)
        loss = torch.nn.functional.mse_loss(img, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu())

    initial_loss = float(losses[0])
    final_loss = float(losses[-1])
    if initial_loss - final_loss < expect_drop:
        raise SystemExit(
            (
                "Smoke test failed: loss did not decrease enough "
                f"({initial_loss:.6f} -> {final_loss:.6f}, expected drop ≥ {expect_drop:.6f})"
            )
        )

    print(
        f"ok: loss {initial_loss:.6f} -> {final_loss:.6f} over {iterations} iterations "
        f"using {svg_path}"
    )


def main() -> None:
    args = parse_args()
    run(args.svg, args.iterations, args.expect_drop)


if __name__ == "__main__":
    main()
