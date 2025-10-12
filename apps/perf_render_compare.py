#!/usr/bin/env python3
"""Tiny perf harness to compare color vs. SDF render times.

Run with optional `--iterations N` or `--use-gpu` (if CUDA build available).
The scene is intentionally simple (one filled circle) to keep setup overhead low.
"""

from __future__ import annotations

import argparse
import time

import torch

import pydiffvg as d


def _build_scene(width: int, height: int):
    circle = d.Circle(radius=torch.tensor(60.0), center=torch.tensor([width / 2, height / 2]))
    group = d.ShapeGroup(
        shape_ids=torch.tensor([0], dtype=torch.int32),
        fill_color=torch.tensor([0.2, 0.8, 0.4, 1.0]),
        use_even_odd_rule=False,
    )
    return [circle], [group]


def _time_render(
    width: int,
    height: int,
    samples: int,
    output_type: d.OutputType,
    iterations: int,
    shapes,
    shape_groups,
) -> float:
    args = d.serialize_scene(
        width,
        height,
        shapes,
        shape_groups,
        output_type=output_type,
    )
    render = d.RenderFunction.apply

    # Warmup
    render(width, height, samples, samples, 0, None, *args)
    if d.get_use_gpu() and torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iterations):
        render(width, height, samples, samples, 0, None, *args)
    if d.get_use_gpu() and torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed / iterations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=256, help="canvas width")
    parser.add_argument("--height", type=int, default=256, help="canvas height")
    parser.add_argument("--samples", type=int, default=2, help="samples per axis (x = y)")
    parser.add_argument("--iterations", type=int, default=10, help="number of timed iterations")
    parser.add_argument("--backend", choices=d.list_backends(), help="optional backend override")
    parser.add_argument("--use-gpu", action="store_true", help="try running on the GPU backend")
    args = parser.parse_args()

    if args.backend:
        d.set_backend(args.backend)

    d.set_use_gpu(args.use_gpu)
    if args.use_gpu and not d.get_use_gpu():
        print("[warn] GPU requested but diffvg is running on CPU (CUDA unavailable).")

    shapes, shape_groups = _build_scene(args.width, args.height)

    color_avg = _time_render(
        args.width,
        args.height,
        args.samples,
        d.OutputType.color,
        args.iterations,
        shapes,
        shape_groups,
    )

    sdf_avg = _time_render(
        args.width,
        args.height,
        args.samples,
        d.OutputType.sdf,
        args.iterations,
        shapes,
        shape_groups,
    )

    device = d.get_device()
    backend = d.get_backend()
    print(f"device={device} backend={backend} samples={args.samples} iterations={args.iterations}")
    print(f"color avg: {color_avg * 1e3:.3f} ms")
    print(f"sdf   avg: {sdf_avg * 1e3:.3f} ms")
    if sdf_avg > 0:
        print(f"color/sdf ratio: {color_avg / sdf_avg:.3f}")


if __name__ == "__main__":
    main()
