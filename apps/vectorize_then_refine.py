"""Vectorize an input image and render the result with the selected backend."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

import pydiffvg
from pydiffvg.vectorizer.pipeline import vectorize_then_render


def _load_image(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        array = np.asarray(rgb, dtype=np.float32) / 255.0
    return array


def _parse_samples(value: int | Tuple[int, int]) -> Tuple[int, int]:
    if isinstance(value, tuple):
        return value
    return value, value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="Input raster image")
    parser.add_argument(
        "--backend",
        default="baseline",
        choices=pydiffvg.list_backends(),
        help="Rendering backend to use",
    )
    parser.add_argument(
        "--samples-per-pixel",
        type=int,
        default=4,
        help="Samples per pixel for the renderer",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for rendering")
    parser.add_argument(
        "--num-layers",
        type=int,
        default=1,
        help="Number of vector layers to synthesize",
    )
    parser.add_argument(
        "--mode",
        choices=("lines", "quad", "cubic"),
        default="lines",
        help="Segment type emitted by the vectorizer",
    )
    parser.add_argument(
        "--fit-bezier",
        action="store_true",
        help="Fit quadratic/cubic Béziers to the traced polylines",
    )
    parser.add_argument(
        "--max-strokes",
        type=int,
        default=64,
        help="Maximum number of strokes to seed",
    )
    parser.add_argument(
        "--seed-method",
        choices=("nms", "poisson"),
        default="nms",
        help="Seed selection strategy",
    )
    parser.add_argument(
        "--edge-weight",
        type=float,
        default=0.5,
        help="Edge weight applied during saliency fusion",
    )
    parser.add_argument(
        "--erase-radius",
        type=int,
        default=3,
        help="Residual erase radius used after each stroke",
    )
    parser.add_argument(
        "--svg-out",
        type=Path,
        default=None,
        help="Optional path to save the vectorized SVG",
    )
    parser.add_argument(
        "--render-out",
        type=Path,
        default=None,
        help="Optional path to save the rendered image",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=2.2,
        help="Gamma used when saving the rendered image",
    )

    args = parser.parse_args()

    image = _load_image(args.image)
    vectorize_kwargs = {
        "num_layers": args.num_layers,
        "mode": args.mode,
        "fit_bezier": args.fit_bezier,
        "max_strokes": args.max_strokes,
        "seed_method": args.seed_method,
        "edge_weight": args.edge_weight,
        "erase_radius": args.erase_radius,
    }

    samples = _parse_samples(args.samples_per_pixel)

    result = vectorize_then_render(
        image,
        backend=args.backend,
        vectorize_kwargs=vectorize_kwargs,
        samples=samples,
        seed=args.seed,
        background_image=None,
        save_svg_path=str(args.svg_out) if args.svg_out is not None else None,
    )

    doc = result["doc"]
    print(
        f"Vectorized {args.image} -> {len(doc.layers)} layer(s), "
        f"{sum(len(layer.paths) for layer in doc.layers)} path(s)."
    )

    if args.render_out is not None:
        pydiffvg.imwrite(
            result["image"].detach().cpu(),
            str(args.render_out),
            gamma=args.gamma,
        )
        print(f"Rendered output saved to {args.render_out}")

    if args.svg_out is not None:
        print(f"SVG saved to {args.svg_out}")


if __name__ == "__main__":
    main()
