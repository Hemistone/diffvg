"""Vectorize an image and export the result without diffvg backends.

This CLI mirrors the output layout used by :mod:`apps.painterly_rendering`
without depending on the compiled diffvg renderer.  It runs the Python-only
vectorizer pipeline, rasterizes a lightweight preview using Pillow, and writes
an SVG approximation of the vector document for inspection.
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _bootstrap_vectorizer_package() -> None:
    """Ensure ``pydiffvg.vectorizer`` can be imported without diffvg."""

    if "pydiffvg" in sys.modules:
        return
    try:  # Prefer the real package when the compiled backend is available.
        import pydiffvg  # type: ignore  # noqa: F401 - imported for side effects
    except ModuleNotFoundError as exc:
        if exc.name != "diffvg":
            raise
        pkg = types.ModuleType("pydiffvg")
        pkg.__path__ = [str(REPO_ROOT / "pydiffvg")]
        sys.modules["pydiffvg"] = pkg


_bootstrap_vectorizer_package()

from pydiffvg.vectorizer.api import Path as VectorPath  # noqa: E402
from pydiffvg.vectorizer.api import Segment, VectorDoc, vectorize


def _load_image(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        array = np.asarray(rgb, dtype=np.float32)
    if array.max(initial=0.0) > 1.0:
        array /= 255.0
    return array


def _sanitize_name(name: str) -> str:
    sanitized = [ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name]
    result = "".join(sanitized).strip("_")
    return result or "target"


def _ensure_results_dir(image_path: Path, *, results_root: Path) -> Path:
    run_name = _sanitize_name(image_path.stem)
    out_dir = results_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _rgba_components(color: Sequence[float]) -> Tuple[float, float, float, float]:
    comps = list(color)
    if len(comps) == 3:
        comps.append(1.0)
    if len(comps) != 4:
        raise ValueError("Expected RGB or RGBA color components")
    return tuple(float(max(0.0, min(1.0, c))) for c in comps)


def _to_rgb255(color: Sequence[float]) -> Tuple[int, int, int]:
    r, g, b, _ = _rgba_components(color)
    return (int(round(r * 255.0)), int(round(g * 255.0)), int(round(b * 255.0)))


def _to_rgba255(color: Sequence[float]) -> Tuple[int, int, int, int]:
    r, g, b, a = _rgba_components(color)
    return (
        int(round(r * 255.0)),
        int(round(g * 255.0)),
        int(round(b * 255.0)),
        int(round(a * 255.0)),
    )


def _points_close(p0: Sequence[float], p1: Sequence[float], *, eps: float = 1e-4) -> bool:
    return abs(p0[0] - p1[0]) <= eps and abs(p0[1] - p1[1]) <= eps


def _segment_samples(segment: Segment, *, steps: int = 32) -> List[Tuple[float, float]]:
    pts = [tuple(float(v) for v in xy) for xy in segment.points]
    if not pts:
        return []
    kind = segment.kind.lower()
    if kind == "line":
        return [pts[0], pts[-1]]
    if kind == "quad":
        p0, p1, p2 = pts
        return [_quad_point(p0, p1, p2, t) for t in np.linspace(0.0, 1.0, steps)]
    if kind == "cubic":
        p0, p1, p2, p3 = pts
        return [_cubic_point(p0, p1, p2, p3, t) for t in np.linspace(0.0, 1.0, steps)]
    raise ValueError(f"Unsupported segment kind: {segment.kind}")


def _quad_point(
    p0: Sequence[float],
    p1: Sequence[float],
    p2: Sequence[float],
    t: float,
) -> Tuple[float, float]:
    u = 1.0 - t
    x = u * u * p0[0] + 2 * u * t * p1[0] + t * t * p2[0]
    y = u * u * p0[1] + 2 * u * t * p1[1] + t * t * p2[1]
    return (x, y)


def _cubic_point(
    p0: Sequence[float],
    p1: Sequence[float],
    p2: Sequence[float],
    p3: Sequence[float],
    t: float,
) -> Tuple[float, float]:
    u = 1.0 - t
    x = (
        u * u * u * p0[0]
        + 3 * u * u * t * p1[0]
        + 3 * u * t * t * p2[0]
        + t * t * t * p3[0]
    )
    y = (
        u * u * u * p0[1]
        + 3 * u * u * t * p1[1]
        + 3 * u * t * t * p2[1]
        + t * t * t * p3[1]
    )
    return (x, y)


def _path_points(vec_path: VectorPath) -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []
    for segment in vec_path.segments:
        samples = _segment_samples(segment)
        if not samples:
            continue
        if not points:
            points.extend(samples)
            continue
        if not _points_close(points[-1], samples[0]):
            points.append(samples[0])
        points.extend(samples[1:])
    if vec_path.closed and points and not _points_close(points[0], points[-1]):
        points.append(points[0])
    return points


def _path_to_svg_d(vec_path: VectorPath) -> str:
    commands: List[str] = []
    current: Tuple[float, float] | None = None
    for segment in vec_path.segments:
        pts = [tuple(float(v) for v in xy) for xy in segment.points]
        if not pts:
            continue
        if current is None:
            commands.append(_move_to(pts[0]))
            current = pts[0]
        elif not _points_close(current, pts[0]):
            commands.append(_line_to(pts[0]))
            current = pts[0]

        kind = segment.kind.lower()
        if kind == "line":
            commands.append(_line_to(pts[-1]))
            current = pts[-1]
        elif kind == "quad":
            commands.append(
                "Q {:.3f} {:.3f} {:.3f} {:.3f}".format(
                    pts[1][0], pts[1][1], pts[2][0], pts[2][1]
                )
            )
            current = pts[2]
        elif kind == "cubic":
            commands.append(
                "C {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}".format(
                    pts[1][0],
                    pts[1][1],
                    pts[2][0],
                    pts[2][1],
                    pts[3][0],
                    pts[3][1],
                )
            )
            current = pts[3]
        else:
            raise ValueError(f"Unsupported segment kind: {segment.kind}")

    if vec_path.closed and commands:
        commands.append("Z")
    return " ".join(commands)


def _move_to(pt: Sequence[float]) -> str:
    return "M {:.3f} {:.3f}".format(pt[0], pt[1])


def _line_to(pt: Sequence[float]) -> str:
    return "L {:.3f} {:.3f}".format(pt[0], pt[1])


def _render_preview(doc: VectorDoc, *, background: Tuple[int, int, int, int] = (255, 255, 255, 255)) -> Image.Image:
    width, height = (int(doc.canvas_size[0]), int(doc.canvas_size[1]))
    if width <= 0 or height <= 0:
        raise ValueError("VectorDoc canvas_size must be positive")
    image = Image.new("RGBA", (width, height), background)
    draw = ImageDraw.Draw(image, "RGBA")

    for layer in doc.layers:
        for path in layer.paths:
            points = _path_points(path)
            if not points:
                continue

            if path.closed and path.pen.fill_color is not None:
                draw.polygon(points, fill=_to_rgba255(path.pen.fill_color))

            stroke_color = _to_rgba255(path.pen.stroke_color)
            stroke_width = max(1, int(round(path.pen.stroke_width)))
            draw.line(points, fill=stroke_color, width=stroke_width, joint="curve")

    return image.convert("RGB")


def _write_svg(doc: VectorDoc, path: Path) -> None:
    import xml.etree.ElementTree as etree

    width, height = doc.canvas_size
    root = etree.Element("svg", attrib={
        "xmlns": "http://www.w3.org/2000/svg",
        "version": "1.1",
        "width": str(int(width)),
        "height": str(int(height)),
    })

    for layer_index, layer in enumerate(doc.layers):
        group = etree.SubElement(root, "g", attrib={"id": f"layer-{layer_index}"})
        for path_index, vec_path in enumerate(layer.paths):
            if not vec_path.segments:
                continue
            attributes = {
                "id": f"path-{layer_index}-{path_index}",
                "d": _path_to_svg_d(vec_path),
                "fill": "none",
                "stroke": "rgb({},{},{})".format(*_to_rgb255(vec_path.pen.stroke_color)),
                "stroke-opacity": f"{_rgba_components(vec_path.pen.stroke_color)[3]:.3f}",
                "stroke-width": f"{vec_path.pen.stroke_width:.3f}",
                "stroke-linecap": "round",
                "stroke-linejoin": "round",
            }
            if vec_path.closed and vec_path.pen.fill_color is not None:
                fill_rgb = _to_rgb255(vec_path.pen.fill_color)
                attributes["fill"] = "rgb({},{},{})".format(*fill_rgb)
                attributes["fill-opacity"] = (
                    f"{_rgba_components(vec_path.pen.fill_color)[3]:.3f}"
                )
            etree.SubElement(group, "path", attrib=attributes)

    tree = etree.ElementTree(root)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="Input raster image")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results") / "vectorize_only",
        help="Directory where outputs are written",
    )
    parser.add_argument("--num-layers", type=int, default=1, help="Number of vector layers")
    parser.add_argument(
        "--mode",
        choices=("lines", "quad", "cubic"),
        default="lines",
        help="Segment type emitted by the vectorizer",
    )
    parser.add_argument("--fit-bezier", action="store_true", help="Fit Bézier curves to strokes")
    parser.add_argument("--max-strokes", type=int, default=64, help="Maximum number of strokes")
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
        help="Residual erase radius after each stroke",
    )

    args = parser.parse_args()

    image = _load_image(args.image)
    results_dir = _ensure_results_dir(args.image, results_root=args.results_root)

    vectorize_kwargs = {
        "num_layers": args.num_layers,
        "mode": args.mode,
        "fit_bezier": args.fit_bezier,
        "max_strokes": args.max_strokes,
        "seed_method": args.seed_method,
        "edge_weight": args.edge_weight,
        "erase_radius": args.erase_radius,
    }

    doc = vectorize(image, **vectorize_kwargs)

    preview = _render_preview(doc)
    final_png = results_dir / "final.png"
    final_svg = results_dir / "final.svg"

    preview.save(final_png)
    _write_svg(doc, final_svg)

    total_paths = sum(len(layer.paths) for layer in doc.layers)
    print(
        f"Vectorized {args.image} -> {len(doc.layers)} layer(s), {total_paths} path(s)."
    )
    print(f"Preview saved to {final_png}")
    print(f"SVG saved to {final_svg}")


if __name__ == "__main__":
    main()

