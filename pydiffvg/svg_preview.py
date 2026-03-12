"""SVG preview rasterization helpers for stroke-first outputs."""

from __future__ import annotations

from pathlib import Path


def _css_rgb(color: tuple[float, float, float] | None) -> str | None:
    if color is None:
        return None
    r, g, b = color
    r8 = max(0, min(255, int(round(float(r) * 255.0))))
    g8 = max(0, min(255, int(round(float(g) * 255.0))))
    b8 = max(0, min(255, int(round(float(b) * 255.0))))
    return f"rgb({r8},{g8},{b8})"


def render_svg_preview(
    svg_path: str | Path,
    output_png: str | Path,
    *,
    background_rgb: tuple[float, float, float] | None = (1.0, 1.0, 1.0),
) -> None:
    try:
        import cairosvg  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency failure
        raise RuntimeError(
            "SVG preview rendering requires the 'cairosvg' package. "
            "Install it in the active environment with 'pip install cairosvg'."
        ) from exc

    svg_file = str(Path(svg_path))
    png_file = str(Path(output_png))
    kwargs = {
        "url": svg_file,
        "write_to": png_file,
    }
    background_css = _css_rgb(background_rgb)
    if background_css is not None:
        kwargs["background_color"] = background_css
    try:
        cairosvg.svg2png(**kwargs)
    except Exception as exc:
        raise RuntimeError(f"Failed to rasterize SVG preview from '{svg_file}'") from exc


__all__ = ["render_svg_preview"]
