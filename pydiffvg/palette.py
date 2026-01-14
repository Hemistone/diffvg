"""Palette loading and utility helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import tomllib
from typing import Iterable


@dataclass(frozen=True)
class PaletteEntry:
    name: str | None
    color_rgba: tuple[float, float, float, float]
    width_px: float | None = None
    width_mm: float | None = None

    def resolve_width_px(self, canvas_width: int, canvas_height: int) -> float:
        if self.width_px is not None:
            return float(self.width_px)
        if self.width_mm is None:
            raise ValueError("PaletteEntry missing width_px/width_mm")
        return _mm_to_px(float(self.width_mm), canvas_width, canvas_height)


@dataclass(frozen=True)
class Palette:
    name: str | None
    entries: tuple[PaletteEntry, ...]

    def resolve_widths(self, canvas_width: int, canvas_height: int) -> list[float]:
        return [entry.resolve_width_px(canvas_width, canvas_height) for entry in self.entries]

    def entry_for_index(self, index: int, canvas_width: int, canvas_height: int) -> tuple[PaletteEntry, float, int]:
        if not self.entries:
            raise ValueError("Palette has no entries")
        idx = int(index) % len(self.entries)
        width = self.entries[idx].resolve_width_px(canvas_width, canvas_height)
        return self.entries[idx], width, idx

    def entry_for_width(self, target_width: float, canvas_width: int, canvas_height: int) -> tuple[PaletteEntry, float, int]:
        if not self.entries:
            raise ValueError("Palette has no entries")
        widths = self.resolve_widths(canvas_width, canvas_height)
        best_idx = min(range(len(widths)), key=lambda i: abs(widths[i] - float(target_width)))
        return self.entries[best_idx], widths[best_idx], best_idx


def load_palette(ref: str) -> Palette:
    path = _resolve_palette_path(ref)
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Palette config must be a TOML table: {path}")
    raw_entries = data.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError(f"Palette must define a non-empty 'entries' list: {path}")

    entries: list[PaletteEntry] = []
    for idx, item in enumerate(raw_entries):
        if not isinstance(item, dict):
            raise ValueError(f"Palette entry #{idx} must be a table: {path}")
        pen_name = item.get("name")
        color_val = item.get("color")
        if color_val is None:
            raise ValueError(f"Palette entry #{idx} missing 'color': {path}")
        color = _parse_color(color_val)

        width_mm = item.get("width_mm")
        width_px = item.get("width_px")
        if width_mm is not None and width_px is not None:
            raise ValueError(f"Palette entry #{idx} specifies both width_mm and width_px: {path}")
        if width_mm is None and width_px is None:
            raise ValueError(f"Palette entry #{idx} must specify width_mm or width_px: {path}")
        if width_mm is not None:
            width_mm = float(width_mm)
        if width_px is not None:
            width_px = float(width_px)

        entries.append(
            PaletteEntry(
                name=str(pen_name) if pen_name is not None else None,
                color_rgba=color,
                width_px=width_px,
                width_mm=width_mm,
            )
        )

    palette_name = data.get("name")
    if palette_name is None:
        palette_name = path.stem
    return Palette(name=str(palette_name), entries=tuple(entries))


def _resolve_palette_path(ref: str) -> Path:
    candidate = Path(ref)
    if candidate.is_file():
        return candidate
    if candidate.suffix == "":
        with_suffix = candidate.with_suffix(".toml")
        if with_suffix.is_file():
            return with_suffix

    name = candidate.name
    if not name:
        raise FileNotFoundError("Palette name is empty")
    if not name.endswith(".toml"):
        name = f"{name}.toml"

    search_roots = [Path.cwd(), Path(__file__).resolve().parents[1]]
    for root in search_roots:
        probe = root / "configs" / "palette" / name
        if probe.is_file():
            return probe
    raise FileNotFoundError(f"Palette not found: {ref}")


def _mm_to_px(width_mm: float, canvas_width: int, canvas_height: int) -> float:
    if canvas_width <= 0 or canvas_height <= 0:
        return float(width_mm)
    if canvas_width >= canvas_height:
        a4_w_mm, a4_h_mm = 297.0, 210.0
    else:
        a4_w_mm, a4_h_mm = 210.0, 297.0
    px_per_mm = min(canvas_width / a4_w_mm, canvas_height / a4_h_mm)
    return float(width_mm) * px_per_mm


def _parse_color(value: object) -> tuple[float, float, float, float]:
    if isinstance(value, (list, tuple)):
        return _parse_color_sequence(value)
    if not isinstance(value, str):
        raise ValueError(f"Unsupported color type: {type(value)}")
    s = value.strip()
    if s.startswith("#"):
        return _parse_hex_color(s)
    m = re.match(r"^(rgba?|RGBA?)\((.+)\)$", s)
    if m:
        return _parse_rgba_function(m.group(2))
    raise ValueError(f"Unsupported color format: {value}")


def _parse_color_sequence(values: Iterable[object]) -> tuple[float, float, float, float]:
    vals = list(values)
    if len(vals) not in (3, 4):
        raise ValueError("Color sequence must have 3 or 4 components")
    channels = [_normalize_channel(v) for v in vals[:3]]
    alpha = 1.0 if len(vals) == 3 else _normalize_alpha(vals[3])
    return (channels[0], channels[1], channels[2], alpha)


def _parse_hex_color(value: str) -> tuple[float, float, float, float]:
    s = value.lstrip("#")
    if len(s) in (3, 4):
        s = "".join(ch * 2 for ch in s)
    if len(s) not in (6, 8):
        raise ValueError(f"Unsupported hex color: {value}")
    r = int(s[0:2], 16)
    g = int(s[2:4], 16)
    b = int(s[4:6], 16)
    a = 255 if len(s) == 6 else int(s[6:8], 16)
    return (_normalize_channel(r), _normalize_channel(g), _normalize_channel(b), _normalize_alpha(a))


def _parse_rgba_function(payload: str) -> tuple[float, float, float, float]:
    parts = [p.strip() for p in payload.split(",") if p.strip()]
    if len(parts) not in (3, 4):
        raise ValueError(f"rgba() expects 3 or 4 values, got {len(parts)}")
    rgb = [_parse_component(part) for part in parts[:3]]
    alpha = 1.0
    if len(parts) == 4:
        alpha = _parse_alpha_component(parts[3])
    return (rgb[0], rgb[1], rgb[2], alpha)


def _parse_component(part: str) -> float:
    if part.endswith("%"):
        val = float(part[:-1]) / 100.0
        return _clamp(val)
    return _normalize_channel(float(part))


def _parse_alpha_component(part: str) -> float:
    if part.endswith("%"):
        val = float(part[:-1]) / 100.0
        return _clamp(val)
    return _normalize_alpha(float(part))


def _normalize_channel(value: object) -> float:
    v = float(value)
    if v > 1.0:
        v = v / 255.0
    return _clamp(v)


def _normalize_alpha(value: object) -> float:
    v = float(value)
    if v > 1.0:
        v = v / 255.0
    return _clamp(v)


def _clamp(v: float) -> float:
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


__all__ = [
    "PaletteEntry",
    "Palette",
    "load_palette",
]
