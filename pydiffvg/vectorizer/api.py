"""Public dataclasses and entry points for the vectorization module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, Tuple, TypeGuard, Union

try:
    import numpy as _np
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    _np = None  # type: ignore[assignment]

try:
    import torch as _torch
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    _torch = None  # type: ignore[assignment]

from . import bezier_fit, residual, strokes, tone

if _np is not None:
    _NpArray = _np.ndarray
else:  # pragma: no cover - fallback for typing
    _NpArray = Any

if _torch is not None:
    _TorchTensor = _torch.Tensor
else:  # pragma: no cover - fallback for typing
    _TorchTensor = Any

ImageInput = Union[_NpArray, _TorchTensor]


@dataclass(slots=True)
class PenSpec:
    """Stroke/fill configuration shared by vector segments."""

    stroke_width: float = 1.0
    stroke_color: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    fill_color: Optional[Tuple[float, float, float, float]] = None


@dataclass(slots=True)
class Segment:
    """Individual segment making up a vector path."""

    kind: str
    points: Tuple[Tuple[float, float], ...]


@dataclass(slots=True)
class Path:
    """Collection of segments rendered with a shared :class:`PenSpec`."""

    segments: Tuple[Segment, ...] = field(default_factory=tuple)
    closed: bool = True
    pen: PenSpec = field(default_factory=PenSpec)


@dataclass(slots=True)
class VectorLayer:
    """A logical grouping of paths, roughly analogous to an SVG layer."""

    name: Optional[str] = None
    paths: Tuple[Path, ...] = field(default_factory=tuple)


@dataclass(slots=True)
class VectorDoc:
    """Top-level container describing a vectorized image."""

    canvas_size: Tuple[int, int] = (0, 0)
    layers: Tuple[VectorLayer, ...] = field(default_factory=tuple)


def _is_numpy_image(image: Any) -> TypeGuard[_NpArray]:
    """Return ``True`` if *image* looks like a NumPy ndarray."""

    return _np is not None and isinstance(image, _np.ndarray)


def _is_torch_image(image: Any) -> TypeGuard[_TorchTensor]:
    """Return ``True`` if *image* looks like a torch tensor."""

    return _torch is not None and isinstance(image, _TorchTensor)


def vectorize(
    image: ImageInput,
    *,
    num_layers: int = 1,
    mode: str = "lines",
    fit_bezier: bool = False,
    max_strokes: int = 64,
    seed_method: str = "nms",
    edge_weight: float = 0.5,
    erase_radius: int = 3,
) -> VectorDoc:
    """Vectorize *image* into a :class:`VectorDoc` using a light-weight pipeline."""

    if _np is None:  # pragma: no cover - guard when numpy is unavailable
        raise RuntimeError("NumPy is required for vectorize")
    if num_layers < 1:
        raise ValueError("num_layers must be >= 1")
    if mode not in {"lines", "quad", "cubic"}:
        raise ValueError("mode must be 'lines', 'quad', or 'cubic'")

    np_image = _image_to_numpy(image)
    height, width = np_image.shape[:2]

    saliency = tone.seed_probability_map(np_image, edge_weight=edge_weight)
    residual_map = _np.ones_like(saliency, dtype=_np.float32)
    cache = residual.ResidualKernelCache()

    seeds = strokes.select_seeds(saliency, count=max_strokes, method=seed_method)
    layer_paths: list[list[Path]] = [[] for _ in range(num_layers)]

    for idx, seed in enumerate(seeds):
        stroke_points = strokes.expand_squiggle(seed, saliency)
        if len(stroke_points) < 2:
            continue
        residual_map = residual.update_residual_map(residual_map, stroke_points, radius=erase_radius, cache=cache)
        saliency = tone.combine_brightness_and_edges(saliency, residual_map, edge_weight=0.3)
        layer_idx = idx % num_layers
        layer_paths[layer_idx].append(_points_to_path(stroke_points, mode=mode, fit_bezier=fit_bezier))

    layers = tuple(VectorLayer(paths=tuple(paths)) for paths in layer_paths)
    return VectorDoc(canvas_size=(width, height), layers=layers)


def _image_to_numpy(image: ImageInput) -> _NpArray:
    if _np is None:  # pragma: no cover - guard when numpy is unavailable
        raise RuntimeError("NumPy is required for vectorize")
    if _is_numpy_image(image):
        arr = _np.asarray(image)
    elif _is_torch_image(image):
        arr = image.detach().cpu().numpy()
    else:
        raise TypeError("Unsupported image type. Expected NumPy array or torch.Tensor.")
    if arr.ndim < 2:
        raise ValueError("Input image must have at least two dimensions")
    # Collapse batch dimension if present (e.g., PyTorch NCHW/NHWC).
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3:
        # Handle channel-first layouts by moving channels to the last axis.
        if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            arr = _np.transpose(arr, (1, 2, 0))
        return arr[..., :3].astype(_np.float32)
    return arr.astype(_np.float32)


def _points_to_path(points: Sequence[Tuple[int, int]], *, mode: str, fit_bezier: bool) -> Path:
    xy_points = [(float(x), float(y)) for y, x in points]
    simplified = xy_points
    if fit_bezier and mode in {"quad", "cubic"} and len(xy_points) >= (4 if mode == "cubic" else 3):
        simplified = bezier_fit.douglas_peucker(xy_points, epsilon=1.5)

    if mode == "lines":
        segments = _build_line_segments(points)
    elif mode == "quad":
        segments = _build_quad_segments(simplified if fit_bezier else xy_points, fit_bezier)
    else:  # mode == "cubic"
        segments = _build_cubic_segments(simplified if fit_bezier else xy_points, fit_bezier)

    return Path(segments=tuple(segments), closed=False)


def _build_line_segments(points: Sequence[Tuple[int, int]]) -> list[Segment]:
    cps = strokes.control_points(points, mode="lines")
    return [Segment(kind="line", points=tuple(cp)) for cp in cps]


def _build_quad_segments(points: Sequence[Tuple[float, float]], fit_bezier: bool) -> list[Segment]:
    if fit_bezier and len(points) >= 3:
        ctrl = bezier_fit.fit_quadratic(points)
        return [Segment(kind="quad", points=ctrl)]
    discrete = strokes.control_points([
        (int(round(y)), int(round(x))) for x, y in points
    ], mode="quad")
    return [Segment(kind="quad", points=tuple(cp)) for cp in discrete]


def _build_cubic_segments(points: Sequence[Tuple[float, float]], fit_bezier: bool) -> list[Segment]:
    if fit_bezier and len(points) >= 4:
        ctrl = bezier_fit.fit_cubic(points)
        return [Segment(kind="cubic", points=ctrl)]
    discrete = strokes.control_points([
        (int(round(y)), int(round(x))) for x, y in points
    ], mode="cubic")
    return [Segment(kind="cubic", points=tuple(cp)) for cp in discrete]
