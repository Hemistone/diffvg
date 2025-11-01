"""Public dataclasses and entry points for the vectorization module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Tuple, TypeGuard, Union

try:
    import numpy as _np
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    _np = None  # type: ignore[assignment]

try:
    import torch as _torch
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    _torch = None  # type: ignore[assignment]

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


def vectorize(image: ImageInput, *, num_layers: int = 1) -> VectorDoc:
    """Vectorize an input raster *image* into an abstract :class:`VectorDoc`.

    Parameters
    ----------
    image:
        Raster image to vectorize. Both NumPy arrays and ``torch.Tensor`` inputs
        are accepted.
    num_layers:
        Desired number of vector layers in the output document. The current
        implementation only validates the input and returns an empty scaffold.
    """

    if num_layers < 1:
        raise ValueError("num_layers must be >= 1")

    width: int
    height: int
    if _is_numpy_image(image):
        if image.ndim < 2:
            raise ValueError("NumPy image input must have at least 2 dimensions")
        if image.ndim == 2:
            height, width = (int(dim) for dim in image.shape)
        else:
            height = int(image.shape[-3])
            width = int(image.shape[-2])
    elif _is_torch_image(image):
        if image.dim() < 2:
            raise ValueError("torch image input must have at least 2 dimensions")
        height = int(image.shape[-2])
        width = int(image.shape[-1])
    else:
        raise TypeError("Unsupported image type. Expected NumPy array or torch.Tensor.")

    return VectorDoc(canvas_size=(width, height), layers=tuple(VectorLayer() for _ in range(num_layers)))
