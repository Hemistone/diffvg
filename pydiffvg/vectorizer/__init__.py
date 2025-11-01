"""Vectorization API surface."""

from .api import (
    PenSpec,
    Segment,
    Path,
    VectorLayer,
    VectorDoc,
    vectorize,
)
from .adapters import (
    vector_doc_to_scene,
    scene_to_vector_doc,
)

try:
    from .pipeline import vectorize_then_render
except ModuleNotFoundError as _exc:  # pragma: no cover - executed when diffvg is absent
    def vectorize_then_render(*args, **kwargs):  # type: ignore[override]
        raise RuntimeError(
            "vectorize_then_render requires the compiled diffvg backend"
        ) from _exc

__all__ = [
    "PenSpec",
    "Segment",
    "Path",
    "VectorLayer",
    "VectorDoc",
    "vectorize",
    "vector_doc_to_scene",
    "scene_to_vector_doc",
    "vectorize_then_render",
]
