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
    vector_doc_to_diffvg,
    diffvg_to_vector_doc,
)

__all__ = [
    "PenSpec",
    "Segment",
    "Path",
    "VectorLayer",
    "VectorDoc",
    "vectorize",
    "vector_doc_to_diffvg",
    "diffvg_to_vector_doc",
]
