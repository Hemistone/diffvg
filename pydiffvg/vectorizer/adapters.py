"""Conversion utilities bridging vectorizer data structures and diffvg primitives."""

from __future__ import annotations

from typing import List, Sequence, Tuple

from .api import VectorDoc
from .. import shape

ShapeLike = shape.Path | shape.Circle | shape.Ellipse | shape.Polygon | shape.Rect


def vector_doc_to_diffvg(doc: VectorDoc) -> Tuple[List[ShapeLike], List[shape.ShapeGroup]]:
    """Convert a :class:`VectorDoc` into diffvg shape/shape_group lists."""

    # Placeholder implementation – actual conversion logic will be implemented later.
    shapes: List[ShapeLike] = []
    shape_groups: List[shape.ShapeGroup] = []
    for layer in doc.layers:
        for path in layer.paths:
            raise NotImplementedError("Vector path conversion is not implemented yet")
    return shapes, shape_groups


def diffvg_to_vector_doc(
    shapes: Sequence[ShapeLike],
    shape_groups: Sequence[shape.ShapeGroup],
    *,
    canvas_size: Tuple[int, int],
) -> VectorDoc:
    """Create a :class:`VectorDoc` scaffold from diffvg primitives."""

    if shapes or shape_groups:
        raise NotImplementedError("diffvg to VectorDoc conversion is not implemented yet")
    return VectorDoc(canvas_size=canvas_size)
