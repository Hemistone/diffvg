"""Helper pipeline to vectorize an image and render it with a diffvg backend."""

from __future__ import annotations

from typing import Any, Dict, Mapping, MutableMapping, Tuple

import torch

from ..renderer import Renderer
from ..save_svg import save_svg
from .api import ImageInput, VectorDoc, vectorize
from .adapters import vector_doc_to_scene


def vectorize_then_render(
    image: ImageInput,
    *,
    backend: str = "baseline",
    vectorize_kwargs: Mapping[str, Any] | None = None,
    serialize_kwargs: Mapping[str, Any] | None = None,
    samples: int | Tuple[int, int] = 4,
    seed: int = 0,
    background_image: torch.Tensor | None = None,
    save_svg_path: str | None = None,
) -> Dict[str, Any]:
    """Vectorize *image* and render it using the requested backend."""

    vectorize_args: MutableMapping[str, Any] = dict(vectorize_kwargs or {})
    doc: VectorDoc = vectorize(image, **vectorize_args)
    shapes, shape_groups = vector_doc_to_scene(doc)
    width, height = doc.canvas_size

    renderer = Renderer(backend=backend)
    serialize_args: MutableMapping[str, Any] = dict(serialize_kwargs or {})
    scene_args = renderer.serialize_scene(
        width,
        height,
        shapes,
        shape_groups,
        **serialize_args,
    )

    if isinstance(samples, int):
        samples_x = samples_y = samples
    else:
        samples_x, samples_y = samples

    rendered = renderer.apply(
        width,
        height,
        samples_x,
        samples_y,
        seed,
        background_image,
        *scene_args,
    )

    if save_svg_path is not None:
        save_svg(save_svg_path, width, height, shapes, shape_groups)

    return {
        "doc": doc,
        "shapes": shapes,
        "shape_groups": shape_groups,
        "scene_args": scene_args,
        "image": rendered,
    }


__all__ = ["vectorize_then_render"]
