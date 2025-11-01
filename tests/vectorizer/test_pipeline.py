"""Tests for :mod:`pydiffvg.vectorizer.pipeline`."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

from .utils import ROOT, load_vectorizer_module


class _RendererStub:
    """Record interactions with the pipeline's renderer."""

    last_instance: "_RendererStub | None" = None

    def __init__(self, backend: str) -> None:
        self.backend = backend
        self.serialize_args = None
        self.apply_args = None
        _RendererStub.last_instance = self

    def serialize_scene(self, width, height, shapes, shape_groups, **kwargs):
        self.serialize_args = {
            "width": width,
            "height": height,
            "shape_types": tuple(type(shape).__name__ for shape in shapes),
            "group_count": len(shape_groups),
            "kwargs": kwargs,
        }
        return ("scene_arg",)

    def apply(self, width, height, samples_x, samples_y, seed, background_image, *scene_args):
        self.apply_args = {
            "width": width,
            "height": height,
            "samples": (samples_x, samples_y),
            "seed": seed,
            "background": background_image,
            "scene_args": scene_args,
        }
        return "rendered-image"


def _install_pipeline_dependencies(monkeypatch):
    """Install lightweight stubs so the pipeline can be imported."""

    pkg = types.ModuleType("pydiffvg")
    pkg.__path__ = [str(ROOT / "pydiffvg")]
    monkeypatch.setitem(sys.modules, "pydiffvg", pkg)

    renderer_module = types.ModuleType("pydiffvg.renderer")
    renderer_module.Renderer = _RendererStub
    monkeypatch.setitem(sys.modules, "pydiffvg.renderer", renderer_module)

    save_svg_calls = []

    def fake_save_svg(path, width, height, shapes, shape_groups, **kwargs):
        save_svg_calls.append(
            {
                "path": Path(path),
                "width": width,
                "height": height,
                "shape_count": len(shapes),
                "group_count": len(shape_groups),
                "kwargs": kwargs,
            }
        )

    save_svg_module = types.ModuleType("pydiffvg.save_svg")
    save_svg_module.save_svg = fake_save_svg
    monkeypatch.setitem(sys.modules, "pydiffvg.save_svg", save_svg_module)

    return save_svg_calls


def test_vectorize_then_render_dispatches_to_renderer_and_svg(tmp_path, monkeypatch):
    save_svg_calls = _install_pipeline_dependencies(monkeypatch)

    pipeline = load_vectorizer_module("pipeline")
    api = load_vectorizer_module("api")

    pen = api.PenSpec(stroke_width=2.0, fill_color=(0.1, 0.2, 0.3, 1.0))
    segment = api.Segment(kind="line", points=((0.0, 0.0), (3.0, 1.5)))
    path = api.Path(segments=(segment,), closed=False, pen=pen)
    doc = api.VectorDoc(canvas_size=(12, 8), layers=(api.VectorLayer(paths=(path,)),))

    pipeline.vectorize = lambda image, **_: doc  # type: ignore[assignment]

    result = pipeline.vectorize_then_render(
        np.zeros((8, 12, 3), dtype=np.float32),
        backend="baseline",
        samples=(2, 3),
        seed=7,
        save_svg_path=str(tmp_path / "scene.svg"),
        serialize_kwargs={"foo": "bar"},
    )

    stub_instance = _RendererStub.last_instance
    assert stub_instance is not None
    assert stub_instance.backend == "baseline"

    assert isinstance(result, dict)
    assert result["doc"] is doc
    assert len(result["shapes"]) == 1
    assert len(result["shape_groups"]) == 1
    assert result["scene_args"] == ("scene_arg",)
    assert result["image"] == "rendered-image"

    assert save_svg_calls
    assert save_svg_calls[0]["path"].name == "scene.svg"
    assert save_svg_calls[0]["shape_count"] == 1
    assert save_svg_calls[0]["group_count"] == 1

    assert stub_instance.serialize_args is not None
    assert stub_instance.serialize_args["width"] == 12
    assert stub_instance.serialize_args["height"] == 8
    assert stub_instance.serialize_args["kwargs"] == {"foo": "bar"}

    assert stub_instance.apply_args is not None
    assert stub_instance.apply_args["samples"] == (2, 3)
    assert stub_instance.apply_args["seed"] == 7
    assert stub_instance.apply_args["scene_args"] == ("scene_arg",)

