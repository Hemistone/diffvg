# Vectorizer Integration Plan

This document summarizes the ongoing effort to integrate DrawingBotv3-inspired sketch algorithms into `pydiffvg`. It collects the
initial research notes, refined design decisions, and the remaining tasks so that individual “Start task” sessions can operate with
full project context.

## Objectives
- Introduce a CPU-first vectorizer that produces a `VectorDoc` intermediate representation compatible with existing diffvg backends.
- Keep the current rendering backends (`baseline`, `splat`) unchanged while adding adapters that convert `VectorDoc` into diffvg
  shapes.
- Ensure every module is directly runnable on CPU for validation during development.

## Architecture Overview
```
pydiffvg/
  vectorizer/
    api.py          # VectorDoc dataclasses, public vectorize() entrypoint
    adapters.py     # VectorDoc <-> diffvg scene conversion helpers
    edges.py        # Sobel/Canny edge detectors (NumPy/OpenCV)
    tone.py         # Brightness/edge density maps and integral images
    residual.py     # Residual map updates after stroke placement
    strokes.py      # Seed selection, stroke search, squiggle assembly
    bezier_fit.py   # Polyline -> Bézier least-squares fitting utilities
    svg_io.py       # Optional VectorDoc-specific SVG metadata helpers
```

`api.vectorize` orchestrates the edge detection, tone mapping, residual management, stroke synthesis, and optional Bézier fitting to
produce a layered `VectorDoc`. This intermediate form is then converted to diffvg `Shape`/`ShapeGroup` instances and fed to the
existing rendering pipeline.

## Completed Work
1. **IR & Package Skeleton** – `pydiffvg/vectorizer/` was created with dataclasses for `PenSpec`, `Segment`, `Path`, `VectorLayer`,
   and `VectorDoc`, plus the `vectorize` entry point (initial stub).
2. **Core Sketch Pipeline Modules** – CPU-first implementations for edges, tone, residual, strokes, and Bézier fitting were added,
   along with integration inside `api.vectorize`.
3. **Renderer Adapters & Pipeline Script** – `vectorizer/adapters.py` converts `VectorDoc` into diffvg scenes, `vectorizer/pipeline.py`
   provides a `vectorize_then_render` helper, and `apps/vectorize_then_refine.py` offers an end-to-end CLI for CPU validation.

## Remaining Tasks
The next stages focus on bridging the vectorizer output to diffvg renderers and establishing automated tests.

### 4. Tests & CI Hooks
- Create CPU-only regression tests (e.g., `tests/test_vectorizer.py`) that check stroke counts, average scores, or structural
  properties on small fixtures.
- Wire the tests into the existing automation (pytest command or Makefile hook) and document the command in `README.md` or here.

## Suggested “Start task” Cards
To keep future sessions scoped yet context-aware, launch tasks with the following instructions (copy/paste allowed):

1. **"Implement renderer adapters for VectorDoc"**
   - Convert `VectorDoc` into diffvg `Shape`/`ShapeGroup` objects inside `pydiffvg/vectorizer/adapters.py`.
   - Add a helper (e.g., `vectorize_then_render`) to run the vectorizer and dispatch to a chosen backend.
   - Produce or update `apps/vectorize_then_refine.py` to demonstrate the pipeline on CPU.

2. **"Add CPU regression tests for the vectorizer"**
   - Introduce fixtures under `tests/` that exercise stroke generation and Bézier fitting.
   - Ensure the command `python -m pytest tests/test_vectorizer.py` passes with `DIFFVG_CUDA=0`.
   - Document the test command in `docs/vectorizer.md` (this file) and reference it from `README.md` if appropriate.

## Manual Verification Checklist
Run these commands when developing or reviewing vectorizer changes:
- `DIFFVG_CUDA=0 python apps/vectorize_then_refine.py --image <path>`
- `python -m pytest tests/test_vectorizer.py`
- Optional debug helpers (if added): `python -m pydiffvg.vectorizer.debug_edges <image>` and
  `python -m pydiffvg.vectorizer.debug_strokes <image>`

Keeping this document updated ensures every contributor (human or agent) has the same context before tackling individual tasks.
