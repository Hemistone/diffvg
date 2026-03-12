# Stroke-First Reboot

This document records the architectural change made after the initial
`bezier_gsplat` integration work.

## Why The Old Integration Was Not Enough

The first `bezier_gsplat` backend proved that `gsplat` could be wired into the
repo and used inside `painterly_rendering.py`, but it kept too much of the old
DiffVG runtime structure alive:

- scene serialization was still rebuilt around generic path/group payloads
- hot-path rendering still paid for Python object traversal and per-segment
  reconstruction
- the renderer still behaved like a generic diffvg adapter instead of a packed
  open-stroke engine

That structure was good enough to validate convergence and app compatibility,
but it was not faithful to the performance-critical design of the original
Bezier Splatting codebase.

## Current Direction

The maintained mainline path is now a stroke-first compiled renderer:

- public Python scene objects stay diffvg-like (`Path`, `ShapeGroup`, SVG I/O)
- the maintained runtime backend is `bezier_gsplat`
- the internal hot path compiles supported scenes into a packed open-stroke IR
- rendering is performed from this compiled representation with `gsplat`

Legacy backends still exist in the repo for comparison, but they are no longer
part of the maintained product path.

## Supported Mainline Scene Features

First-class support in the stroke-first renderer is intentionally narrow:

- open `Path` / open `Polygon`
- constant RGBA stroke colors
- scalar stroke widths
- identity `shape_to_canvas`
- `OutputType.color`
- line / quadratic / cubic segments, compiled into a fixed 3-segment cubic IR

Unsupported features are expected to fail explicitly instead of silently
falling back.

## Internal Representation

The new compiled scene path normalizes frontend shapes into a fixed hot-path
layout:

- internal primitive: fixed 3-segment cubic stroke
- shorter paths are padded and masked
- longer paths are split into multiple internal strokes
- line and quadratic segments are degree-elevated to cubic
- rendering samples are generated from cached Bernstein bases

This keeps the frontend flexible enough for SVG import/export while ensuring the
renderer sees a stable tensor layout.

## Runtime Behavior

`pydiffvg.Renderer.serialize_scene()` still returns a tuple that can be passed
unchanged into `Renderer.apply(...)`, so existing app-level calling conventions
remain intact.

For `bezier_gsplat`, that tuple now starts with a compiled scene object rather
than a legacy flat diffvg argument payload.

## Legacy Backends

`baseline` and `splat` are treated as legacy comparison paths.

- They remain available only when `DIFFVG_ENABLE_LEGACY=1` is set.
- Mainline defaults now point to `bezier_gsplat`.
- Benchmark helpers automatically re-enable legacy backends when a comparison
  run requests them.

## Current Implementation Scope

Implemented in this phase:

- compiled open-stroke scene IR under `pydiffvg/openstroke/`
- `bezier_gsplat` serialization now compiles once and reuses live tensor refs
- batched cubic sampling via cached Bernstein bases
- painterly/precondition/bench flows now default to the stroke-first backend
- legacy backends gated behind `DIFFVG_ENABLE_LEGACY`
- renderer-only microbenchmark harness in `apps/bench_renderer_micro.py`

Still intentionally deferred:

- removal of old exact-renderer code from the tree
- migration of every legacy sample app
- global packed parameter bank for eliminating remaining per-iteration tensor packing
- plotter-aware optimization objectives beyond hard palette/width constraints

## Benchmark Entry Points

- App-level painterly benchmark: `apps/bench_painterly_backends.py`
- Renderer-only microbenchmark: `apps/bench_renderer_micro.py`

The microbenchmark is the preferred tool for separating compile, forward,
backward, and one-step timing from app-level logging, SVG, and loss-pipeline
overhead.

## Files To Start From

- `pydiffvg/openstroke/compiler.py`
- `pydiffvg/openstroke/renderer.py`
- `pydiffvg/render_bezier_gsplat.py`
- `pydiffvg/backend.py`
- `pydiffvg/backends/registry.py`
- `apps/painterly_rendering.py`
- `apps/bench_painterly_backends.py`
- `apps/bench_renderer_micro.py`
