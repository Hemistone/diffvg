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

The older `baseline` and `splat` backends have now been removed from the
maintained tree. The repo no longer carries dual runtime paths.

This narrowing is also compatible with downstream sketch workflows such as
ControlSketch/SwiftSketch, whose current diffvg usage is stroke-only
(`fill_color=None`, open `Path`, constant `stroke_color`).

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

## Current Implementation Scope

Implemented in this phase:

- compiled open-stroke scene IR under `pydiffvg/openstroke/`
- `bezier_gsplat` serialization now compiles once and reuses live tensor refs
- global packed parameter banks for points, widths, and colors
- batched cubic sampling via cached Bernstein bases
- painterly/precondition/bench flows now default to the stroke-first backend
- legacy backends and their runtime modules removed from the mainline tree
- renderer-only microbenchmark harness in `apps/bench_renderer_micro.py`
- one small runtime smoke check in `apps/test_stroke_first_runtime.py`

Still intentionally deferred:

- cleanup of remaining historical docs that still mention removed backends
- compiler/round-trip truth checks beyond smoke coverage
- plotter-aware optimization objectives beyond hard palette/width constraints

## Benchmark Snapshot

The packed-bank refactor was the first change that materially removed the old
DiffVG runtime overhead from iterative painterly optimization.

`flower.jpg`, `64` iterations, `bezier_gsplat`, no precondition:

- previous branch, `512` paths: `141.95s`, loss `0.1271`
- stroke-first branch, `512` paths: `6.79s`, loss `0.1199`
- previous branch, `1024` paths: `306.46s`, loss `0.0465`
- stroke-first branch, `1024` paths: `6.78s`, loss `0.0540`

Interpretation:

- runtime scaling is now effectively flat between `512` and `1024` paths for
  this workload
- the largest removed cost was per-iteration scene/object packing, not the
  `gsplat` kernel itself
- speed improved enough that cleanup/stabilization now takes priority over
  another immediate renderer rewrite
- quality recovery at higher path counts is a follow-up phase, not part of the
  core runtime reboot milestone

## Stabilization Phase

The next maintained phase is not a feature expansion. It is a cleanup and
consolidation pass on top of the new runtime.

Priority order:

1. document the narrowed supported feature set and the legacy/modern boundary
2. remove or quarantine obviously dead runtime paths that duplicate the new
   open-stroke engine
3. keep only small validation coverage for invariants that would silently break
   the new runtime:
   - compile-once / render-many behavior
   - optimizer parameter binding to packed banks
   - SVG export/import round-trip for supported scenes
   - one app-level smoke benchmark for regression detection
4. defer optimizer-quality work until the runtime surface is cleaner

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
