# Plotter Workflow Context

This document captures the current engineering/research context for the
`feature/bezier-gsplat-backend` work. It is intended to help future sessions
recover the reasoning behind the current code layout and benchmark choices.

## Project Direction

The long-term goal is not just a faster raster reconstruction backend.
The target workflow is:

1. Input raster image
2. Open-stroke vector generation with a controllable artistic/sketch bias
3. Fixed pen palette and physically meaningful pen widths
4. SVG inspection/cleanup
5. G-code / pen-plotter export

This means backend speed matters, but it is not the only success criterion.
A renderer that reconstructs well but produces fragmented, travel-heavy stroke
fields is not yet good enough for the real plotter use case.

## Why We Kept diffvg As The Main Codebase

We reviewed the original `Bezier_splatting` codebase and decided **not** to use
it as a drop-in replacement for this repo.

Reasons:

- It is a research trainer/monolith, not a scene-graph-oriented renderer API.
- It does not directly replace `painterly_rendering.py` or the existing
  preconditioning/vectorizer pipeline.
- Its SVG export path is not plotter-oriented.
- It does not carry the palette/precondition/service-layer structure already
  present in this repo.

Instead, we kept this repo as the product/workflow shell and added a new
backend, `bezier_gsplat`, inside `pydiffvg`.

## What Has Been Implemented

### 1. `bezier_gsplat` backend

A new backend was added to `pydiffvg` as an internal alternative to `baseline`
and the older Triton `splat` path.

Main idea:

- keep diffvg scene/path abstractions
- sample open strokes into Gaussian-like support points
- use `gsplat` projection/rasterization for the renderer core
- keep the existing app/service layer (`painterly_rendering.py`, vectorizer,
  palette support, SVG output)

This is tracked by commit `bf9ac48`:

- `feat(pydiffvg): add bezier gsplat backend`

### 2. Painterly benchmark harness

`apps/bench_painterly_backends.py` benchmarks painterly runs across backends and
path counts.

Originally it only benchmarked render/runtime. It now also supports
plotter-style workflows and can emit plotter-oriented metrics.

### 3. Plotter-oriented SVG support

The following capabilities were added:

- `final.svg` is always written by `painterly_rendering.py`
- raw SVG plotter metrics
- conservative SVG cleanup
- CLI wrappers for both

Shared implementation lives in:

- `pydiffvg/plotter/metrics.py`
- `pydiffvg/plotter/cleanup.py`

CLI wrappers:

- `apps/svg_plotter_metrics.py`
- `apps/plotter_cleanup.py`

The point of this split is to keep logic centralized in `pydiffvg.plotter.*`
so benchmarks and CLI tools reuse the same code.

## Important Benchmark Findings

### 1. `bezier_gsplat` is not simply “always faster”

For short iterative runs it can outperform the older `splat` path, but for long
fixed-iteration painterly runs the older `splat` path may still have lower
per-iteration cost.

Observed pattern:

- `bezier_gsplat` often reaches a better loss in the same or similar wall time
- `splat` can still win on raw steady-state runtime for long fixed-iteration
  runs

Interpretation:

- `bezier_gsplat` currently has promising **time-to-quality** behavior
- it still needs per-iteration cost work to dominate long painterly loops

### 2. Random painterly + palette is not the right plotter benchmark

When `painterly_rendering.py` is run without preconditioning and with a palette,
colors are assigned by path index, not by image semantics.

That mode is useful to enforce fixed pen colors/widths, but it is **not** a
meaningful evaluation of palette-aware plotter output.

For plotter-focused evaluation, the better paths are:

- `--plotter-mode teed`
- `--plotter-mode flowline`
- `--plotter-mode lineart` with an explicit palette and mask settings

### 3. Fixed palette / fixed pen width makes cleanup much more meaningful

The cleanup pass is conservative and “layer strict”: it only reorders/merges
within the same color/layer bucket.

Therefore:

- random painterly outputs with near-unique per-stroke colors show only small
  cleanup gains
- fixed-palette outputs show much stronger gains, because many strokes actually
  share a pen bucket

This is the expected behavior, not a bug.

### 4. Single-pen TEED/flowline is currently the most honest plotter baseline

For actual pen plotting, the best current benchmark is often a single fixed pen
with preconditioned open strokes.

Why:

- it matches real machine constraints well
- it exposes pen-up travel and fragmentation clearly
- cleanup gains are easy to interpret

Multi-pen lineart is useful, but it is closer to a color-segmentation experiment
than to a stable “final” plotter pipeline.

## Bench Harness: Current Plotter Features

`apps/bench_painterly_backends.py` now supports:

- `--plotter-mode {teed,lineart,flowline}`
- default palette fallback to `single_black_pen`
- optional lineart mask overrides:
  - `--plotter-lineart-mask-count`
  - `--plotter-lineart-mask-mode`
- `--plotter-report`
  - analyze the run's `final.svg`
- `--plotter-cleanup`
  - run conservative cleanup and record cleaned metrics

This means a benchmark can now answer both:

- how long did the optimization take?
- how plotter-friendly was the resulting SVG before/after cleanup?

## Example Commands

Single-pen TEED benchmark:

```bash
source .venv/bin/activate
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python apps/bench_painterly_backends.py \
  apps/imgs/flower.jpg \
  --backends baseline,splat,bezier_gsplat \
  --path-counts 128,512 \
  --num-iter 8 \
  --repeats 1 \
  --warmup 0 \
  --plotter-mode teed \
  --plotter-report \
  --plotter-cleanup
```

Equivalent shorthand using the built-in bench preset:

```bash
source .venv/bin/activate
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python apps/bench_painterly_backends.py \
  apps/imgs/flower.jpg \
  --backends baseline,splat,bezier_gsplat \
  --path-counts 128,512 \
  --num-iter 8 \
  --repeats 1 \
  --warmup 0 \
  --plotter-preset single_black \
  --plotter-report \
  --plotter-cleanup
```

Multi-pen lineart benchmark:

```bash
source .venv/bin/activate
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python apps/bench_painterly_backends.py \
  apps/imgs/flower.jpg \
  --backends baseline \
  --path-counts 128 \
  --num-iter 8 \
  --repeats 1 \
  --warmup 0 \
  --plotter-mode lineart \
  --plotter-lineart-mask-count 3 \
  --palette /path/to/your_plotter_palette.toml \
  --plotter-report \
  --plotter-cleanup
```

Equivalent shorthand using the built-in bench preset:

```bash
source .venv/bin/activate
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python apps/bench_painterly_backends.py \
  apps/imgs/flower.jpg \
  --backends baseline \
  --path-counts 128 \
  --num-iter 8 \
  --repeats 1 \
  --warmup 0 \
  --plotter-preset trio \
  --plotter-report \
  --plotter-cleanup
```

The repo also includes preset files for these workflows:

- `configs/precondition/teed_plotter_single_black.toml`
- `configs/precondition/lineart_plotter_trio.toml`
- `configs/palette/plotter_trio.toml`

## Current Open Questions

1. How should plotter quality be folded into optimization, not just postprocess?
   Examples: short-stroke penalties, endpoint stitching priors, travel-aware
   ordering/layering objectives.

2. Should palette-aware cleanup become stricter or more semantic?
   Current cleanup is conservative by design.

3. How should NN-based upstream generators (`Clipasso`, `ControlSketch`, etc.)
   integrate with this pipeline?
   Generic geometric merging is risky once stroke semantics matter.

4. What is the right balance between reconstruction quality and plotter-friendly
   stroke structure?

## Practical Rule Of Thumb

When comparing backends for this project, do not stop at final loss or runtime.
Check at least these three dimensions together:

1. optimization/runtime
2. final raster quality
3. plotter metrics (`travel_ratio`, fragmentation, stroke count, cleanup gain)

That triad better reflects the actual product goal than any single metric.
