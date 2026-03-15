# Project Context

This repository is no longer a general-purpose differentiable SVG renderer.
Its maintained purpose is narrower and more practical:

- build a fast open-stroke vectorization engine
- keep a diffvg-like Python-facing interface for downstream tools
- support painterly / precondition / line-art workflows on top of that engine
- preserve a path toward fixed-palette, fixed-pen-width plotter output

## Core Problem

The target workflow is:

1. raster input image
2. open-stroke vector generation
3. optional upstream NN-guided initialization (`ControlSketch`, `Clipasso`, etc.)
4. canonical SVG output
5. optional cleanup / plotter export

The engine is judged by:

- time-to-quality on high-stroke open-stroke workloads
- stability of optimization
- fidelity of the final SVG artifact
- compatibility with future NN-based sketch pipelines

It is **not** judged by support for arbitrary SVG fills, exact legacy diffvg
rendering, or old C++/CUDA renderer parity.

## Maintained Identity

The maintained product identity is:

**a stroke-first differentiable vector engine with a diffvg-like Python API**

That means:

- `bezier_gsplat` is the only supported runtime backend
- the internal execution path is compiled, packed, and stroke-only
- exact/fill-heavy legacy renderer paths are out of scope
- old native diffvg C++/CUDA code is no longer part of the maintained branch

## Current Priority Order

Priority order should stay:

1. core runtime stability
2. canonical SVG export fidelity
3. optimizer quality stabilization
4. upstream NN integration
5. plotter tooling / cleanup / G-code export
6. artistic stylization

This ordering is intentional.

- Plotter tooling matters, but it is downstream of the engine.
- Stylization matters, but it is downstream of optimization quality.
- Benchmarking mattered to choose the engine; it is no longer the main task.

## Why `bezier_gsplat`

`bezier_gsplat` was kept because it won in the workload that matters most here:

- high path count
- long iterative optimization
- open-stroke painterly/vectorization
- time-to-threshold-loss rather than fixed-iteration vanity comparisons

It is not perfect yet. Known weak points include:

- preconditioned runs can develop orthogonal scaffold artifacts
- long-run quality still needs stabilization
- SVG export fidelity is now a first-class concern because `final.svg` is the
  canonical artifact

Those are considered **fixable quality/runtime-layer issues**, not reasons to
return to the removed legacy renderers.

## What To Avoid

Do not reintroduce the following as maintained-surface goals:

- arbitrary fill support
- exact legacy diffvg renderer parity
- native C++/CUDA renderer rebuilds
- keeping dead backends or docs around "just in case"

If a historical idea is worth preserving, keep it as a research note under
`docs/research/`, not as a maintained code path.
