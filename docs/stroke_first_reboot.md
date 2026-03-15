# Stroke-First Reboot

This document records the architectural reset that turned the repo from a
legacy diffvg fork into a stroke-first runtime.

## Why The Reboot Happened

The first `bezier_gsplat` integration proved that `gsplat` could be wired into
the repo, but it kept too much of the old runtime shape alive:

- generic scene serialization every iteration
- Python object traversal in the hot path
- per-iteration repacking costs that hid the actual renderer gains

That version was enough to validate convergence, but not enough to justify the
repo’s long-term direction.

## The New Runtime Boundary

The maintained runtime is now:

- stroke-only
- open-path focused
- compiled once, rendered many times
- backed by `bezier_gsplat`

The maintained frontend remains diffvg-like:

- `Path`, `Polygon`, `ShapeGroup`
- `RenderFunction`, `Renderer`
- `save_svg`
- optional SVG parsing utilities

But the internal execution path is no longer the old diffvg renderer.

## Main Architectural Changes

### 1. Compiled open-stroke scene IR

Supported scenes are normalized into a packed internal representation under
`pydiffvg/openstroke/`.

Key ideas:

- fixed hot-path primitive: cubic stroke chunks
- compile once per topology
- keep parameter banks packed across iterations
- render directly from the compiled representation

### 2. `bezier_gsplat` as the only runtime backend

Legacy `baseline` and `splat` backends have been removed from the maintained
tree.

This repo now assumes:

- open-stroke workloads are the product path
- time-to-quality matters more than exact legacy parity
- old runtime branches are maintenance debt

### 3. SVG promoted to canonical output

Painterly artifacts now mean:

- `final_splatted.png`: internal renderer output
- `final.svg`: canonical vector artifact
- `final.png`: preview rasterized back from `final.svg`

This changed SVG export from a side utility into a core product step.

## What Was Removed

The reboot intentionally removed:

- native diffvg C++/CUDA renderer code
- legacy runtime backends
- fill/SDF demo paths from the maintained surface
- dual-path runtime complexity

## What Still Needs Work

The reboot solved architecture and runtime shape first.
It did **not** solve everything.

The next layer of work is:

1. quality stabilization for `bezier_gsplat`
2. SVG fidelity and export robustness
3. upstream NN integration and better initialization
4. plotter-aware postprocess and export

## File Landmarks

- `pydiffvg/openstroke/compiler.py`
- `pydiffvg/openstroke/renderer.py`
- `pydiffvg/render_bezier_gsplat.py`
- `pydiffvg/render_function.py`
- `apps/painterly_rendering.py`
- `apps/bench_renderer_micro.py`
- `apps/test_stroke_first_runtime.py`
