# Refactor Plan & TODO (diffvg modern fork)

Goals

- Precise: tight, well-defined module boundaries and minimal public surface.
- Concise: reduce monolithic modules; remove redundancy and globals.
- Expandable: enable future performance work (preallocation, profiling, CUDA/CPU parity) without large rewrites.

Current Layout (quick map)

- C++ core in repo root: `diffvg.cpp` (large), `scene.cpp` (CUDA TU), headers under root.
- Python API in `pydiffvg/`: rendering (`render_pytorch.py`), shapes/colors/utilities, SVG parse/save, device control.
- Examples in `apps/`; convenience scripts in `scripts/` and Makefile targets for smoke tests.
- Build via scikit-build/pyproject or `setup.py`; CUDA optional; Thrust from CUDA Toolkit.

Principles

- Keep behavior identical per step; prefer pure move/extract refactors.
- Land small, verifiable patches; run `make dev-check` (CPU) and a CUDA example when available.
- Avoid large file moves that disrupt blame unless necessary; introduce new files, then migrate callers.

Phases (high level)

1) Safety & DX polish (no behavior change)
2) Python API boundary cleanup (smaller modules, typing)
3) C++ bindings boundary cleanup (separate bindings vs. algorithms)
4) Performance scaffolding (renderer object, preallocation hooks)
5) Backend abstraction for new renderers (Bezier Splatting)

Milestone naming

- To avoid confusion, we’ll refer to the backend scaffolding work as “Milestone 2A: Backend scaffold (selector/registry + Renderer stub)”, which bridges Phase 4 → 5. The actual Bézier Splatting implementation remains tracked in `docs/bezier_splatting_todo.md`.

Acceptance checks per change

- `pip install .` succeeds (CPU and CUDA paths as applicable)
- `make dev-check` passes; selected `apps/single_*.py` run as before
- Diff in rendered images within tolerance for examples (manual/visual for now)

Backlog (small, incremental tasks)

- [done] Add `docs/refactor_todo.md` (this file) and keep updated
- [done] Document stable public API surface in `pydiffvg/__init__.py`
- [done] Add `__all__` and avoid wildcard re-exports in package modules (device, color, pixel_filter, shape, image, parse_svg, save_svg, render_pytorch, serialization, optimize_svg)
- [done] Factor `RenderFunction.serialize_scene` into `pydiffvg/serialization.py`
- [done] Add docstrings + type hints for `device.py`, `pixel_filter.py`, `shape.py`, `color.py`
- [done] Add docstrings/types for `image.py`, `save_svg.py`, and key functions in `parse_svg.py`
- [done] Ensure `apps/single_*` use `pydiffvg.get_device()` consistently (majority updated)
- [done] Introduce `pydiffvg/dev.py` for debug toggles (e.g., `set_print_timing`) and wire into renderer
- [done] Add a tiny smoke script for both CPU/CUDA: `scripts/test_render_paths.py` (no gradients)
- [done] CMake: add optional `-fsanitize=address,undefined` for host files via `-DDIFFVG_SANITIZE=ON`
- [done] C++: create `bindings.cpp` and continue migrating pybind11 binds from `diffvg.cpp` (remaining enums/functions still inline)
- [done] C++: split render host logic into `render.cpp` + `render_support.cpp` (keep each < 1k lines)
- [done] Build: remove vendored `pybind11` submodule; rely on external `pybind11>=3.0.1`
- [done] C++: extract GPU sort dispatch; ensure CPU fallback without Thrust device when `DIFFVG_DISABLE_GPU_SORT=1`
- [done] C++: add lightweight `public_api.h` exposing only functions used by bindings
- [done] Python: introduce a `Renderer` class (preallocation, reusable scene) — design stub only
- [done] Add minimal perf harness under `apps/` to time color vs. sdf outputs (`apps/perf_render_compare.py`)
- [done] Improve error messages for non-closed paths with fill (actionable hints)
- [done] Add CONTRIBUTING notes for CPU vs CUDA builds and debugging tips


Proposed First Milestone (Week 1)

- Extract scene serialization (no behavior change) — done
 - Add typing/docstrings in core Python modules (device, filter, shapes, color; plus image/save_svg/parse_svg) — done
 - Normalize device usage in examples — done

Proposed Second Milestone

- 2A: Backend scaffold — add backend selector/registry and minimal `Renderer` stub — done (see `pydiffvg/backend.py`, `pydiffvg/backends/registry.py`, `pydiffvg/renderer.py`)
- 2B: Add `bindings.cpp` and start moving pybind11 registration (leave algorithms in `diffvg.cpp`) — done
- 2B: Add `DIFFVG_SANITIZE` CMake option for host-only sanitizers (off by default) — done

Note: Bézier Splatting implementation tasks are tracked in `docs/bezier_splatting_todo.md`. This file only keeps prep work (scaffolding) needed to enable that backend.

Notes / Risks

- Avoid touching `scene.cpp` CUDA kernel boundaries early; first isolate bindings.
- Keep `render_pytorch.py` API untouched while extracting helpers; only internal imports change.
- GPU + CUDA toolchain inconsistencies can cause segfault; keep `PYTHONDONTWRITEBYTECODE=1` in dev and clean caches on crash.

Status Legend

- todo: not started
- doing: in progress
- done: merged and verified via examples

## Optimize SVG Refactor Roadmap

Objective: retire the 1,600-line `pydiffvg/optimize_svg.py` in favor of a composable package while keeping the public entrypoints stable and preserving optimization results byte-for-byte at each stage.

- [done] Stage 0 — Baseline & guardrails  
  Capture current behavior before moving code. The CPU-only smoke script `scripts/test_optimize_svg.py` (run via `python scripts/test_optimize_svg.py`) optimizes `apps/imgs/note_small.svg` toward a white target for five Adam steps on CPU and asserts the MSE drops by ≥0.005. Invariants: rely on default `SvgOptimizationSettings`, disable background optimization, and expect loss monotonicity when seeds are `[0..n)`. Document additional CLI entrypoints here once new front-ends appear.
- [done] Stage 1 — Package skeleton & settings extraction  
  Introduced `pydiffvg/optimize/__init__.py` and moved `SvgOptimizationSettings` (plus JSON helpers) into `pydiffvg/optimize/settings.py`, with `optimize_svg.py` re-exporting the class for compatibility. The script `scripts/test_optimize_settings.py` covers serialization, overrides, and round-trip save/load behavior.
- [done] Stage 2 — Transform utilities module  
  Relocated `TransformTools` (including decomposition helpers) into `pydiffvg/optimize/transforms.py` and aliased it back through `OptimizableSvg.TransformTools` for compatibility. Added a re-export via `pydiffvg.optimize` and verified the Stage 0/1 smoke tests still pass.
- [done] Stage 3 — Scene description data models  
  Moved the scene graph node classes (`SvgNode`, `PathNode`, etc.) into `pydiffvg/optimize/scene_graph.py`, wired them via a configure hook, and reattached them to `OptimizableSvg` to preserve the public surface. Existing parse/build logic now depends on the shared module, keeping behavior identical while isolating the data structures for future refinement.
- [done] Stage 4 — Parser & writer separation  
  Extracted the parsing logic into `SvgParserMixin` (`pydiffvg/optimize/parser.py`) and writing helpers into `SvgWriterMixin` (`pydiffvg/optimize/writer.py`). `OptimizableSvg` now inherits these mixins, isolating the XML/CSS dependencies from the core class while preserving behavior. Existing smoke tests remain green.
- [done] Stage 5 — Optimization driver orchestration  
  Added `SvgOptimizationDriver` under `pydiffvg/optimize/driver.py` to wrap `OptimizableSvg` with a small optimization loop (seed/scale schedules, callbacks) while keeping the original API intact. The driver exposes rendering/step helpers and returns per-iteration loss history for downstream integrations.
- [todo] Stage 6 — Cleanup & migration  
  Once stages 1–5 land, reduce `pydiffvg/optimize_svg.py` to deprecated shims, update documentation/examples to import from the new package, and run full CPU + CUDA smoke tests (apps and optimization script) before removing the monolith.
