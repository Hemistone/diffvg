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
- [todo] Document stable public API surface in `pydiffvg/__init__.py`
- [todo] Add `__all__` and avoid wildcard re-exports in package modules
- [done] Factor `RenderFunction.serialize_scene` into `pydiffvg/serialization.py`
- [done] Add docstrings + type hints for `device.py`, `pixel_filter.py`, `shape.py`, `color.py`
- [done] Add docstrings/types for `image.py`, `save_svg.py`, and key functions in `parse_svg.py`
- [doing] Ensure `apps/single_*` use `pydiffvg.get_device()` consistently (majority updated)
- [done] Introduce `pydiffvg/dev.py` for debug toggles (e.g., `set_print_timing`) and wire into renderer
- [todo] Add a tiny smoke script for both CPU/CUDA: `scripts/test_render_paths.py` (no gradients)
- [done] CMake: add optional `-fsanitize=address,undefined` for host files via `-DDIFFVG_SANITIZE=ON`
- [doing] C++: create `bindings.cpp` (added empty TU) and plan gradual migration of pybind11 binds from `diffvg.cpp`
- [todo] C++: extract GPU sort dispatch; ensure CPU fallback without Thrust device when `DIFFVG_DISABLE_GPU_SORT=1`
- [todo] C++: add lightweight `public_api.h` exposing only functions used by bindings
- [done] Python: introduce a `Renderer` class (preallocation, reusable scene) — design stub only
- [todo] Add minimal perf harness under `apps/` to time color vs. sdf outputs
- [todo] Improve error messages for non-closed paths with fill (actionable hints)
- [todo] Add CONTRIBUTING notes for CPU vs CUDA builds and debugging tips


Proposed First Milestone (Week 1)

- Extract scene serialization (no behavior change) — done
- Add typing/docstrings in core Python modules (device, filter, shapes, color; plus image/save_svg/parse_svg) — done
- Normalize device usage in examples — doing

Proposed Second Milestone

- 2A: Backend scaffold — add backend selector/registry and minimal `Renderer` stub — done (see `pydiffvg/backend.py`, `pydiffvg/backends/registry.py`, `pydiffvg/renderer.py`)
- 2B: Add `bindings.cpp` and start moving pybind11 registration (leave algorithms in `diffvg.cpp`)
- 2B: Add `DIFFVG_SANITIZE` CMake option for host-only sanitizers (off by default)

Note: Bézier Splatting implementation tasks are tracked in `docs/bezier_splatting_todo.md`. This file only keeps prep work (scaffolding) needed to enable that backend.

Notes / Risks

- Avoid touching `scene.cpp` CUDA kernel boundaries early; first isolate bindings.
- Keep `render_pytorch.py` API untouched while extracting helpers; only internal imports change.
- GPU + CUDA toolchain inconsistencies can cause segfault; keep `PYTHONDONTWRITEBYTECODE=1` in dev and clean caches on crash.

Status Legend

- todo: not started
- doing: in progress
- done: merged and verified via examples
