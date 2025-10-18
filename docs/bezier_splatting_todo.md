# Bézier Splatting — TODO & Integration Plan

Reference

- See docs/bezier_splatting.md for equations, sampling, Gaussian parameterization, alpha blending, backward, and the integration notes targeting the modern pybind11-backed diffvg.

Goals

- Add a Bézier Splatting renderer as an alternative backend without changing the public pydiffvg API.
- Deliver immediate GPU speedups for open curves; extend to closed fills; keep baseline renderer intact.

Preintegration Prep (Serialization & Cache)

- [done] Extend `pydiffvg.serialize_scene` with an opt-in path that preserves tensors on `pydiffvg.get_device()`; keep baseline renderer on CPU while splat backend consumes the flag. (Profiling pending.)
- [done] Thread a `keep_on_device`/`device` hint through backend selection so splat can request device-resident tensors without touching public APIs.
- [todo] Audit shape/gradient handling once splat forward/backward exist to confirm autograd stability with GPU-resident tensors.
- [todo] Record profiling numbers (CPU vs CUDA) before/after enabling `keep_on_device` in the splat path to quantify the bandwidth win.

Phases

0) Scaffold & Switches

- [done] Backend selector: `pydiffvg.set_backend('baseline'|'splat')`, `pydiffvg.get_backend()`, env `DIFFVG_BACKEND`
- [done] Backend registry: `pydiffvg/backends/registry.py` (internal), register baseline + splat
- [done] Define backend_config (K, R, ρ, tile size, depth policy), defaults + env overrides
- [todo] Wiring doc links: point contributors at docs/bezier_splatting.md for rationale + API expectations

1) Forward (Open Strokes)

- [done] New module `pydiffvg/render_splat.py` with `SplatRenderFunction` (forward path + guarded fallbacks)
- [done] Reuse `pydiffvg.serialize_scene` to gather tensors; keep tensors on-device when supported and downcast to CPU before baseline fallback
- [done] Allocate tensors on `pydiffvg.get_device()`; pull K/R/ρ/tile from `pydiffvg.backend.get_backend_config()`
- [done] Decompose paths into Bézier segments; sample t (eq. (3))
- [done] Compute μ, σ_x, σ_y, θ (eq. (7)(8)), Σ (eq. (11)), α (eq. (10))
- [done] Tiled alpha blending (eq. (9)); Torch implementation (CPU/CUDA) with optional depth sort policy hook
- [todo] Unit test “forward sanity” image vs. baseline on simple scenes

2) Backward (Gradients)

- [done] Wrap forward in `torch.autograd.Function`; store lightweight request state and lean on Torch autograd until custom backward lands
- [todo] Implement custom backward to propagate dL/dμ, dΣ, dα → control points, stroke width, colors
- [todo] Clamp σ and α in forward/backward to avoid NaNs; respect `torch.cuda.amp` autocast
- [todo] Gradient check: single cubic stroke, finite-diff validator

3) Closed Fills

- [wip] Constant-color closed paths via radial interior refinement; extend toward paired-curve strip interpolation (eq. (4)(5)(6)) with non-uniform tk near boundaries
- [todo] Depth policy for occlusion (small-area priority)
- [todo] Preserve even-odd fill semantics and SVG round-trip at API boundary

4) Preallocation & Caching

- [todo] Introduce `Renderer` cache (tile bins, splat buffers), keyed by canvas + shapes signature
- [todo] Reuse allocations across iterations; expose light knobs to disable cache for debug

5) Performance & Kernels (Optional)

- [todo] Profile Torch implementation; identify hotspots (binning/sort/accumulation)
- [later] Triton/CUDA kernels guarded by feature flags; retain Torch fallback
- [later] Expose optional CUDA kernels via pybind11 helper while keeping Python entry point stable

Immediate Validation Queue

- [todo] Smoke-test `python apps/single_circle.py` with `DIFFVG_BACKEND=splat` (CPU + CUDA) and capture timing deltas against baseline.
- [todo] Render a simple open-path optimization loop (e.g., `apps/painterly_rendering.py`) under autograd to confirm gradients remain finite with the splat backend.
- [todo] Regression images: compare splat vs. baseline outputs on representative SVGs; log acceptable error bounds.

6) Integration & Determinism

- [done] Thread `SceneOptions.seed` through sampling to guarantee deterministic runs
- [done] Ensure backend toggle (`pydiffvg.set_backend`) leaves baseline behavior unchanged when unset (legacy `RenderFunction` now dispatches through registry; baseline regressions pending)

Validation & Acceptance

- API: Existing examples run unchanged on baseline; splat opt-in via API/env
- Visual: Tolerance-based image checks on simple open/closed scenes
- Gradients: Finite-diff checks match analytic backward within small epsilon
- Perf: Measurable GPU speedups for open curves (target ≥ 10× forward on moderate scenes)
- Determinism: Repeated renders with fixed seed match bit-for-bit

Risks & Mitigations

- Approximation artifacts at edges: document; increase K/R near high curvature; clamp σ
- Memory growth from many splats: cap per-tile splats; adapt K/R; shard tiles if needed
- CPU-only runs: keep slow Torch CPU path for tests; don’t block baseline
