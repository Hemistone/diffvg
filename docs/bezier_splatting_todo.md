# Bézier Splatting — TODO & Integration Plan

Reference

- See docs/bezier_splatting.md for equations and method details (sampling strategy, Gaussian parameterization, alpha blending, backward).

Goals

- Add a Bézier Splatting renderer as an alternative backend without changing the public pydiffvg API.
- Deliver immediate GPU speedups for open curves; extend to closed fills; keep baseline renderer intact.

Phases

0) Scaffold & Switches

- [done] Backend selector: `pydiffvg.set_backend('baseline'|'splat')`, `pydiffvg.get_backend()`, env `DIFFVG_BACKEND`
- [done] Backend registry: `pydiffvg/backends/registry.py` (internal), register baseline + splat
- [done] Define backend_config (K, R, ρ, tile size, depth policy), defaults + env overrides

1) Forward (Open Strokes)

- [todo] New module `pydiffvg/render_splat.py` with `SplatRenderFunction` (forward only)
- [todo] Decompose paths into Bézier segments; sample t (eq. (3))
- [todo] Compute μ, σ_x, σ_y, θ (eq. (7)(8)), Σ (eq. (11)), α (eq. (10))
- [todo] Tiled alpha blending (eq. (9)); simple Torch implementation (CPU/CUDA)
- [todo] Unit test “forward sanity” image vs. baseline on simple scenes

2) Backward (Gradients)

- [todo] Save forward caches (Σ^{-1}, coverage masks, α accumulations) minimal set
- [todo] Implement custom backward to propagate dL/dμ, dΣ, dα → control points, stroke width, colors
- [todo] Gradient check: single cubic stroke, finite-diff validator

3) Closed Fills

- [todo] Paired-curve strip interpolation (eq. (4)(5)(6)); non-uniform tk near boundaries
- [todo] Depth policy for occlusion (small-area priority)
- [todo] Parity with even-odd rule at API boundary

4) Preallocation & Caching

- [todo] Introduce `Renderer` cache (tile bins, splat buffers), keyed by canvas + shapes signature
- [todo] Reuse allocations across iterations; expose light knobs to disable cache for debug

5) Performance & Kernels (Optional)

- [todo] Profile Torch implementation; identify hotspots (binning/sort/accumulation)
- [later] Triton/CUDA kernels guarded by feature flags; retain Torch fallback

Validation & Acceptance

- API: Existing examples run unchanged on baseline; splat opt-in via API/env
- Visual: Tolerance-based image checks on simple open/closed scenes
- Gradients: Finite-diff checks match analytic backward within small epsilon
- Perf: Measurable GPU speedups for open curves (target ≥ 10× forward on moderate scenes)

Risks & Mitigations

- Approximation artifacts at edges: document; increase K/R near high curvature; clamp σ
- Memory growth from many splats: cap per-tile splats; adapt K/R; shard tiles if needed
- CPU-only runs: keep slow Torch CPU path for tests; don’t block baseline
