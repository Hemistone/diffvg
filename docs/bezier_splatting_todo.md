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
- [done] Unified tracing via `DIFFVG_SPLAT_TRACE` for parity/instrumentation

1a) Forward (Open Strokes)

- [done] New module `pydiffvg/render_splat.py` with `SplatRenderFunction` (forward path + guarded fallbacks)
- [done] Reuse `pydiffvg.serialize_scene` to gather tensors; keep tensors on-device when supported and downcast to CPU before baseline fallback
- [done] Allocate tensors on `pydiffvg.get_device()`; pull K/R/ρ/tile from `pydiffvg.backend.get_backend_config()`
- [done] Decompose paths into Bézier segments; sample t (eq. (3))
- [done] Compute μ, σ_x, σ_y, θ (eq. (7)(8)), Σ (eq. (11)), α (eq. (10))
- [done] Tiled alpha blending (eq. (9)); Torch implementation (CPU/CUDA) with optional depth sort policy hook
- [done] Triton forward (optional): full‑frame chunked kernel + CSR‑tiled kernel (fp32). Guarded by `DIFFVG_SPLAT_IMPL=triton`; Torch remains the default and reference implementation.

1b) Quality Hardening (Open Strokes)

- [done] Length-adaptive sampling along arclength. Target ~1px spacing per segment via `K_seg = ceil(L_px/Δ)` with Δ≈1px (deterministic center-of-bin positions). Implemented in `render_splat._sample_path_geometry` using a 16-sample polyline arclength estimate; endpoints preserved.
- [done] σy calibration from stroke width via FWHM. Use `σ_y = width/(2·√(2ln2)·ρ)` for visually crisp, width-faithful splats (replaces `width/(2·ρ)`).
- [done] Spacing-aware opacity normalization (local). Per‑splat opacity can use a continuous coverage model with local spacing Δs: `α_i = 1 − (1 − α)^(Δs/(β·σx))`, with `β≈2.5` as an effective overlap width.
- [note] For correctness parity and to avoid under‑coverage on some scenes, the current default temporarily uses stroke alpha directly (spacing normalization disabled). Re‑enable behind a flag after Triton backward lands.
- [done] Centered θ at samples. Use central differences between neighbor sample positions for interior points; endpoints fall back to analytic tangents. This removes polyline-like kinks in orientation.
- [todo] Round caps/joins approximation. Add endpoint/turn handling to match baseline round caps/joins using a small cap fan or analytic ellipse aligned with the normal; keep minimal extra splats.
- [todo] Keep tiling differentiable. Rework tiled compositor to use additive accumulation (e.g., scatter/add or blockwise writes) so tiling can remain enabled under autograd; keep the current full-frame fallback as a debug path.
- [planned] Flat-top radial profile with a 2-Gaussian mixture to approximate a box interior and ~1 px edge rolloff; remove soft halo while preserving differentiability. Guarded by a config flag.
- [planned] Along-curve anisotropy knob: decouple σx from spacing with `σx = a·Δs`, `a ∈ (0,1]`, and fold into α re-normalization so coverage remains density-invariant.
- [planned] Cap/edge calibration to baseline AA: match endpoint caps and side-edge sharpness by aligning the edge-spread; optionally add a minimal cap fan or analytic end-ellipse.
- [planned] Fidelity tests: PSNR/SSIM vs baseline; edge lineouts (ESF/MTF) for straight strokes; small sweep over `ρ`, `a`, and mixture parameters to record matching presets.

2) Backward (Gradients)

- [done] Wrap forward in `torch.autograd.Function`; store lightweight request state and lean on Torch autograd until custom backward lands
- [done] Hybrid per‑tile Torch backward (sparse). Build per‑tile CSR over splats, recompute tile images with the same alpha‑over math, accumulate a single scalar loss, and call `torch.autograd.grad` on original inputs. Enabled by default for grad scenes when tiling>0.
- [done] Triton backward (per‑tile) — color + alpha (prefix T). Emits dL/dcolor and dL/dα per splat in tiles; reduces on device; VJP bridge to inputs.
- [done] Full Triton backward (per‑tile). Two‑pass (prefix/suffix) kernel emits per‑splat grads for {color, α, μx, μy, θ, 1/σx, 1/σy}; fused VJP maps directly to inputs; parity validated on painterly (tuning ongoing).
- [todo] Clamp σ and α in forward/backward to avoid NaNs; respect `torch.cuda.amp` autocast
- [todo] Gradient check: single cubic stroke, finite-diff validator
- [todo] Validate grads under length-adaptive sampling and spacing-aware α; ensure smoothness at joints/caps.

3) Closed Fills

- [wip] Constant-color closed paths via radial interior refinement; extend toward paired-curve strip interpolation (eq. (4)(5)(6)) with non-uniform tk near boundaries
- [todo] Depth policy for occlusion (small-area priority)
- [todo] Preserve even-odd fill semantics and SVG round-trip at API boundary
- [todo] Interior σ calibration and α normalization mirroring strokes so filled regions remain crisp; add tests with concave shapes and multiple subpaths.

4) Preallocation & Caching

- [todo] Introduce `Renderer` cache (tile bins, splat buffers), keyed by canvas + shapes signature
- [todo] Reuse allocations across iterations; expose light knobs to disable cache for debug

5) Performance & Kernels (Optional)

- [done] Profile Torch implementation; identify hotspots (binning/sort/accumulation). Added CSR‑tiled compositor; grid caching; optional Gaussian chunking.
- [done] Triton forward kernels guarded by feature flags (`DIFFVG_SPLAT_IMPL=triton`), with tiled CSR and full‑frame variants. Torch remains as fallback/reference.
- [done] Triton backward kernel (per‑tile) for color + alpha.
- [done] Triton backward kernel (geometry/width) functionally complete; parity validated (tuning pending).
- [done] Env knobs for backward launch: `DIFFVG_SPLAT_BWD_WARPS`, `DIFFVG_SPLAT_BWD_STAGES`, `DIFFVG_SPLAT_BWD_SCHUNK`.
- [todo] In‑kernel S‑chunking (per tile): process splats in fixed chunks (e.g., 64/128) with shared prefetch; reduces register pressure and atomic contention. Expected +10–25% on heavy‑overlap tiles.
- [planned] Per‑pixel backward variant: one program per pixel with small loops over tile splats (forward prefix + reverse suffix); block‑reduce then atomically add once per splat. Often higher occupancy; evaluate after fused per‑tile kernel.
- [todo] Early culling inside tile: compute a cheap mask for negligible contribution (e.g., (qx²+qy²) < r²) to skip work on empty pixels per splat.
- [todo] Launch tuning after refactors: pick per‑arch presets (e.g., BLOCK≈2048–4096, WARPS≈8, STAGES≈3 on SM89) and expose knob docs.
- [todo] Optional CUDA (C++) kernels after Triton parity (keep Python entry stable).

Immediate Validation Queue

- [todo] Kernel perf parity tests: tile‑program vs pixel‑program on 341×512, paths ∈ {64,128,256,512}, TILE ∈ {32,64}. Report Δt_backward and overall Δt; record best configs.
- [todo] Atomics contention microbench: sweep S‑chunk size (64/128/256) and measure scaling under high‑overlap tiles.
- [todo] Numerical checks with suffix scans enabled: verify dA/da_i stability near a→1 and low‑opacity regimes.
- [todo] “Crispness” checks: measure PSNR/SSIM vs. baseline on single-stroke scenes across widths and scales; assert no visible bead/blur at default settings.
- [todo] FD-grad check on points, stroke width, and color for cubic path; tolerance thresholds recorded in repo.

6) Integration & Determinism

- [done] Thread `SceneOptions.seed` through sampling to guarantee deterministic runs
- [done] Ensure backend toggle (`pydiffvg.set_backend`) leaves baseline behavior unchanged when unset (legacy `RenderFunction` now dispatches through registry; baseline regressions pending)

---

Appendix) Performance notes (painterly)

- On painterly configs (e.g., 341×512, 128 paths), splat+Triton backward shows smaller speedups than baseline C++/CUDA. Likely causes:
  - Python VJP bridge: geometry→Gaussian re-sampling and autograd graph build cost dominates. Next step: fuse VJP into Triton backward (direct grads to control points/widths).
  - CSR binning is on GPU (done), but current implementation uses sort+bincount; consider two‑pass no‑sort to reduce overhead in overlap‑heavy scenes.
  - Forward path: ensure Triton forward is enabled (`DIFFVG_SPLAT_IMPL=triton`) to avoid Torch compositing cost.
  - Atomics contention on overlap‑heavy tiles; tune `DIFFVG_SPLAT_BWD_SCHUNK`, `DIFFVG_SPLAT_WARPS`, `DIFFVG_SPLAT_STAGES`
  
Acceleration Roadmap

- [todo] Triton fused VJP kernel (single‑kernel): compute Bézier weights on‑the‑fly and atomically accumulate dμ/dθ/dσx into control points in one kernel to reduce launches and host overhead.
- [todo] GPU CSR no‑sort (two‑pass count→exclusive‑scan→scatter) to remove sort cost on overlap‑heavy tiles.
- [todo] GPU CSR no‑sort (two‑pass count→exclusive‑scan→scatter) to remove sort cost on overlap‑heavy tiles.
- [todo] CUDA Graph capture / `torch.compile` once kernel set is stable (no autograd VJP), to cut Python overhead per iteration.
- [todo] Policy for no‑grad + tiling → Triton tiled forward by default when requested (raster‑only workloads).
- [planned] Optional forward‑side caching of Bézier weights (wpos/wtan) in fp16 if profiling shows benefit in combination with fused kernel; keep disabled otherwise.
