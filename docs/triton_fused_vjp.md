# Triton Fused VJP — Plan, Notes, and Acceptance

Goal

- Build a single (or single‑stage) Triton backward that maps splat‑space grads directly to scene parameters without the Python autograd/VJP bridge.
- Target parameters: control points, stroke width, and RGBA color. Keep numerical guards and maintain parity with the current Python fused_map path.
- When the work is complete, flip the entry in docs/bezier_splatting_todo.md:111 from [todo] to [done].

Scope

- Backends: pydiffvg splat backend only. Baseline renderer untouched.
- Inputs (from forward ctx): SoA (mu, theta, inv_sigma_x, inv_sigma_y, color_rgb, opacity), tile CSR (tile_ptr, tile_idx, tiles_x/tiles_y, tile_size), optional depth order permutation and its inverse.
- Mapping: per‑splat metadata that links a Gaussian sample back to (spec_id, seg_idx, t) and to the originating tensors (color RGBA, stroke width, points tensor + control layout).

Design Overview

- Phase 1 (partial fused mapping on GPU):
  - Keep the existing Triton tiled backward for {color, opacity, mu, theta, 1/sigma_x, 1/sigma_y}.
  - Add a GPU‑resident reduce+scatter stage that consumes per‑splat grads and accumulates directly into:
    - per‑spec RGBA grads (sum across splats belonging to the spec)
    - per‑path stroke_width grad via sigma_y mapping
    - per‑path control‑point grads via the current geometry chain rule.
  - Replace Python fused_map + autograd VJP in pydiffvg/render_splat.py with this GPU stage. Retain the Python path as a fallback.

- Phase 2 (fully fused):
  - Extend the Triton tiled backward kernel to directly index_add atomically into the parameter buffers (points/width/color) using the per‑splat mapping, avoiding the extra GPU post‑pass.
  - Keep an env flag to switch back to Phase 1 mapping for debugging.

Saved‑State Schema (ctx.splat_saved)

- Existing (kept): `mu, theta, sigma_x, sigma_y, color_rgb, opacity, tile_ptr, tile_idx, tiles_x, tiles_y, tile_size, width, height, order, spec_counts, color_rgba_refs, stroke_width_refs, seg_idx_list, t_list, points_refs, control_counts`.
- New per‑splat tables (forward builds once; stored on device):
  - `spec_id[S]` (int32): which spec a sample belongs to.
  - `seg_idx[S]` (int32): segment index within that spec (already available; flatten if needed).
  - `t[S]` (float32): parametric t per sample (already available).
  - `deg_flags[S]` (int8 or 2 bits): quadratic vs cubic.
  - `cp_idx0[S..S+3]` (int32): control point indices in a flattened points pool for si0, ci0, ci1, si_end (quadratic uses si0, ci0, si_end).
  - `points_pool` (float32, flat view): optional flat buffer view of concatenated points to enable single index_add; or keep per‑tensor index_add with id→offset table.
  - `spec_color_idx[Nspec]`/`spec_width_idx[Nspec]` (int32): indices for color/width params. If per‑spec tensors are separate, store an id→offset map instead.
  - `order_inv[S]` (int32, optional): inverse permutation to restore pre‑sort order if DepthPolicy sorting is used.

Math/Mapping (parity with current Python path)

- Reference implementation to mirror: pydiffvg/render_splat.py:656–950.
- Color/Opacity: sum dcolor over splats in spec; sum dalpha to opacity channel; reshape to RGBA like the original tensor shape.
- Stroke width: sigma_y = width / (2·sqrt(2ln2)·rho). Thus d(width) = sum(dsigma_y) · (1 / (2·sqrt(2ln2)·rho)).
- Control points: use the same chain rule as in the Python fused_map path:
  - dmu → points via basis derivative (cubic/quadratic Bernstein). See pydiffvg/render_splat.py and docs/bezier_splatting.md for equations.
  - dtheta path: convert orientation grads to orthogonal components wrt local tangent; use analytic tangent of Bézier at t. Normalize with small epsilon and apply the cross‑term mapping used in Python path for stability.
  - dsigma_x path: propagate along‑curve spacing dependence (neighbors) exactly as the current implementation, including endpoint handling and central‑diff interior.
  - Accumulate with index_add into si0/ci0/ci1/si_end using the same weights as the Python code to preserve parity.

Kernel Plan

- Start with a GPU reduce stage (Torch CUDA or small Triton kernels) that:
  - Reorders grads back to pre‑sort order via `order_inv` when needed.
  - Performs segmented reductions by `spec_id` for RGBA and width.
  - Applies per‑splat chain rules to compute control‑point contributions and `index_add_` into a flat points buffer (or per‑tensor shards if we keep per‑path tensors distinct).
- Then, integrate the above mapping into the existing tiled backward kernel as an optional fused path guarded by an env flag.

Flags & Controls

- `DIFFVG_SPLAT_IMPL=triton` → require the Triton forward/backward path (error if unavailable); default keeps automatic fallback.
- `DIFFVG_SPLAT_TRACE=1` → emit debug traces when the fused mapper falls back to the Python bridge (reason tagged once per process).
- Core tunables: `DIFFVG_SPLAT_TILE` (tile size), `DIFFVG_SPLAT_CHUNK` (per-launch chunk size), `DIFFVG_SPLAT_BWD_WARPS`, `DIFFVG_SPLAT_BWD_STAGES`, `DIFFVG_SPLAT_BWD_SCHUNK`.
- Device override: `DIFFVG_DEVICE=cpu|cuda[:index]` (defaults to CUDA when available).

Numerics & AMP

- Clamp in forward/backward: sigma ≥ 1e−3, (1−alpha) ≥ 1e−6; log‑domain transmittance kept in backprop to prevent underflow.
- AMP: keep kernel math in FP32, allow storage of SoA/mapping buffers in FP16/BF16 via `DIFFVG_SPLAT_DTYPE` after PSNR/LPIPS checks.
- Sanitize outputs: `nan_to_num` on grads before accumulation.

Acceptance & Benchmarks

- Finite differences: single cubic stroke, perturb control points/width/color; assert relative error below threshold (e.g., <1e−3 to 1e−2 depending on scale).
- Painterly microbench: 341×512, paths ∈ {128, 256, 512}, TILE ∈ {32, 64}, LPIPS on/off.
  - Report: Δt_backward, Δt_iter, PSNR/LPIPS deltas vs current main.
  - Expectation: backward 1.4–2.0× faster; end‑to‑end 1.3–1.6× on LPIPS configs.
- Determinism: seed fixed; ensure run‑to‑run stable outputs within numerical jitter.

Fallbacks

- If Triton fused path yields non‑finite or near‑zero grads, fall back to the Python reference (or to Phase 1 GPU mapping), unless `DIFFVG_SPLAT_IMPL=triton` enforces strict.
- Preserve the current autograd VJP bridge as a last‑resort fallback.

Work Items (Checklist)

1) Forward: build/store per‑splat mapping tables in ctx.
2) Backward Phase 1: GPU reduce+scatter mapping (Torch CUDA or small Triton kernels).
3) Wire env/flags and fallbacks; update render_splat backward to bypass autograd VJP when fused mapping succeeds.
4) Backward Phase 2: integrate mapping into tiled Triton kernel (optional; flag‑guarded).
5) Add clamps/AMP guards shared by kernels.
6) Add gradient tests + painterly microbench harness output.
7) Tune presets; document recommended presets.
8) Finally, update docs/bezier_splatting_todo.md:111 to [done].

References

- docs/bezier_splatting_todo.md:111 (Acceleration Roadmap — Triton fused VJP kernel)
- pydiffvg/render_splat.py:656 (Python fused_map reference implementation)
- pydiffvg/splat/triton/backward.py (current tiled backward kernels)
- docs/bezier_splatting.md (equations, chain rules, numerical notes)

## Progress Log

- 2024-10-28: Forward caches now stash per-splat metadata via `build_splat_mapping_payload`; added `_map_triton_grads_to_slots` helper and smoke test (`scripts/smoke_triton_fused_vjp.py`) verifying parity with the existing autograd path. Remaining work: GPU reduce/scatter, kernel fusion, benchmarks, and doc status flip.
- 2024-10-29: Implemented the scripted GPU reduce/scatter mapper (`_map_triton_grads_to_slots_gpu`) that consumes the cached metadata to accumulate color/opacity, stroke width, and control point grads directly. With `DIFFVG_SPLAT_IMPL=triton` the fused mapper now runs (and is required) end-to-end. Trimmed the mapper to operate directly on per-spec tensors, added Triton reductions for color/opacity/width + point scatter, and updated fallbacks to emit debug reasons. Next up: extend kernels for θ central-diff, add AMP guards, benchmark painterly on-GPU, and flip the TODO once fused consistently wins.
