# Flowline Preconditioning Mode — Draft Spec

**Status:** prototype implemented (2026-01-13)

This document proposes a **Flowline** preconditioning mode that generates long, coherent stroke paths by tracing along an orientation field derived from the image. The goal is to avoid skeleton-based "bubble" artifacts and produce painterly, pen-plotter-friendly strokes.

---

## 1) Goals

- Produce **long, coherent strokes** with consistent directionality.
- Reduce reliance on skeletonization, which tends to create short, fragmented polylines.
- Stay CPU-friendly by default; allow optional GPU acceleration for parts that benefit.
- Preserve compatibility with **splat backend constraints** (open paths, scalar stroke width, identity transform).

---

## 2) High-Level Pipeline

1. **Edge strength**
   - Use existing TEED/XDoG to compute a strength map `S ∈ [0,1]`.
   - Optional: TEED lineart intensity boost if enabled.

2. **Orientation field**
   - Compute structure tensor from image gradients (Sobel or Scharr).
   - Smooth tensor field with Gaussian blur (σ configurable).
   - Extract dominant **tangent direction** (per-pixel 2D unit vector) from eigenvectors.

3. **Seed selection**
   - Candidate pixels where `S >= seed_threshold` (quantile or fixed).
   - Prefer dark/strong edges; optionally enforce **min seed distance**.

4. **Greedy tracing**
   - Starting from seed, trace **forward & backward** along tangent direction.
   - Step size = `flow_step_px` (subpixel OK, use bilinear sampling of `S` and field).
   - Stop if strength falls below `flow_min_strength`, curvature too high, or max length reached.
   - Mark visited/covered regions with decay to reduce redundant strokes.

5. **Polyline postprocess**
   - Simplify (RDP), optional smoothing.
   - Clip or discard short polylines (< `flow_min_len`).
   - Convert to diffvg `Path` via existing vectorize code.

---

## 3) New Config Parameters (proposed)

Add to `PreconditionConfig`:

```
flow_edge_backend = "teed" | "xdog"
flow_seed_mode = "quantile" | "fixed"
flow_seed_quantile = 0.85
flow_seed_threshold = 0.2
flow_min_strength = 0.2
flow_step_px = 1.0
flow_max_len = 256
flow_min_len = 8
flow_min_seed_dist = 6
flow_curvature_deg = 60
flow_field_sigma = 2.0
flow_field_iters = 1
flow_coverage_decay = 0.85
flow_coverage_radius = 2
```

Notes:
- `flow_seed_quantile` uses the edge strength map distribution.
- `flow_field_sigma` controls smoothing of the structure tensor.
- `flow_coverage_decay` dims strength where strokes already traced to avoid stacking.

---

## 4) Orientation Field Details

Given grayscale image `I`:

```
Gx, Gy = sobel(I)
J = [[Gx^2, Gx*Gy], [Gx*Gy, Gy^2]]
J_blur = gaussian_blur(J, sigma=flow_field_sigma)
```

The tangent direction is orthogonal to the dominant gradient direction:
- Compute eigenvectors of `J_blur` (2x2 per pixel).
- Use **smallest eigenvalue vector** as tangent.
- Normalize to unit vector.

This gives a per-pixel flow field `T(x, y)` used for tracing.

---

## 5) Tracing Algorithm (sketch)

Pseudo:

```
for seed in seeds_sorted:
  if S(seed) < flow_min_strength: continue
  poly = trace(seed, dir=+1) + trace(seed, dir=-1)
  if len(poly) >= flow_min_len:
      add poly
      suppress S around poly (coverage_decay)
```

`trace` step:
- position p (float) → sample `T(p)` (bilinear)
- step p += dir * flow_step_px * T(p)
- append rounded pixel (or keep float)
- stop conditions:
  - length >= flow_max_len
  - strength < flow_min_strength
  - curvature > flow_curvature_deg

---

## 6) Integration Plan

1. Add new `mode = "flowline"` to `PreconditionConfig`.
2. Implement `pydiffvg/precondition/flowline.py` with:
   - `compute_flow_field(image_rgb, cfg)`
   - `trace_flowlines(strength, flow, cfg)`
3. Update `init_paths.py` to branch:
   - if `cfg.mode == "flowline"` → use flowline module → polylines → paths
4. Add CLI flags in `apps/painterly_rendering.py` and `apps/precondition_vectorize.py`.
5. Add preset under `configs/precondition_flowline.toml` (baseline settings).

---

## 7) Performance Notes

- CPU-only is acceptable for moderate resolutions (e.g., 512–1024 min-side).
- GPU acceleration is possible for structure tensor + filtering using torch if needed.
- Avoid GPU↔CPU round trips by keeping pipeline on one device when possible.

---

## 8) Open Questions

- Should flowline tracing be done on downscaled image then upscaled paths?
- Should we bias seed selection by darkness instead of edge strength?
- Should flowline paths be ranked by average strength × length (like current vectorize)?

---

## 9) Minimal MVP

- CPU-only flow field + greedy tracing
- No curvature constraint
- Use existing `vectorize.py` for simplification + path conversion
- Single config preset for quick testing

---

**End of draft**
