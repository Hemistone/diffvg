# Flowline Preconditioning Mode

**Status:** research / prototype note

This is a research note for a future preconditioning path. It should be read in
the context of the current stroke-first runtime, not the removed legacy
backends.

This document proposes a **Flowline** preconditioning mode that generates long, coherent stroke paths by tracing along an orientation field derived from the image. The goal is to avoid skeleton-based "bubble" artifacts and produce painterly, pen-plotter-friendly strokes.

---

## 1) Goals

- Produce **long, coherent strokes** with consistent directionality.
- Reduce reliance on skeletonization, which tends to create short, fragmented polylines.
- Stay CPU-friendly by default; allow optional GPU acceleration for parts that benefit.
- Preserve compatibility with the **current stroke-first backend constraints**
  (open paths, scalar stroke width, identity transform).

---

## 2) High-Level Pipeline

1. **Edge strength**
   - Use existing TEED/XDoG to compute a strength map `S ∈ [0,1]`.
   - Optional: TEED lineart intensity boost if enabled.

2. **Orientation field (ETF)**
   - Compute image gradients and initial tangent field (perpendicular to gradient).
   - Iteratively smooth the tangent field using **Edge Tangent Flow (ETF)**.
   - ETF aligns local tangents into coherent stroke directions while preserving strong edges.

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
flow_field_sigma = 2.0  # ETF spatial smoothing sigma
flow_field_iters = 1    # ETF smoothing iterations
flow_coverage_decay = 0.85
flow_coverage_radius = 2
```

Notes:
- `flow_seed_quantile` uses the edge strength map distribution.
- `flow_field_sigma` controls ETF spatial smoothing.
- `flow_coverage_decay` dims strength where strokes already traced to avoid stacking.

---

## 4) Orientation Field Details

Given grayscale image `I`, compute the tangent field perpendicular to the gradient and
smooth it with ETF-style orientation averaging:

```
Gx, Gy = sobel(I)
T0 = normalize([-Gy, Gx])
theta = atan2(T0y, T0x)
for i in range(flow_field_iters):
  u = cos(2*theta) * |∇I|
  v = sin(2*theta) * |∇I|
  u = gaussian(u, sigma=flow_field_sigma)
  v = gaussian(v, sigma=flow_field_sigma)
  theta = 0.5 * atan2(v, u)
T = (cos(theta), sin(theta))
```

Using the **double-angle** representation lets us average orientations while treating
`theta` and `theta + π` as equivalent (directionless), then recover a unit tangent field
for tracing.

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
5. Add preset under `configs/precondition/flowline.toml` (starter settings).

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
