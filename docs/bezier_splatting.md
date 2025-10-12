# Bézier Splatting for Fast and Differentiable Vector Graphics (Notes for neo-diffvg)

This document summarizes the paper “Bézier Splatting for Fast and Differentiable Vector Graphics Rendering” (arXiv:2503.16424) with the core equations and implementation details relevant to integrating Bézier Splatting into neo-diffvg while maintaining pydiffvg API parity.

> Legacy note: early drafts targeted the JAX prototype (`jaxdiffvg` branch). This revision aligns the guidance with the PyTorch backend while keeping that branch for archival reference.

## Overview

- Goal: Replace expensive boundary sampling and prefiltering in DiffVG-style rasterization with 2D Gaussian splatting sampled along Bézier curves (open strokes and closed fills).
- Key idea: Sample anisotropic 2D Gaussians along curves/interiors. Each Gaussian carries color and opacity, and alpha-blending handles compositing/occlusion. Gradients flow efficiently through the Gaussian formulation, greatly accelerating backward passes.
- Reported speedups (GPU): ~30× forward and ~150× backward vs DiffVG for open curves, while matching or improving fidelity.

## Primitives and Notation

- Bézier curve of degree M with control points P(i)
  j and normalized parameter t ∈ [0, 1]:
  
  Bi(t) = Σ_{j=0..M} B^M_j(t) P(i)_j  … (1)
  
  where B^M_j(t) = (M choose j) (1 − t)^{M−j} t^j  … (2)

- Curve attributes per curve i: color c_i ∈ R^3, opacity o_i ∈ [0, 1].
- Open curve representation: follow DiffVG; use 3 sequential, connected Bézier segments (the paper uses degree 4 segments; endpoints shared). Stroke has an extra width parameter (handled via Gaussian scales; see below).
- Closed curve representation: use a paired-curve structure (two boundary curves) and interpolate R intermediate curves to sample interior Gaussians efficiently (details below). Color is filled; opacity may vary per connected component/segment.

## Sampling Strategy

- Uniform sampling along a curve i:
  
  b_i = [ Bi(t_0), Bi(t_1), …, Bi(t_{K−1}) ], with t_k ∼ Uniform[0, 1]  … (3)

- Closed regions (paired-curve structure): Given two boundary curves B₁(t), B_{R+1}(t) that share endpoints, linearly interpolate corresponding control points to build R intermediates:
  
  P^{(k)}_j = (1 − t_k) P^{(0)}_j + t_k P^{(R+1)}_j,  k = 1..R  … (4)
  
  B^{interp}_k(t) = Σ_{j=0..M} B^M_j(t) P^{(k)}_j  … (5)
  
  Non-uniform t_k (from a normalized CDF) shrinks Gaussians near boundaries to reduce interior spill.

- Aggregate sampled points across the strip:
  
  X = [ b_0, b_1, …, b_{R+1} ] ∈ R^{(R+2) × K × 2}  … (6)

- Depth assignment: per-Gaussian depth d chosen so smaller curves are not fully occluded by larger ones (prevents gradients vanishing when accumulated opacity saturates).

## Gaussian Parameterization

- For each sampled point X_{r,k} (r: strip index, k: arclength sample), build an anisotropic 2D Gaussian aligned to local curve geometry:
  
  - Scales from neighbor distances:
    
    σ_x(r,k) = ||X_{r,k+1} − X_{r,k}||_2 / ρ,
    
    σ_y(r,k) = ||X_{r+1,k} − X_{r,k}||_2 / ρ  … (7)
    
    ρ is a global density/overlap control.
  
  - Orientation from local tangent:
    
    θ_{r,k} = atan2(y_{r,k+1} − y_{r,k−1}, x_{r,k+1} − x_{r,k−1})  … (8)

- Covariance from rotation and scales:
  
  Σ_i = (R_i S_i)(R_i S_i)^T,  where
  
  R_i = [ [cosθ_i, −sinθ_i], [sinθ_i, cosθ_i] ],  S_i = diag(σ_x, σ_y)  … (11)

- Per-Gaussian opacity inherits the curve’s opacity o_i (may be further modulated during optimization/pruning).

Notes:
- For open strokes, σ_y relates to stroke width; σ_x follows along-curve spacing. For closed fills, σ_x/σ_y derive from interpolated strip geometry.

## Rasterization (Alpha Blending)

- Use ordered alpha blending (front-to-back) over the set M of Gaussians covering a pixel:
  
  C_n = Σ_{i∈M} c_i α_i Π_{j < i} (1 − α_j)  … (9)

- Gaussian alpha from Mahalanobis distance of pixel center to Gaussian center:
  
  α_i = o_i · exp(−½ d_n^T Σ_i^{-1} d_n)  … (10)

- Sorting key: depth d (near to far) to enforce occlusion consistent with vector graphics layering.

## Optimization Objective and Dynamics

- Training objective for vectorization (I target vs rendered Ĩ):
  
  L = λ₁ ||Ĩ − I||²₂ + λ₂ · L_Xing  … (13)
  
  L_Xing from LIVE enforces convexity to stabilize strip interpolation in non-convex shapes.

- Adaptive pruning & densification (to escape local minima and reduce redundancy):
  - Prune curves with negligible opacity or very small/overly large Gaussians; remove highly overlapping curves via AABB overlap threshold.
  - Densify: add new curves in high-error regions (e.g., circle-initialized as in LIVE); match added count to removed to keep total constant.
  - Example settings (from experiments): opacity threshold 0.02; AABB overlap threshold 0.9; apply every ~400 steps; open curves optimized ~15k steps, closed ~10k steps. LR example: color 1e−2, control points 2e−4, opacity 1e−1.

## Implementation Notes for neo-diffvg (PyTorch)

- API parity: pydiffvg’s public API must remain unchanged. Internally, implement the PyTorch renderer so scene construction mirrors DiffVG semantics and tensors are created on `pydiffvg.get_device()` with the caller’s requested dtype.
- Open strokes:
  - Keep DiffVG’s poly-Bézier stroke representation; sample K along segments; compute θ, σ_x, σ_y via (7)(8). Derive σ_y from the stroke width and enforce contiguous tensors so vectorized kernels can stride across segments efficiently.
- Closed fills:
  - Map a filled path to a paired-curve strip for interior Gaussians (as in (4)(5)(6)). Preserve DiffVG’s even-odd fill semantics at the API boundary by constructing appropriate boundary pairs per subpath/winding parity. Serialization should continue to round-trip cleanly to SVG.
- Splatting kernel:
  - Build per-Gaussian parameters (μ, Σ, c, o, d) and perform tiled/blocked alpha blending in screen space. Sort by depth within tile; compute α via (10); composite via (9).
  - Express the kernel in vectorized PyTorch operations first; profile hotspots and optionally introduce custom CUDA/Triton kernels guarded by feature flags. `torch.compile` can wrap the forward pass once correctness is locked in.
- Differentiation:
  - Wrap the rasterization path in a `torch.autograd.Function` so forward allocates the minimal caches required for backward (e.g., Σ^{-1}, per-tile coverage masks, accumulated alphas). Backward should propagate gradients to Gaussian parameters and then to control points via the sampling map.
  - Clamp σ and α inside the custom backward to avoid NaNs; consider mixed-precision pathways gated by `torch.cuda.amp` if needed.
- Torch integration:
  - Return PyTorch tensors directly; respect the active `torch.Generator` seeded from `SceneOptions.seed` for deterministic sampling. Avoid inter-framework bridges; rely on standard Tensor APIs for downstream consumption.

## Practical Knobs and Defaults

- K (samples along t): higher for complex curvature; start modest (e.g., 16–64 per segment) and adapt if needed.
- R (interior strips for fills): 2–8 typically sufficient; use non-uniform tk near borders.
- ρ (density/overlap): controls σ magnitudes; tune to balance blur vs coverage; couple to pixel size.
- Depth policy: prioritize small-area curves to preserve detail.

## Minimal Forward Pass (high level)

1) For each path:
- Decompose into connected Bézier segments; tag open vs closed.
- Sample points: open via (3); closed via (4)(5)(6).
- Compute θ via (8); σ_x, σ_y via (7); set μ = X_{r,k}; set color/opacity from curve; assign depth.

2) Rasterize:
- Bin Gaussians to tiles; sort by depth per tile; for each pixel, accumulate via (9) using α from (10).

3) Return the image tensor; the `torch.autograd.Function` saves the intermediates needed for backward.

## Integration Considerations

- Even-odd and winding: keep DiffVG’s fill rule behavior externally. Internally, use the paired-curve strip to emulate fills; ensure parity for complex multi-subpath shapes.
- Serialization: emit standard SVG; preserve DiffVG quirks (linear sRGB, stroke width conventions). The Bézier Splatting internals must not leak into the public file format.
- Numerical stability: clamp σ to [σ_min, σ_max]; epsilon in Σ^{-1}; cap α to [0, 1].
- Determinism: seed sampling via SceneOptions.seed.

## References to Equations

- (1)(2): Bézier + Bernstein
- (3): Uniform sampling along curves
- (4)(5)(6): Closed-region interpolation and sampling strip
- (7)(8): Gaussian scales and orientation
- (9): Alpha compositing
- (10)(11): Gaussian alpha and covariance
- (13): Training objective

This note should be sufficient to implement a faithful, performant PyTorch rasterizer using Bézier Splatting under neo-diffvg, while keeping pydiffvg API behavior intact.
