# Bézier Splatting Research Note

This file is kept as a compact theory/reference note for the current
`bezier_gsplat` runtime.

It is **not** an implementation spec for the old native diffvg renderer.

## What We Took From The Paper

The maintained runtime uses the paper’s core idea:

- represent open strokes with sampled support points along Bézier geometry
- convert those samples into anisotropic 2D Gaussian-like splats
- rasterize with a fast splatting backend (`gsplat`)

The practical benefits for this repo are:

- strong scaling for high-stroke open-stroke workloads
- efficient backward passes for iterative painterly/vectorization loops
- a clean fit with packed tensor execution

## What Matters In This Repo

The parts of the paper that matter most here are:

- Bézier sampling
- tangent-aligned anisotropy
- splat-based differentiable rasterization
- time-to-quality advantages over exact stroke rasterization in the target
  workload regime

## What We Do Not Carry Forward

The maintained repo does **not** currently treat the paper as a prescription
for:

- fill support
- exact legacy diffvg parity
- multi-backend coexistence
- native C++/CUDA renderer integration

Those were earlier transition concerns and are no longer part of the maintained
direction.

## Working Interpretation

For this repo, “Bézier Splatting” means:

- a stroke-first packed runtime
- `bezier_gsplat` as the core renderer
- SVG as the canonical external artifact
- optimization quality and SVG fidelity as the next bottlenecks, not renderer
  existence proof

## Related Docs

- [`project_context.md`](project_context.md): current project identity and
  priority order
- [`stroke_first_reboot.md`](stroke_first_reboot.md): architecture reboot
- [`plotter_workflow_context.md`](plotter_workflow_context.md): downstream
  plotter-facing perspective
