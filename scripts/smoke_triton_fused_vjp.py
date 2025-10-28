#!/usr/bin/env python3
"""Quick parity check for the Triton fused VJP mapper.

The script renders a simple stroke scene, extracts the cached splat data,
and compares gradients produced by `_map_triton_grads_to_slots` against the
reference gradients from the existing fallback/autograd path. It can run on
CPU (no Triton kernels required).
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import sys

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pydiffvg
from pydiffvg.splat.scene import _prepare_render_request
from pydiffvg.splat.mapping import map_triton_grads_to_slots, build_splat_mapping_payload
from pydiffvg.splat.debug import backward_tiled_full_python as _backward_tiled_full_python
from pydiffvg.splat.vjp import _enable_gradient_args
from pydiffvg.splat.geometry import _gather_specs
from pydiffvg.splat.gauss import _path_to_gaussians, _fill_to_gaussians
import pydiffvg.triton_splat as _triton


def _build_test_scene(width: int, height: int):
    # Single cubic stroke with mild curvature; deterministic for reproducibility.
    control_points = torch.tensor([[8.0, 12.0], [24.0, 20.0], [40.0, 28.0], [52.0, 40.0]], dtype=torch.float32)
    num_control_points = torch.tensor([2], dtype=torch.int32)
    path = pydiffvg.Path(
        num_control_points=num_control_points,
        points=control_points.clone(),
        stroke_width=torch.tensor(3.0, dtype=torch.float32),
        is_closed=False,
    )
    stroke_color = torch.tensor([0.8, 0.3, 0.1, 0.9], dtype=torch.float32)
    group = pydiffvg.ShapeGroup(
        shape_ids=torch.tensor([0], dtype=torch.int32),
        fill_color=None,
        stroke_color=stroke_color.clone(),
    )
    # Ensure gradients flow by flipping requires_grad on tensors we care about.
    path.points.requires_grad_(True)
    path.stroke_width.requires_grad_(True)
    group.stroke_color.requires_grad_(True)
    return [path], [group]


def main() -> int:
    os.environ.setdefault("DIFFVG_SPLAT_TILE", "32")
    pydiffvg.set_use_gpu(False)
    pydiffvg.set_backend("splat")
    torch.manual_seed(0)

    width = height = 64
    shapes, shape_groups = _build_test_scene(width, height)

    # Serialize scene and render once through the public API to get reference grads.
    args = pydiffvg.RenderFunction.serialize_scene(width, height, shapes, shape_groups)
    render = pydiffvg.RenderFunction.apply
    img = render(width, height, 2, 2, 0, None, *args)
    loss = img.sum()
    loss.backward()

    reference: dict[int, torch.Tensor] = {}
    for tensor in (shapes[0].points, shapes[0].stroke_width, shape_groups[0].stroke_color):
        if tensor.grad is None:
            raise RuntimeError("reference backward path produced no gradient")
        reference[id(tensor)] = tensor.grad.clone()
        tensor.grad.zero_()

    request = _prepare_render_request(width, height, 2, 2, 0, None, args)
    scene = request.scene
    device = pydiffvg.get_device()
    dtype = torch.float32
    stroke_specs, fill_specs = _gather_specs(scene, device, dtype)

    path_points_map = {payload.shape_id: payload.points for payload in scene.paths}

    batches = []
    seg_meta: list[Optional[torch.Tensor]] = []
    t_meta: list[Optional[torch.Tensor]] = []
    pts_meta: list[Optional[torch.Tensor]] = []
    cc_meta: list[list[int]] = []

    for spec in stroke_specs:
        gb = _path_to_gaussians(spec, request.config, device, dtype, request.generator)
        batches.append(gb)
        seg_meta.append(gb.seg_idx)
        t_meta.append(gb.t)
        cc_meta.append([len(seg.controls) for seg in spec.segments])
        pts_meta.append(path_points_map.get(spec.shape_id))
    for spec in fill_specs:
        gb = _fill_to_gaussians(spec, request.config, device, dtype, request.generator)
        batches.append(gb)
        seg_meta.append(gb.seg_idx)
        t_meta.append(gb.t)
        cc_meta.append([len(seg.controls) for seg in spec.segments])
        pts_meta.append(path_points_map.get(spec.shape_id))

    if not batches:
        raise RuntimeError("test scene produced no Gaussian batches")

    mu = torch.cat([b.mu for b in batches], dim=0)
    theta = torch.cat([b.theta for b in batches], dim=0)
    sigma_x = torch.cat([b.sigma_x for b in batches], dim=0)
    sigma_y = torch.cat([b.sigma_y for b in batches], dim=0)
    color_rgb = torch.cat([b.color_rgb for b in batches], dim=0)
    opacity = torch.cat([b.opacity for b in batches], dim=0)

    spec_counts = [int(b.mu.shape[0]) for b in batches]

    stroke_color_ref: dict[int, torch.Tensor] = {}
    fill_color_ref: dict[int, torch.Tensor] = {}
    for group in scene.shape_groups:
        if group.stroke.color_type is not None:
            for sid in group.shape_ids.to(torch.int64).tolist():
                stroke_color_ref[sid] = group.stroke.params[0]
        if group.fill.color_type is not None:
            for sid in group.shape_ids.to(torch.int64).tolist():
                fill_color_ref[sid] = group.fill.params[0]
    stroke_width_ref = {payload.shape_id: payload.stroke_width for payload in scene.paths}

    color_rgba_refs: list[Optional[torch.Tensor]] = []
    stroke_width_refs: list[Optional[torch.Tensor]] = []
    for spec in stroke_specs:
        color_rgba_refs.append(stroke_color_ref.get(spec.shape_id))
        stroke_width_refs.append(stroke_width_ref.get(spec.shape_id))
    for spec in fill_specs:
        color_rgba_refs.append(fill_color_ref.get(spec.shape_id))
        stroke_width_refs.append(None)

    tile_size = int(os.environ.get("DIFFVG_SPLAT_TILE", "32"))
    tile_ptr, tile_idx, tiles_x, tiles_y = _triton._build_tile_csr(
        mu, theta, sigma_x, sigma_y, width, height, tile_size
    )

    saved = {
        "mu": mu,
        "theta": theta,
        "sigma_x": sigma_x,
        "sigma_y": sigma_y,
        "color_rgb": color_rgb,
        "opacity": opacity,
        "tile_ptr": tile_ptr,
        "tile_idx": tile_idx,
        "tiles_x": tiles_x,
        "tiles_y": tiles_y,
        "tile_size": tile_size,
        "width": width,
        "height": height,
        "order": None,
        "spec_counts": spec_counts,
        "color_rgba_refs": color_rgba_refs,
        "stroke_width_refs": stroke_width_refs,
        "seg_idx_list": seg_meta,
        "t_list": t_meta,
        "points_refs": pts_meta,
        "control_counts": cc_meta,
    }
    saved.update(
        build_splat_mapping_payload(
            batches,
            len(stroke_specs),
            cc_meta,
            mu,
            spec_counts,
        )
    )

    grad_img = torch.ones_like(img)
    dcolor, dalpha, dmu_x, dmu_y, dtheta, disx, disy = _backward_tiled_full_python(
        mu, theta, sigma_x, sigma_y, color_rgb, opacity,
        tile_ptr, tile_idx, width, height, tile_size, grad_img,
    )
    sigma_x = saved["sigma_x"].clamp_min(1e-6)
    sigma_y = saved["sigma_y"].clamp_min(1e-6)
    dsx = -disx / (sigma_x * sigma_x)
    dsy = -disy / (sigma_y * sigma_y)

    args_with_grad, grad_slots = _enable_gradient_args(args)
    mapped = map_triton_grads_to_slots(
        saved,
        request,
        args_with_grad,
        grad_slots,
        dcolor,
        dalpha,
        dmu_x,
        dmu_y,
        dtheta,
        dsx,
        dsy,
    )

    assert mapped is not None, "fused mapping declined to produce gradients"

    max_diff = 0.0
    for slot in grad_slots:
        if id(slot.tensor) not in reference:
            # Non-parameter slots (e.g., tensors without grad) show up as zeros.
            continue
        fused = mapped[6 + slot.arg_index]
        ref = reference[id(slot.tensor)].to(fused.device, fused.dtype)
        diff_tensor = (fused - ref).abs()
        diff = diff_tensor.max().item()
        print(
            f"slot {slot.arg_index} shape={tuple(slot.tensor.shape)}"
            f" fused_sum={float(fused.abs().sum().item()):.3e}"
            f" ref_sum={float(ref.abs().sum().item()):.3e}"
            f" max_diff={diff:.3e}"
        )
        max_diff = max(max_diff, diff)

    tol = 1e-2
    if max_diff > tol:
        raise SystemExit(f"Fused VJP smoke test failed: max diff {max_diff:.3e} > {tol:.3e}")

    print("Smoke test passed: fused mapping matches fallback within tolerance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
