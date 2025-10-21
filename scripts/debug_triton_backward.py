#!/usr/bin/env python
"""
Compare Triton backward gradients against the Python reference and the pure-Torch
kernel-formula mirror on a tiny tile. Helpful for auditing kernel math without
running the full renderer.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Iterable, Tuple

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from pydiffvg.splat import triton
from pydiffvg.splat.debug import (
    backward_tiled_full_kernel_formula,
    backward_tiled_full_python,
)


def build_case(seed: int, device: torch.device, n_gaussians: int) -> Tuple[torch.Tensor, ...]:
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    mu = torch.rand((n_gaussians, 2), generator=rng, device=device) * 4.0
    theta = torch.rand((n_gaussians,), generator=rng, device=device) * 2.0 - 1.0
    sigma_x = torch.rand((n_gaussians,), generator=rng, device=device) * 0.5 + 0.6
    sigma_y = torch.rand((n_gaussians,), generator=rng, device=device) * 0.5 + 0.6
    color = torch.rand((n_gaussians, 3), generator=rng, device=device)
    opacity = torch.rand((n_gaussians,), generator=rng, device=device) * 0.6 + 0.2

    tile_ptr = torch.tensor([0, n_gaussians], dtype=torch.int32, device=device)
    tile_idx = torch.arange(n_gaussians, dtype=torch.int32, device=device)

    width = height = 8
    tile_size = 8
    grad_img = torch.randn((height, width, 4), generator=rng, device=device)

    return (
        mu,
        theta,
        sigma_x,
        sigma_y,
        color,
        opacity,
        tile_ptr,
        tile_idx,
        width,
        height,
        tile_size,
        grad_img,
    )


def fmt_tensor(name: str, tensor: torch.Tensor) -> str:
    abs_sum = float(tensor.abs().sum().detach().cpu())
    max_abs = float(tensor.abs().max().detach().cpu())
    return f"{name:8s} sum={abs_sum:10.4e} max={max_abs:10.4e}"


def print_diffs(
    title: str,
    lhs: Iterable[torch.Tensor],
    rhs: Iterable[torch.Tensor],
    names: Iterable[str],
) -> None:
    print(title)
    for name, a, b in zip(names, lhs, rhs):
        diff = (a - b).abs().max()
        print(f"{name:8s} max_diff={float(diff.detach().cpu()):10.4e}")


def print_per_splat(
    title: str,
    lhs: Iterable[torch.Tensor],
    rhs: Iterable[torch.Tensor],
    names: Iterable[str],
) -> None:
    print(title)
    for name, a, b in zip(names, lhs, rhs):
        a_flat = a.detach().cpu().reshape(-1)
        b_flat = b.detach().cpu().reshape(-1)
        if a_flat.numel() == 0:
            continue
        diff = a_flat - b_flat
        max_idx = int(torch.argmax(diff.abs()).item())
        print(
            f"{name:8s} max_diff={float(diff[max_idx]): .4e} "
            f"idx={max_idx} lhs={float(a_flat[max_idx]): .4e} "
            f"rhs={float(b_flat[max_idx]): .4e}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument(
        "--n-gaussians", type=int, default=3, help="Number of splats placed inside the tile."
    )
    parser.add_argument(
        "--dump-intermediates",
        action="store_true",
        help="Print per-splat prefix/alpha diagnostics from the kernel-formula mirror.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA device required for this harness.", file=sys.stderr)
        return 1
    if not triton.is_available():
        print("Triton backend is not available; install triton>=2.0 and enable CUDA.", file=sys.stderr)
        return 1

    device = torch.device("cuda")
    case = build_case(args.seed, device, args.n_gaussians)

    (
        mu,
        theta,
        sigma_x,
        sigma_y,
        color,
        opacity,
        tile_ptr,
        tile_idx,
        width,
        height,
        tile_size,
        grad_img,
    ) = case

    python_grads = backward_tiled_full_python(
        mu,
        theta,
        sigma_x,
        sigma_y,
        color,
        opacity,
        tile_ptr,
        tile_idx,
        width,
        height,
        tile_size,
        grad_img,
    )
    kernel_tuple = backward_tiled_full_kernel_formula(
        mu,
        theta,
        sigma_x,
        sigma_y,
        color,
        opacity,
        tile_ptr,
        tile_idx,
        width,
        height,
        tile_size,
        grad_img,
        return_intermediates=args.dump_intermediates,
    )
    kernel_grads = kernel_tuple[:7]
    kernel_info = kernel_tuple[7]

    triton_grads = triton.backward_tiled_full_triton(
        mu,
        theta,
        sigma_x,
        sigma_y,
        color,
        opacity,
        tile_ptr,
        tile_idx,
        width,
        height,
        tile_size,
        grad_img,
    )

    names = ["dcol", "dopa", "dmu_x", "dmu_y", "dtheta", "disx", "disy"]

    print("== Triton raw grads ==")
    for name, tensor in zip(names, triton_grads):
        print(fmt_tensor(name, tensor))

    print("\n== Kernel formula (recomputed) ==")
    for name, tensor in zip(names, kernel_grads):
        print(fmt_tensor(name, tensor))

    print("\n== Python reference grads ==")
    for name, tensor in zip(names, python_grads):
        print(fmt_tensor(name, tensor.to(device)))

    print_diffs("\n== Absolute max diff (Triton vs Python) ==", triton_grads, python_grads, names)
    print_diffs(
        "== Absolute max diff (Kernel formula vs Python) ==",
        kernel_grads,
        python_grads,
        names,
    )
    print_diffs(
        "== Absolute max diff (Triton vs Kernel) ==",
        triton_grads,
        kernel_grads,
        names,
    )
    print_per_splat(
        "\n== Per-splat diff (Triton - Kernel) ==",
        triton_grads,
        kernel_grads,
        names,
    )

    capture = triton.get_last_backward_capture()
    if capture:
        print("\n== Triton capture sums ==")
        cap_T = capture["sum_T"]
        cap_ai = capture["sum_ai"]
        cap_contrib = capture["sum_contrib"]
        cap_grad_ai = capture["sum_grad_ai"]
        tile_ptr_cpu = capture["tile_ptr"]
        max_splats = cap_T.shape[1]
        for tile_id in range(tile_ptr_cpu.numel() - 1):
            start = int(tile_ptr_cpu[tile_id].item())
            end = int(tile_ptr_cpu[tile_id + 1].item())
            count = min(max_splats, max(0, end - start))
            for local_idx in range(count):
                print(
                    f"[tile={tile_id} splat={local_idx}] "
                    f"sum_T={float(cap_T[tile_id, local_idx]): .4e} "
                    f"sum_ai={float(cap_ai[tile_id, local_idx]): .4e} "
                    f"sum_contrib={float(cap_contrib[tile_id, local_idx]): .4e} "
                    f"sum_grad_ai={float(cap_grad_ai[tile_id, local_idx]): .4e}"
                )

        if kernel_info:
            print("\n== Kernel formula sums ==")
            for entry in kernel_info:
                idx = entry["idx"]
                print(
                    f"[tile={entry['tile_id']} splat={idx}] "
                    f"T_recon_sum={entry['T_recon_sum']:.4e} "
                    f"alpha_sum={entry['alpha_sum']:.4e} "
                    f"T_prefix_sum={entry['T_prefix_sum']:.4e}"
                )

    if args.dump_intermediates and kernel_info:
        print("\n-- Per-splat intermediates --")
        for idx, entry in enumerate(kernel_info):
            print(
                f"[{idx}] tile={entry['tile_id']:d} idx={entry['idx']:d} "
                f"T_prefix_sum={entry['T_prefix_sum']:.4e} "
                f"T_recon_sum={entry['T_recon_sum']:.4e} "
                f"T_diff_max={entry['T_diff_max']:.4e} "
                f"alpha_sum={entry['alpha_sum']:.4e}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
