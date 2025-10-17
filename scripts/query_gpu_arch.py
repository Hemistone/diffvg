#!/usr/bin/env python3
"""
Print GPU compute capability and recommended CMake/Torch arch strings.

Usage:
  python scripts/query_gpu_arch.py

Outputs lines like:
  GPU 0: NVIDIA GeForce RTX 4090 | CC 8.9 -> sm_89 | CMAKE=89 | TORCH=8.9
"""
from __future__ import annotations

import os
import subprocess
import sys
from typing import List, Optional, Tuple


def _from_torch() -> List[Tuple[str, int, int]]:
    try:
        import torch  # type: ignore
    except Exception:
        return []
    if not torch.cuda.is_available():
        return []
    gpus = []
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        major, minor = torch.cuda.get_device_capability(i)
        gpus.append((name, major, minor))
    return gpus


def _from_nvidia_smi() -> List[Tuple[str, int, int]]:
    try:
        # Prefer direct CSV query with compute_cap if supported
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,name,compute_cap",
            "--format=csv,noheader",
        ]
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
        gpus = []
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 3:
                continue
            _, name, cc = parts
            if "." in cc:
                maj_s, min_s = cc.split(".", 1)
            else:
                # sometimes reported as 89 already
                maj_s, min_s = cc[:-1], cc[-1:]
            try:
                major, minor = int(maj_s), int(min_s)
            except ValueError:
                continue
            gpus.append((name, major, minor))
        if gpus:
            return gpus
    except Exception:
        pass

    try:
        # Fallback: parse human-readable output
        out = subprocess.check_output(["nvidia-smi", "-q"], text=True)
    except Exception:
        return []
    gpus = []
    name, major, minor = None, None, None
    for line in out.splitlines():
        if "Product Name" in line:
            name = line.split(":", 1)[1].strip()
        elif "Compute Capability" in line:
            cc = line.split(":", 1)[1].strip()
            if "." in cc:
                maj_s, min_s = cc.split(".", 1)
                try:
                    major, minor = int(maj_s), int(min_s)
                except ValueError:
                    major, minor = None, None
        if name and major is not None and minor is not None:
            gpus.append((name, major, minor))
            name, major, minor = None, None, None
    return gpus


def main() -> int:
    gpus = _from_torch()
    if not gpus:
        gpus = _from_nvidia_smi()

    if not gpus:
        print("No GPUs detected via torch or nvidia-smi.")
        return 1

    for idx, (name, major, minor) in enumerate(gpus):
        sm = major * 10 + minor
        torch_arch = f"{major}.{minor}"
        print(
            f"GPU {idx}: {name} | CC {major}.{minor} -> sm_{sm} | CMAKE={sm} | TORCH={torch_arch}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

