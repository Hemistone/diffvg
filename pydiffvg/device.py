"""Device selection and CUDA/CPU toggle helpers for pydiffvg.

This module centralizes device handling so examples and library code can rely on
`get_device()` and `set_use_gpu()` without duplicating torch/cuda checks.
"""

import os
import warnings
from typing import Optional

import torch
import diffvg


def _torch_gpu_available() -> bool:
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


def _cuda_compiled() -> bool:
    checker = getattr(diffvg, "is_cuda_compiled", None)
    if checker is None:
        return False
    return bool(checker())


def _default_use_gpu() -> bool:
    if os.environ.get("DIFFVG_FORCE_CPU", "").strip() not in ("", "0", "false", "False"):
        return False
    if os.environ.get("DIFFVG_FORCE_GPU", "").strip() not in ("", "0", "false", "False"):
        return _torch_gpu_available() and _cuda_compiled()
    return _torch_gpu_available() and _cuda_compiled()


use_gpu: bool = _default_use_gpu()
device: torch.device = torch.device('cuda') if use_gpu else torch.device('cpu')

def set_use_gpu(v: bool) -> None:
    global use_gpu
    global device
    requested = bool(v)
    if requested and not _cuda_compiled():
        warnings.warn("diffvg was built without CUDA support; falling back to CPU.", RuntimeWarning)
        requested = False
    if requested and not _torch_gpu_available():
        warnings.warn("PyTorch reports no CUDA device; falling back to CPU.", RuntimeWarning)
        requested = False
    use_gpu = requested
    device = torch.device('cuda') if use_gpu else torch.device('cpu')

def get_use_gpu() -> bool:
    global use_gpu
    return use_gpu

def set_device(d: torch.device | str) -> None:
    global device
    global use_gpu
    target = torch.device(d)
    if target.type == 'cuda':
        if not _cuda_compiled():
            warnings.warn("diffvg was built without CUDA support; keeping CPU device.", RuntimeWarning)
            device = torch.device('cpu')
            use_gpu = False
            return
        if not _torch_gpu_available():
            warnings.warn("PyTorch reports no CUDA device; keeping CPU device.", RuntimeWarning)
            device = torch.device('cpu')
            use_gpu = False
            return
        device = target
        use_gpu = True
    else:
        device = target
        use_gpu = False

def get_device() -> torch.device:
    global device
    return device

__all__ = [
    "set_use_gpu",
    "get_use_gpu",
    "set_device",
    "get_device",
    "use_gpu",
    "device",
]
