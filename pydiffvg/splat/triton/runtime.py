from __future__ import annotations

import torch

try:  # pragma: no cover - runtime availability probe
    import triton as _triton_mod
    import triton.language as _triton_lang
    HAS_TRITON = True
except Exception:  # pragma: no cover - gracefully degrade when Triton is missing
    _triton_mod = None
    _triton_lang = None
    HAS_TRITON = False


class _MissingTritonModule:
    """Lightweight stand-in so module import still succeeds when Triton is absent."""

    @staticmethod
    def jit(*args, **kwargs):
        def _decorator(fn):
            return fn

        return _decorator

    @staticmethod
    def cdiv(x: int, y: int) -> int:
        return (x + y - 1) // y


class _MissingTritonLanguage:
    def __getattr__(self, name: str):  # pragma: no cover - raises on actual use
        raise RuntimeError("Triton language module unavailable; install triton to enable kernels")


triton = _triton_mod if _triton_mod is not None else _MissingTritonModule()
tl = _triton_lang if _triton_lang is not None else _MissingTritonLanguage()


def is_available() -> bool:
    return HAS_TRITON and torch.cuda.is_available()


__all__ = ["HAS_TRITON", "is_available", "tl", "triton"]
