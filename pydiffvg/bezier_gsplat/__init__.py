"""Utilities for the optional gsplat-backed Bézier renderer."""

from .runtime import gsplat_available, load_gsplat_ops

__all__ = ["gsplat_available", "load_gsplat_ops"]
