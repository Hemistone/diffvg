"""Compatibility wrapper for the old render_pytorch module.

The maintained runtime is stroke-first and routes through the pure-Python
RenderFunction dispatcher. Legacy exact rendering is no longer available.
"""

from __future__ import annotations

from .output import OutputType
from .render_function import RenderFunction, print_timing, set_print_timing


class BaselineRenderFunction:
    @staticmethod
    def serialize_scene(*args, **kwargs):
        raise RuntimeError("The legacy 'baseline' renderer has been removed. Use 'bezier_gsplat'.")

    @staticmethod
    def apply(*args, **kwargs):
        raise RuntimeError("The legacy 'baseline' renderer has been removed. Use 'bezier_gsplat'.")

    @staticmethod
    def render_grad(*args, **kwargs):
        raise RuntimeError("The legacy 'baseline' renderer has been removed. Use 'bezier_gsplat'.")


__all__ = ["RenderFunction", "BaselineRenderFunction", "OutputType", "set_print_timing", "print_timing"]
