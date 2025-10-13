"""Legacy shim for backward compatibility.

This module is retained to avoid breaking older imports. New code should
prefer the APIs under :mod:`pydiffvg.optimize`.
"""

from __future__ import annotations

import warnings

from .optimize.settings import SvgOptimizationSettings
from .optimize.core import OptimizableSvg
from .optimize.driver import SvgOptimizationDriver

warnings.warn(
    "pydiffvg.optimize_svg is deprecated; use pydiffvg.optimize instead",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "SvgOptimizationSettings",
    "OptimizableSvg",
    "SvgOptimizationDriver",
]
