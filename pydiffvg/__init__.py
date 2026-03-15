"""Public Python API surface for pydiffvg.

This module aggregates the stable entry points that downstream callers rely on
while keeping implementation modules small and focused.
"""

from . import device as _device
from . import shape as _shape
from . import precondition as _precondition
from . import plotter as _plotter

from .device import *  # noqa: F401,F403 - re-exported for legacy callers
from .shape import *  # noqa: F401,F403
from .precondition import *  # noqa: F401,F403
from .plotter import *  # noqa: F401,F403

from .render_function import RenderFunction, OutputType, set_print_timing
from .image import imwrite
from .parse_svg import svg_to_scene, parse_scene, parse_transform, parse_color
from .save_svg import save_svg
from .svg_preview import render_svg_preview
from .serialization import serialize_scene

# Backend selection API (minimal)
from .backend import (
    set_backend,
    get_backend,
    list_backends,
    get_backend_config,
    BezierGsplatConfig,
)
from .renderer import Renderer


__all__ = (
    list(_device.__all__)
    + list(_shape.__all__)
    + list(_precondition.__all__)
    + list(_plotter.__all__)
    + [
        "RenderFunction",
        "OutputType",
        "set_print_timing",
        "imwrite",
        "svg_to_scene",
        "parse_scene",
        "parse_transform",
        "parse_color",
        "save_svg",
        "render_svg_preview",
        "serialize_scene",
        "set_backend",
        "get_backend",
        "list_backends",
        "get_backend_config",
        "BezierGsplatConfig",
        "Renderer",
    ]
)
