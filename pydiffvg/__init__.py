"""Public Python API surface for pydiffvg.

This module aggregates the stable entry points that downstream callers rely on
while keeping implementation modules small and focused.
"""

from . import device as _device
from . import shape as _shape
from . import pixel_filter as _pixel_filter
from . import color as _color
from . import optimize_svg as _optimize_svg
from . import optimize as _optimize
from . import precondition as _precondition

from .device import *  # noqa: F401,F403 - re-exported for legacy callers
from .shape import *  # noqa: F401,F403
from .pixel_filter import *  # noqa: F401,F403
from .color import *  # noqa: F401,F403
from .optimize_svg import *  # noqa: F401,F403
from .optimize import SvgOptimizationDriver, SvgParserMixin, SvgWriterMixin
from .precondition import *  # noqa: F401,F403

from .render_pytorch import RenderFunction, OutputType, set_print_timing
from .image import imwrite
from .parse_svg import svg_to_scene, parse_scene, parse_transform, parse_color
from .save_svg import save_svg
from .serialization import serialize_scene

# Backend selection API (minimal)
from .backend import set_backend, get_backend, list_backends, get_backend_config, SplatConfig, DepthPolicy
from .renderer import Renderer


__all__ = (
    list(_device.__all__)
    + list(_shape.__all__)
    + list(_pixel_filter.__all__)
    + list(_color.__all__)
    + list(_optimize_svg.__all__)
    + list(_optimize.__all__)
    + list(_precondition.__all__)
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
        "serialize_scene",
        "set_backend",
        "get_backend",
        "list_backends",
        "get_backend_config",
        "SplatConfig",
        "DepthPolicy",
        "Renderer",
    ]
)
