from .device import *
from .shape import *
from .pixel_filter import *
from .color import *
from .optimize_svg import *

from .render_pytorch import RenderFunction, OutputType, set_print_timing
from .image import imwrite
from .parse_svg import svg_to_scene, parse_scene, parse_transform, parse_color
from .save_svg import save_svg
from .serialization import serialize_scene

# Backend selection API (minimal)
from .backend import set_backend, get_backend, list_backends, get_backend_config, SplatConfig, DepthPolicy
from .renderer import Renderer
