from .device import *
from .shape import *
from .pixel_filter import *
from .render_pytorch import *
from .image import *
from .parse_svg import *
from .color import *
from .optimize_svg import *
from .save_svg import *

# Backend selection API (minimal)
from .backend import set_backend, get_backend, list_backends, get_backend_config, SplatConfig, DepthPolicy
from .renderer import Renderer
