"""Small wrapper class for pixel filters.

This remains only as a compatibility container on the stroke-first Python path.
"""

import torch
from typing import Any

import pydiffvg

class PixelFilter:
    """Pixel filter description.

    Parameters
    - type: int-compatible filter identifier
    - radius: filter radius as a scalar tensor
    """

    def __init__(self, type, radius: torch.Tensor = torch.tensor(0.5)) -> None:
        self.type = type
        self.radius: torch.Tensor = radius

__all__ = ["PixelFilter"]
