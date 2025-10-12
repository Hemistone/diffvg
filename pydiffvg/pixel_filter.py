"""Small wrapper class for pixel filters used by diffvg.

Instances are passed through the serialization path to the C++ extension.
"""

import torch
from typing import Any

import pydiffvg

class PixelFilter:
    """Pixel filter description.

    Parameters
    - type: Filter enum value from `diffvg.FilterType` (int-compatible)
    - radius: filter radius as a scalar tensor
    """

    def __init__(self, type, radius: torch.Tensor = torch.tensor(0.5)) -> None:
        # Keep the enum instance (diffvg.FilterType) to match the C++ binding expectations
        self.type = type
        self.radius: torch.Tensor = radius

__all__ = ["PixelFilter"]
