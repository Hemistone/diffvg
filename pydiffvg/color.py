"""Color/gradient parameter containers for pydiffvg shapes.

Kept simple as tensor holders, consumed by the serialization pipeline.
"""

import torch
from typing import Optional

import pydiffvg

class LinearGradient:
    """Linear gradient definition for fill/stroke colors."""

    def __init__(
        self,
        begin: torch.Tensor = torch.tensor([0.0, 0.0]),
        end: torch.Tensor = torch.tensor([0.0, 0.0]),
        offsets: torch.Tensor = torch.tensor([0.0]),
        stop_colors: torch.Tensor = torch.tensor([0.0, 0.0, 0.0, 0.0]),
    ) -> None:
        self.begin: torch.Tensor = begin
        self.end: torch.Tensor = end
        self.offsets: torch.Tensor = offsets
        self.stop_colors: torch.Tensor = stop_colors

class RadialGradient:
    """Radial gradient definition for fill/stroke colors."""

    def __init__(
        self,
        center: torch.Tensor = torch.tensor([0.0, 0.0]),
        radius: torch.Tensor = torch.tensor([0.0, 0.0]),
        offsets: torch.Tensor = torch.tensor([0.0]),
        stop_colors: torch.Tensor = torch.tensor([0.0, 0.0, 0.0, 0.0]),
    ) -> None:
        self.center: torch.Tensor = center
        self.radius: torch.Tensor = radius
        self.offsets: torch.Tensor = offsets
        self.stop_colors: torch.Tensor = stop_colors

__all__ = ["LinearGradient", "RadialGradient"]
