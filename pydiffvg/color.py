"""Color/gradient parameter containers for pydiffvg shapes.

Kept simple as tensor holders, consumed by the serialization pipeline.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch


class Paint:
    """Simple RGBA color wrapper used by vectorizer adapters.

    ``diffvg`` historically accepted bare ``torch.Tensor`` instances for constant
    colors.  The vectorizer pipeline benefits from a lightweight value object so
    the adapters can reason about colors without worrying about tensor
    contiguity or dtype conversions.  ``Paint`` keeps the tensor contiguous and
    provides a ``to_tensor`` helper that mirrors what serialization expects.
    """

    def __init__(self, rgba: Sequence[float] | torch.Tensor) -> None:
        if isinstance(rgba, torch.Tensor):
            tensor = rgba.detach().clone().to(dtype=torch.float32)
        else:
            tensor = torch.tensor(list(rgba), dtype=torch.float32)
        if tensor.ndim != 1:
            raise ValueError("Paint expects a 1D tensor of length 3 or 4")
        if tensor.numel() == 3:
            tensor = torch.cat([tensor, torch.tensor([1.0], dtype=torch.float32)])
        if tensor.numel() != 4:
            raise ValueError("Paint expects RGBA values (length 4)")
        self._rgba = tensor.contiguous()

    def to_tensor(self) -> torch.Tensor:
        """Return the underlying contiguous tensor."""

        return self._rgba

    def __iter__(self) -> Iterable[float]:
        return iter(self._rgba.tolist())

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        rgba = ", ".join(f"{c:.3f}" for c in self._rgba.tolist())
        return f"Paint([{rgba}])"


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

__all__ = ["Paint", "LinearGradient", "RadialGradient"]
