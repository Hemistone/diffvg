from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


class OpenStrokeUnsupported(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class ShapeRef:
    shape: object
    start: int
    end: int
    index: int


@dataclass(frozen=True)
class StyleRef:
    group: object
    index: int


@dataclass(frozen=True)
class CompiledOpenStrokeScene:
    canvas_width: int
    canvas_height: int
    output_type: object
    use_prefiltering: bool
    eval_positions: torch.Tensor
    point_bank: torch.Tensor
    stroke_width_bank: torch.Tensor
    stroke_rgba_bank: torch.Tensor
    shape_refs: Sequence[ShapeRef]
    style_refs: Sequence[StyleRef]
    control_source_indices: torch.Tensor
    control_source_weights: torch.Tensor
    segment_mask: torch.Tensor
    style_index: torch.Tensor
    chunk_order: torch.Tensor
    max_segments: int = 3

    @property
    def stroke_count(self) -> int:
        return int(self.control_source_indices.shape[0])

    def bind_frontend_views(self) -> None:
        for ref in self.shape_refs:
            ref.shape.points = self.point_bank[ref.start:ref.end]
            ref.shape.stroke_width = self.stroke_width_bank[ref.index]
        for ref in self.style_refs:
            ref.group.stroke_color = self.stroke_rgba_bank[ref.index]

    def point_parameters(self) -> list[torch.Tensor]:
        return [self.point_bank]

    def width_parameters(self) -> list[torch.Tensor]:
        return [self.stroke_width_bank]

    def color_parameters(self) -> list[torch.Tensor]:
        return [self.stroke_rgba_bank]
