from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


class OpenStrokeUnsupported(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class StrokeStyleRef:
    shape: object
    group: object


@dataclass(frozen=True)
class CompiledOpenStrokeScene:
    canvas_width: int
    canvas_height: int
    output_type: object
    use_prefiltering: bool
    eval_positions: torch.Tensor
    point_refs: Sequence[torch.Tensor]
    style_refs: Sequence[StrokeStyleRef]
    control_source_indices: torch.Tensor
    control_source_weights: torch.Tensor
    segment_mask: torch.Tensor
    style_index: torch.Tensor
    chunk_order: torch.Tensor
    max_segments: int = 3

    @property
    def stroke_count(self) -> int:
        return int(self.control_source_indices.shape[0])
