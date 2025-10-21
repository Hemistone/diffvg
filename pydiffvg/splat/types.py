from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import diffvg
import torch

from ..backend import SplatConfig


@dataclass(frozen=True)
class PaintPayload:
    color_type: Optional[diffvg.ColorType]
    params: Tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class PathPayload:
    shape_id: int
    num_control_points: torch.Tensor
    points: torch.Tensor
    thickness: Optional[torch.Tensor]
    is_closed: bool
    use_distance_approx: bool
    stroke_width: torch.Tensor


@dataclass(frozen=True)
class NonPathShapePayload:
    shape_id: int
    shape_type: diffvg.ShapeType
    tensors: Tuple[torch.Tensor, ...]
    stroke_width: torch.Tensor


@dataclass(frozen=True)
class ShapeGroupPayload:
    shape_ids: torch.Tensor
    fill: PaintPayload
    stroke: PaintPayload
    use_even_odd_rule: bool
    shape_to_canvas: torch.Tensor


@dataclass(frozen=True)
class ScenePayload:
    canvas_width: int
    canvas_height: int
    output_type: Optional[int]
    use_prefiltering: bool
    eval_positions: torch.Tensor
    paths: List[PathPayload]
    non_path_shapes: List[NonPathShapePayload]
    shape_groups: List[ShapeGroupPayload]
    filter_type: Optional[diffvg.FilterType]
    filter_radius: torch.Tensor


@dataclass(frozen=True)
class RenderRequest:
    width: int
    height: int
    num_samples_x: int
    num_samples_y: int
    seed: int
    background_image: Optional[torch.Tensor]
    config: SplatConfig
    scene: ScenePayload
    generator: Optional[torch.Generator]


@dataclass(frozen=True)
class SegmentData:
    start: torch.Tensor
    controls: Tuple[torch.Tensor, ...]
    end: torch.Tensor


@dataclass(frozen=True)
class PathSpec:
    shape_id: int
    segments: List[SegmentData]
    stroke_width: torch.Tensor
    color_rgb: torch.Tensor
    opacity: torch.Tensor


@dataclass(frozen=True)
class FillSpec:
    shape_id: int
    segments: List[SegmentData]
    color_rgb: torch.Tensor
    opacity: torch.Tensor


@dataclass(frozen=True)
class GaussianBatch:
    mu: torch.Tensor
    theta: torch.Tensor
    sigma_x: torch.Tensor
    sigma_y: torch.Tensor
    color_rgb: torch.Tensor
    opacity: torch.Tensor


@dataclass(frozen=True)
class GradSlot:
    arg_index: int
    tensor: torch.Tensor


class _SplatUnsupported(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


__all__ = [
    "PaintPayload",
    "PathPayload",
    "NonPathShapePayload",
    "ShapeGroupPayload",
    "ScenePayload",
    "RenderRequest",
    "SegmentData",
    "PathSpec",
    "FillSpec",
    "GaussianBatch",
    "GradSlot",
    "_SplatUnsupported",
]
