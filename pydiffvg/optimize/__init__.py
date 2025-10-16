"""Optimization utilities for diffvg's SVG workflows."""

from .settings import SvgOptimizationSettings
from .transforms import TransformTools
from .scene_graph import (
    SvgNode,
    GroupNode,
    RootNode,
    ShapeNode,
    PathNode,
    RectNode,
    CircleNode,
    EllipseNode,
    PolygonNode,
    GradientNode,
    configure_scene_graph,
)
from .parser import SvgParserMixin
from .writer import SvgWriterMixin
from .driver import SvgOptimizationDriver
from .core import OptimizableSvg

__all__ = [
    "SvgOptimizationSettings",
    "TransformTools",
    "SvgParserMixin",
    "SvgWriterMixin",
    "SvgOptimizationDriver",
    "SvgNode",
    "GroupNode",
    "RootNode",
    "ShapeNode",
    "PathNode",
    "RectNode",
    "CircleNode",
    "EllipseNode",
    "PolygonNode",
    "GradientNode",
    "configure_scene_graph",
    "OptimizableSvg",
]
