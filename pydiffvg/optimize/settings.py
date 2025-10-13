"""Configuration helpers for diffvg SVG optimization routines."""

from __future__ import annotations

import copy
import json
from typing import Any, Dict, MutableMapping, Sequence, TextIO, Tuple

import torch


class SvgOptimizationSettings:
    """Mutable configuration store for SVG optimization runs."""

    default_params: Dict[str, Any] = {
        "optimize_color": True,
        "color_lr": 2e-3,
        "optimize_alpha": False,
        "alpha_lr": 2e-3,
        "optimizer": "Adam",
        "transforms": {
            "optimize_transforms": True,
            "transform_mode": "rigid",
            "translation_mult": 1e-3,
            "transform_lr": 2e-3,
        },
        "circles": {
            "optimize_center": True,
            "optimize_radius": True,
            "shape_lr": 2e-1,
        },
        "paths": {
            "optimize_points": True,
            "shape_lr": 2e-1,
        },
        "gradients": {
            "optimize_stops": True,
            "stop_lr": 2e-3,
            "optimize_color": True,
            "color_lr": 2e-3,
            "optimize_alpha": False,
            "alpha_lr": 2e-3,
            "optimize_location": True,
            "location_lr": 2e-1,
        },
    }

    optims: Dict[str, type[torch.optim.Optimizer]] = {
        "Adam": torch.optim.Adam,
        "SGD": torch.optim.SGD,
        "ASGD": torch.optim.ASGD,
    }

    def __init__(self, f: TextIO | None = None) -> None:
        self.store: Dict[str, Dict[str, Any]] = {}
        if f is None:
            self.store["default"] = copy.deepcopy(SvgOptimizationSettings.default_params)
        else:
            self.store = json.load(f)

    def default_name(self, dname: str) -> None:
        """Create an alias for the root settings entry."""
        self.dname = dname
        if dname not in self.store:
            self.store[dname] = self.store["default"]

    def retrieve(self, node_id: str) -> Tuple[Dict[str, Any], bool]:
        """Return settings for ``node_id`` and whether they are customized."""
        if node_id not in self.store:
            return (self.store["default"], False)
        return (self.store[node_id], True)

    def reset_to_defaults(self, node_id: str) -> Dict[str, Any]:
        """Remove overrides and return the default configuration."""
        if node_id in self.store:
            del self.store[node_id]
        return self.store["default"]

    def undefault(self, node_id: str) -> Dict[str, Any]:
        """Ensure ``node_id`` has its own mutable copy of the defaults."""
        if node_id not in self.store:
            self.store[node_id] = copy.deepcopy(self.store["default"])
        return self.store[node_id]

    def override_optimizer(self, optimizer: str | None) -> None:
        """Force all entries to use the provided optimizer name."""
        if optimizer is not None:
            for entry in self.store.values():
                entry["optimizer"] = optimizer

    def global_override(self, path: Sequence[str], value: Any) -> None:
        """Override a nested key across all stored settings."""
        for entry in self.store.values():
            target: MutableMapping[str, Any] = entry
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value

    def save(self, file: TextIO) -> None:
        """Serialize settings to a JSON file-like object."""
        self.store["default"] = self.store[self.dname]
        json.dump(self.store, file, indent="\t")
