"""Renderer stub class to route rendering via the selected backend.

This class is intentionally thin: it defers to the backend's RenderFunction
and can later host lightweight caches once the splatting backend is added.
"""

from __future__ import annotations

from typing import Any, Optional

from .backend import get_backend, current_api, get_backend_config


class Renderer:
    def __init__(self, backend: Optional[str] = None) -> None:
        self.backend = (backend or get_backend()).lower()
        # Placeholder for future cache structures
        self._cache: dict[str, Any] = {}
        # Snapshot backend-specific config (None for baseline)
        self.config = get_backend_config(self.backend)

    def serialize_scene(self, *args, **kwargs):
        api = current_api()
        return api.serialize_scene(*args, **kwargs)

    def apply(self, *args, **kwargs):
        api = current_api()
        return api.apply(*args, **kwargs)

    def render_grad(self, *args, **kwargs):
        api = current_api()
        return api.render_grad(*args, **kwargs)


__all__ = ["Renderer"]

