"""Renderer wrapper for the maintained stroke-first backend."""

from __future__ import annotations

from typing import Any, Optional

from .backend import get_backend, get_backend_config
from .backends.registry import get_api
from .device import get_device


class Renderer:
    def __init__(self, backend: Optional[str] = None) -> None:
        self.backend = (backend or get_backend()).lower()
        self._cache: dict[str, Any] = {}
        self.config = get_backend_config(self.backend)
        self._api = get_api(self.backend)

    def serialize_scene(self, *args, **kwargs):
        cache_key = kwargs.pop("cache_key", None)
        invalidate_cache = bool(kwargs.pop("invalidate_cache", False))
        if cache_key is not None and invalidate_cache:
            self._cache.pop(cache_key, None)

        api = self._api
        if cache_key is not None and cache_key in self._cache:
            return self._cache[cache_key]

        if "keep_on_device" not in kwargs and getattr(api, "prefer_device_serialization", False):
            kwargs["keep_on_device"] = True
        if kwargs.get("keep_on_device") and "device" not in kwargs:
            kwargs["device"] = get_device()
        scene_args = api.serialize_scene(*args, **kwargs)
        if cache_key is not None:
            self._cache[cache_key] = scene_args
        return scene_args

    def apply(self, *args, **kwargs):
        return self._api.apply(*args, **kwargs)

    def render_grad(self, *args, **kwargs):
        return self._api.render_grad(*args, **kwargs)


__all__ = ["Renderer"]
