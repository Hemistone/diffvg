from __future__ import annotations

import os


def _env_flag(name: str, default: bool = False) -> bool:
    val = os.environ.get(name, "")
    if not val:
        return default
    v = val.strip().lower()
    return v in ("1", "true", "yes", "on")


__all__ = ["_env_flag"]
