"""Developer toggles for pydiffvg.

Centralizes runtime debug flags so examples and library code don’t keep
their own globals. Keep this intentionally tiny and stable.
"""

from __future__ import annotations

from typing import Final

_PRINT_TIMING: bool = False

def set_print_timing(v: bool) -> None:
    global _PRINT_TIMING
    _PRINT_TIMING = bool(v)

def get_print_timing() -> bool:
    return _PRINT_TIMING

__all__: Final = [
    "set_print_timing",
    "get_print_timing",
]

