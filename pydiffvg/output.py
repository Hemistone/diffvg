from __future__ import annotations

from enum import IntEnum


class OutputType(IntEnum):
    color = 1
    sdf = 2


__all__ = ["OutputType"]
