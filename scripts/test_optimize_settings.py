#!/usr/bin/env python3
"""Smoke checks for SvgOptimizationSettings serialization behavior."""

from __future__ import annotations

import io

import pydiffvg


def main() -> None:
    settings = pydiffvg.SvgOptimizationSettings()
    settings.default_name("root")

    custom = settings.undefault("circle-1")
    custom["optimizer"] = "SGD"
    settings.global_override(["color_lr"], 1e-2)
    settings.override_optimizer("ASGD")

    retrieved, customized = settings.retrieve("circle-1")
    assert customized is True
    assert retrieved["optimizer"] == "ASGD"
    assert abs(retrieved["color_lr"] - 1e-2) < 1e-9

    buffer = io.StringIO()
    settings.save(buffer)
    buffer.seek(0)

    restored = pydiffvg.SvgOptimizationSettings(buffer)
    restored.default_name("root")

    restored_circle, customized = restored.retrieve("circle-1")
    assert customized is True
    assert restored_circle["optimizer"] == "ASGD"
    assert abs(restored_circle["color_lr"] - 1e-2) < 1e-9

    default_after_reset = restored.reset_to_defaults("circle-1")
    assert customized is True
    assert default_after_reset is restored.store["default"]

    print("ok: SvgOptimizationSettings serialization round-trip succeeded")


if __name__ == "__main__":
    main()
