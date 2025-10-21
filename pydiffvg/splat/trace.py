from __future__ import annotations

import atexit
import os
import sys
import warnings

_TRACE_FORWARD = 0
_TRACE_BACKWARD = 0
_TRACE_FALLBACK = 0
_TRACE_LIMIT: int | None = None


def _trace_settings() -> tuple[bool, int | None]:
    raw = os.environ.get("DIFFVG_SPLAT_TRACE", "").strip()
    if not raw:
        return False, None
    lowered = raw.lower()
    if lowered in {"0", "false", "no", "off"}:
        return False, None
    if "limit=" in lowered:
        try:
            limit = int(lowered.split("limit=", 1)[1].strip())
        except ValueError:
            limit = 5
        return True, limit
    try:
        limit = int(raw)
        return True, limit
    except ValueError:
        pass
    if lowered in {"1", "true", "yes", "on"} or lowered:
        return True, 5
    return False, None


def debug_enabled() -> bool:
    enabled, _ = _trace_settings()
    return enabled


def trace(message: str) -> None:
    if not debug_enabled():
        return
    sys.stdout.write(f"\n[splat-trace] {message}\n")
    sys.stdout.flush()


def _trace_limit() -> int:
    global _TRACE_LIMIT
    if _TRACE_LIMIT is None:
        enabled, limit = _trace_settings()
        if not enabled:
            _TRACE_LIMIT = 0
        else:
            _TRACE_LIMIT = 5 if limit is None else int(limit)
    return _TRACE_LIMIT


def should_print(count: int) -> bool:
    limit = _trace_limit()
    return limit <= 0 or count <= limit


def trace_summary() -> None:
    if not debug_enabled():
        return
    print(
        f"[splat-trace] summary: forward={_TRACE_FORWARD} backward={_TRACE_BACKWARD} fallback={_TRACE_FALLBACK}",
        file=sys.stderr,
        flush=True,
    )


def increment_forward() -> int:
    global _TRACE_FORWARD
    _TRACE_FORWARD += 1
    return _TRACE_FORWARD


def increment_backward() -> int:
    global _TRACE_BACKWARD
    _TRACE_BACKWARD += 1
    return _TRACE_BACKWARD


def increment_fallback() -> int:
    global _TRACE_FALLBACK
    _TRACE_FALLBACK += 1
    return _TRACE_FALLBACK


def warn_fallback(reason: str) -> None:
    key = reason or "<unspecified>"
    seen = getattr(warn_fallback, "_seen", set())
    if key in seen:
        return
    increment_fallback()
    trace(f"fallback to baseline (reason: {reason})")
    message = (
        "Bézier Splatting backend falling back to baseline renderer"
        f" (reason: {reason}). See docs/bezier_splatting_todo.md for progress."
    )
    warnings.warn(message, RuntimeWarning, stacklevel=3)
    seen.add(key)
    setattr(warn_fallback, "_seen", seen)


atexit.register(trace_summary)

__all__ = [
    "debug_enabled",
    "trace",
    "should_print",
    "trace_summary",
    "increment_forward",
    "increment_backward",
    "increment_fallback",
    "warn_fallback",
]
