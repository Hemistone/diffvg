from __future__ import annotations

from functools import lru_cache
from importlib.util import find_spec
from typing import Callable, NamedTuple


class GsplatOps(NamedTuple):
    project_gaussians_2d_scale_rot: Callable[..., object]
    rasterize_gaussians: Callable[..., object]


def gsplat_available() -> bool:
    try:
        return find_spec("gsplat") is not None
    except Exception:
        return False


def _missing_dependency_error(exc: Exception) -> RuntimeError:
    err = RuntimeError(
        "The 'bezier_gsplat' backend requires the optional 'gsplat' package. "
        "Install a compatible build in this repo's virtualenv, then retry. "
        "The reference setup is the one used by ../Bezier_splatting."
    )
    err.__cause__ = exc
    return err


@lru_cache(maxsize=1)
def load_gsplat_ops() -> GsplatOps:
    try:
        from gsplat import (
            project_gaussians_2d_scale_rot,
            rasterize_gaussians,
        )
    except Exception as exc:  # pragma: no cover - optional dependency
        raise _missing_dependency_error(exc)
    return GsplatOps(
        project_gaussians_2d_scale_rot=project_gaussians_2d_scale_rot,
        rasterize_gaussians=rasterize_gaussians,
    )
