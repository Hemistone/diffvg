"""Testing utilities for loading :mod:`pydiffvg.vectorizer` modules."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VECTOR_DIR = ROOT / "pydiffvg" / "vectorizer"
PACKAGE_NAME = "pydiffvg.vectorizer"


def load_vectorizer_module(module_name: str):
    """Load a vectorizer submodule without importing the heavy :mod:`pydiffvg` package."""

    full_name = f"{PACKAGE_NAME}.{module_name}"
    module_path = VECTOR_DIR / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(full_name, module_path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"Unable to load module {full_name!r} for testing")

    if "pydiffvg" not in sys.modules:
        pkg = types.ModuleType("pydiffvg")
        pkg.__path__ = [str(ROOT / "pydiffvg")]
        sys.modules["pydiffvg"] = pkg

    if PACKAGE_NAME not in sys.modules:
        subpkg = types.ModuleType(PACKAGE_NAME)
        subpkg.__path__ = [str(VECTOR_DIR)]
        sys.modules[PACKAGE_NAME] = subpkg

    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module
