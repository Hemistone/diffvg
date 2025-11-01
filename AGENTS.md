# Repository Guidelines

This repository is a modernized fork of https://github.com/BachiLi/diffvg aimed at Python 3.12+, CUDA 13.x, latest pybind11, and using CCCL/Thrust inherited from the CUDA Toolkit. Modernization is in progress; some rough edges remain.

## Project Structure & Module Organization
- `pydiffvg/`: Python API (PyTorch bindings, SVG parsing/rendering utilities).
- C++ sources in repo root: core rasterizer and bindings (`*.cpp`, `*.h`).
- `apps/`: runnable examples and small sanity checks (e.g., `single_circle.py`).
- `cmake/`: CMake helpers.
- Generated build outputs: `build/`, wheels in `dist/`.

## Build, Test, and Development Commands
- Please use .venv python(.venv/bin/activate) for using installed packages, along with pip.
- Install (CUDA default): `pip install .`
- CPU-only build: `DIFFVG_CUDA=0 pip install .`
- Manual CMake (GPU): `mkdir -p build && cd build && cmake -G Ninja -DDIFFVG_CUDA=1 .. && ninja -j`
- Examples: `cd apps && python single_circle.py` (see `README.md` for more).
- One-liners: `make run-direct` (tests `diffvg` only), `make run-samples`, `make dev-check` (clean caches + both).

## Coding Style & Naming Conventions
- C++: C++14, 2–4 space indent, headers in root, filenames `lower_snake_case`.
  - Prefer `std::` containers, early returns, minimal headers in `*.h`.
- Python: PEP 8, 4 spaces, module and script names `lower_snake_case`.
  - Docstrings for public functions; type hints where practical.
- Keep bindings minimal and stable; place Python-facing logic in `pydiffvg/`.

## Testing Guidelines
- Smoke tests live in `apps/` (e.g., `python apps/svg_parse_test.py`, `python apps/test_eval_positions.py`).
- Run a couple of examples after building to validate both CPU/CUDA paths.
- If adding features, prefer small, runnable examples under `apps/` mirroring existing patterns.
- For vectorizer-related work, consult `docs/vectorizer.md` for the current roadmap, module ownership, and CPU verification
  commands before starting a task.

## Commit & Pull Request Guidelines
- Commits: imperative, scoped messages (e.g., `build: set default CUDA arch to 89`, `fix(pydiffvg): handle empty path list`).
- CI/local checks: ensure `pip install .` succeeds and key `apps/` scripts run.

## Security & Configuration Tips
- pybind11: install from pip/conda/system (e.g., `pip install pybind11>=3.0.1`) before configuring CMake.
- CUDA builds require a CUDA 13.x toolkit; for CPU-only, set `DIFFVG_CUDA=0` and ensure Thrust headers are available or from Toolkit.

## Known Issue: Segfault and .pyc Corruption
- Symptom: running diffvg segfaults; subsequent runs raise `bad marshal data`/`marshal data too short` due to corrupted `__pycache__/*.pyc` from the crash.
- Immediate mitigations:
  - Disable bytecode in dev: `PYTHONDONTWRITEBYTECODE=1` (enabled automatically via `sitecustomize.py` in repo).
  - Clean caches: `find . -name __pycache__ -exec rm -rf {} +` before re-run.
  - Force CPU build to isolate CUDA path: `DIFFVG_CUDA=0 pip install .` then rerun examples.
