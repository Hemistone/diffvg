# Repository Guidelines

This repository is a stroke-first raster-to-vector engine built around the
`bezier_gsplat` runtime. The old native diffvg renderer has been removed from
the maintained path.

## Project Structure & Module Organization
- `pydiffvg/`: Python API, stroke-first runtime, SVG/render/export helpers.
- `apps/`: runnable workflows, smokes, and small benchmarks.
- `docs/`: design notes and workflow context.
- Generated build outputs: wheels in `dist/`.

## Build, Test, and Development Commands
- Please use .venv python(.venv/bin/activate) for using installed packages, along with pip.
- Install package: `pip install -e .`
- Install runtime deps: `pip install -r requirements.txt`
- Minimal checks: `make smoke-runtime`, `make smoke-renderer`, `make smoke-painterly`
- One-liner: `make dev-check`

## Coding Style & Naming Conventions
- Python: PEP 8, 4 spaces, module and script names `lower_snake_case`.
  - Docstrings for public functions; type hints where practical.
- Keep runtime logic in `pydiffvg/`; avoid reintroducing native bindings or
  general-purpose fill/exact renderer paths.

## Testing Guidelines
- Keep validation small and high-signal:
  - `python apps/test_stroke_first_runtime.py`
  - `python apps/bench_renderer_micro.py --backends bezier_gsplat --path-counts 32 --repeats 1 --warmup 1`
  - short `apps/painterly_rendering.py` smoke
- If adding features, prefer small, runnable examples under `apps/`.

## Commit & Pull Request Guidelines
- Commits: imperative, scoped messages (e.g., `refactor(runtime): remove generic svg parser dependency`, `fix(output): reject non-finite stroke points`).
- CI/local checks: ensure `pip install -e .` succeeds and the stroke-first smokes run.

## Security & Configuration Tips
- CUDA rendering requires a CUDA-capable PyTorch install and the optional
  `gsplat` package in the active virtualenv.

## Known Issue: Segfault and .pyc Corruption
- Symptom: running diffvg segfaults; subsequent runs raise `bad marshal data`/`marshal data too short` due to corrupted `__pycache__/*.pyc` from the crash.
- Immediate mitigations:
  - Disable bytecode in dev: `PYTHONDONTWRITEBYTECODE=1` (enabled automatically via `sitecustomize.py` in repo).
  - Clean caches: `find . -name __pycache__ -exec rm -rf {} +` before re-run.
  - Force CPU execution during debugging with `DIFFVG_DEVICE=cpu` or `pydiffvg.set_device("cpu")`.
