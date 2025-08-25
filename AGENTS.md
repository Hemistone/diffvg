# Repository Guidelines

This repository is a modernized fork of https://github.com/BachiLi/diffvg aimed at Python 3.10+, CUDA 12.x, latest pybind11, and using CCCL/Thrust inherited from the CUDA Toolkit. Modernization is in progress; some rough edges remain.

## Project Structure & Module Organization
- `pydiffvg/`: Python API (PyTorch bindings, SVG parsing/rendering utilities).
- C++ sources in repo root: core rasterizer and bindings (`*.cpp`, `*.h`).
- `apps/`: runnable examples and small sanity checks (e.g., `single_circle.py`).
- `cmake/`: CMake helpers; `pybind11/`: submodule for bindings.
- Generated build outputs: `build/`, wheels in `dist/`.

## Build, Test, and Development Commands
- Install (CUDA default): `pip install .`
- CPU-only build: `DIFFVG_CUDA=0 pip install .`
- Manual CMake (GPU): `mkdir -p build && cd build && cmake -G Ninja -DDIFFVG_CUDA=1 .. && ninja -j`
- Debug build: `python setup.py build --debug install`
- Examples: `cd apps && python single_circle.py` (see `README.md` for more).
- Common env knobs: `CMAKE_CUDA_ARCHITECTURES=75;86;89`, `TORCH_CUDA_ARCH_LIST=75;86;89`.
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

## Commit & Pull Request Guidelines
- Commits: imperative, scoped messages (e.g., `build: set default CUDA arch to 89`, `fix(pydiffvg): handle empty path list`).
- PRs: include a clear summary, motivation, and usage snippet; link issues; note CPU vs CUDA impact; add before/after images for rendering changes when possible.
- CI/local checks: ensure `pip install .` succeeds and key `apps/` scripts run.

## Security & Configuration Tips
- Submodules: run `git submodule update --init --recursive` (required for `pybind11/`).
- CUDA builds require a CUDA 12.x toolkit; for CPU-only, set `DIFFVG_CUDA=0` and ensure Thrust headers are available or from Toolkit.

## Known Issue: Segfault and .pyc Corruption
- Symptom: running diffvg segfaults; subsequent runs raise `bad marshal data`/`marshal data too short` due to corrupted `__pycache__/*.pyc` from the crash.
- Immediate mitigations:
  - Disable bytecode in dev: `PYTHONDONTWRITEBYTECODE=1` (enabled automatically via `sitecustomize.py` in repo).
  - Clean caches: `find . -name __pycache__ -exec rm -rf {} +` before re-run.
  - Force CPU build to isolate CUDA path: `DIFFVG_CUDA=0 pip install .` then rerun examples.

## Debugging Plan (in progress)
- Reproduce with a minimal script (`import diffvg; import pydiffvg as d; d.render(...)`) on both CPU and CUDA builds.
- Build with symbols: `CMAKE_BUILD_TYPE=RelWithDebInfo` or `Debug`; run under `gdb --args python apps/single_circle.py` and collect backtrace.
- CPU path: run `valgrind --tool=memcheck python -X dev apps/single_circle.py` to catch invalid accesses.
- Host sanitizers: add `-fsanitize=address,undefined` to CMake for host files; CUDA: validate kernels after import (defer if sanitizer conflicts).
- Check ABI/linking: verify single libstdc++ ABI, pybind11 version, and that the module does not call CUDA when `torch.cuda.is_available()` is false.
- Once the crash site is known, patch C++/binding layer and add a smoke test in `apps/` to prevent regressions.
