# Contributing

Thanks for helping improve this modernized diffvg fork! The project prioritizes
incremental, well-tested changes that keep existing render behaviour intact.

## Environment Setup

- Use Python 3.10+ (we develop primarily on 3.12) and create a dedicated
  virtual environment.
- Install build dependencies via `pip install -r requirements-build.txt`
  and the runtime dependencies with `pip install -r requirements.txt`.
- Activate the virtual environment before building (`source .venv/bin/activate`).
- Install `pybind11>=3.0.1` into the environment when building from source:
  `pip install pybind11`.

## Build Workflows

### Editable/packaged build

```bash
pip install -e .
```

Set `DIFFVG_CUDA=0` to force a CPU-only build when CUDA is unavailable.

### Manual CMake build (CUDA)

```bash
rm -rf build && mkdir build
cmake -G Ninja \
  -DDIFFVG_CUDA=ON \
  -DDIFFVG_USE_SYSTEM_PYBIND11=ON \
  -DCMAKE_CUDA_FORWARD_UNKNOWN_TO_HOST_COMPILER=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=75\;86\;89 \
  -Dpybind11_DIR=$(python -m pybind11 --cmakedir) \
  ..
ninja -C build -j"$(nproc)"
```

Key flags:

- `DIFFVG_USE_SYSTEM_PYBIND11=ON` picks up the pip-installed pybind11.
- `CMAKE_CUDA_FORWARD_UNKNOWN_TO_HOST_COMPILER=OFF` avoids a known nvcc 12.x +
  GCC 12 segfault (`-forward-unknown-to-host-compiler`).
- Adjust `CMAKE_CUDA_ARCHITECTURES` for your GPU generation; the listed values
  cover Turing → Ada.

Host sanitizers can be enabled with `-DDIFFVG_SANITIZE=ON`, but GCC 12 may emit
internal compiler errors. Prefer GCC 13 or Clang when using ASan/UBSan.

## Testing & Validation

- Quick regression check: `make dev-check` (cleans caches, runs minimal smoke
  tests).
- CPU sanity: `python apps/single_circle.py`
- CUDA sanity: `python apps/single_rect.py`
- Lightweight render serialization check: `python scripts/test_render_paths.py`

When working on GPU code paths, also run one of the `apps/single_*.py` scripts
with CUDA enabled to confirm no device regressions.

## Debugging Tips

- Set `PYTHONDONTWRITEBYTECODE=1` (already default via `sitecustomize.py`) to
  avoid `.pyc` corruption if a segfault occurs.
- `DIFFVG_DISABLE_GPU_SORT=1` falls back to a host sort for boundary samples,
  which is useful when debugging Thrust/device issues.
- For runtime crashes, gather stack traces with `gdb --args python apps/...` on
  CPU, or use `cuda-gdb`/`nsys` for CUDA kernels.
- The backend selector lives in `pydiffvg/backends`; keep it working when
  introducing new RenderFunction variants.

## Coding Guidelines

- Follow the style expectations in `docs/refactor_todo.md` (C++14,
  lower_snake_case filenames, docstrings + type hints for public Python APIs).
- Keep changes small and update the TODO documents when tasks are completed.
- Avoid creating translation units that exceed roughly 1k lines; prefer
  extracting helpers to new files.

We welcome feedback on documentation, build tooling, and new smoke tests—feel
free to file issues or open PRs as you discover gaps.

