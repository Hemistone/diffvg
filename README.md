# diffvg

This repository is no longer maintained as a general differentiable SVG
renderer. It is now a stroke-first raster-to-vector engine built around a
compiled open-stroke runtime and the `bezier_gsplat` backend.

Maintained product scope:
- open stroke scenes only
- constant RGBA stroke colors
- scalar stroke widths
- `bezier_gsplat` as the only supported runtime backend
- painterly / precondition / line-art vectorization workflows

Final painterly artifacts now follow this convention:
- `final_splatted.png`: direct internal `bezier_gsplat` raster
- `final.svg`: canonical vector output
- `final.png`: preview rasterized back from `final.svg` on a white background

Legacy exact-renderer backends, fill-heavy demos, SDF demos, and old sample apps
have been removed from the maintained path.

See:
- [`docs/stroke_first_reboot.md`](docs/stroke_first_reboot.md)
- [`docs/plotter_workflow_context.md`](docs/plotter_workflow_context.md)

## Install

PyTorch-only. TensorFlow support has been removed.

We use a PEP 517 build with CMake (scikit-build-core). Legacy `setup.py` builds are removed.

Important: activate the Python environment you intend to use before running any `pip`/`uv` commands to avoid mixing environments.

  - System prerequisites (Debian/Ubuntu via apt)

  - `sudo apt-get update`

  - `sudo apt-get install -y python3 python3-venv build-essential git ninja-build cmake`

      - Need a newer CMake? Use the Kitware APT repo:
          - `sudo apt-get install -y apt-transport-https ca-certificates gnupg lsb-release`
          - `sudo mkdir -p /etc/apt/keyrings`
          - `curl -fsSL https://apt.kitware.com/keys/kitware-archive-latest.asc | sudo gpg --dearmor -o /etc/apt/keyrings/kitware-archive-keyring.gpg`
          - `echo "deb [signed-by=/etc/apt/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/kitware.list`
          - `sudo apt-get update && sudo apt-get install -y cmake`
      - CUDA Toolkit 13.x: Install from NVIDIA’s apt repositories for your distro (recommended) or use the official runfile installer.

  - Install uv without global pip

      - `curl -LsSf https://astral.sh/uv/install.sh | sh`
      - Ensure your shell PATH includes the installer’s suggested directory (e.g., `~/.local/bin`).

  - **Clone the repository**

      - `git clone https://github.com/BachiLi/diffvg.git`
      - `cd diffvg`

  - Create and activate a venv:

      - `python3 -m venv .venv && source .venv/bin/activate`
      - `python -m pip install -U pip setuptools wheel`
      - `pip install -r requirements-build.txt` *(optional but recommended for local builds; includes `pybind11` and modern CMake/Ninja)*
      - If you skip `requirements-build.txt`, install `pybind11>=3.0.1` manually: `pip install pybind11`

### A) Install runtime dependencies

  - With uv: `uv pip install -r requirements.txt`
  - With pip: `pip install -r requirements.txt`

Install PyTorch/torchvision first if you need CUDA wheels (recommended):

  - CUDA 13.0 wheels: `uv pip install --index-url https://download.pytorch.org/whl/cu130 'torch>=2.9' 'torchvision>=0.24'`
  - CPU-only wheels: `uv pip install --index-url https://download.pytorch.org/whl/cpu 'torch>=2.9' 'torchvision>=0.24'`

### B) Build and install diffvg into venv (recommended)

  - With pip

      - CUDA (default): `pip install .`
      - CPU-only: `CMAKE_ARGS="-DDIFFVG_CUDA=0" pip install .` or `DIFFVG_CUDA=0 pip install .`

  - With uv (behaves like pip for local builds)

      - CUDA (default): `uv pip install .`
      - CPU-only: `DIFFVG_CUDA=0 uv pip install .`

Notes

  - CUDA Toolkit 13.x must be installed and `nvcc` available for the CUDA build.
  - We source Thrust/CCCL headers from the CUDA Toolkit for both CPU and CUDA builds when available. For CPU-only builds without the Toolkit, install a system Thrust and ensure its include path is visible to CMake.
  - GPU architectures: defaults to `89`. Set explicitly with `CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=75;86;89"`.
  - You can also set `TORCH_CUDA_ARCH_LIST` or `DIFFVG_CUDA_ARCHS` (e.g. `80;86`).

### C) Build wheels (for distribution)

  - CPU wheel: `python -m pip install build && python -m build`
  - CUDA wheel: `CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=75;86;89" python -m build`

uv users

  - In an active venv, `uv pip install .` is sufficient.
  - Build wheels with uv: `uv pip install build && uv run -m build`

### D) Manual CMake build (Ninja) + wheel

  - Prereqs: `cmake` (\>=3.25), `ninja`, a C++14 compiler, and for CUDA builds a CUDA Toolkit 13.x with `nvcc`.
  - Note: You must have `pybind11>=3.0.1` available (e.g., `pip install pybind11` in your active environment) or pass `-Dpybind11_DIR=$(python -m pybind11 --cmakedir)` to CMake.
  - Configure + build (GPU example):
      - `mkdir -p build && cd build`
      - `cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DDIFFVG_CUDA=1 -DCMAKE_CUDA_ARCHITECTURES=75;86;89 ..`
      - `ninja -j`
  - Configure + build (CPU): use `-DDIFFVG_CUDA=0` and omit CUDA architectures.
  - Produce a wheel (packaging): Prefer letting scikit-build drive CMake to ensure correct layout:
      - `cd ..`
      - `python -m pip install build`
      - `CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=75;86;89" python -m build`
      - With uv: `uv pip install build && CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=75;86;89" uv run -m build`
      - The wheel lands in `dist/`.

Compatibility

  - With CUDA 13+, very old GPU targets (e.g., 5.2) are avoided by default to prevent nvcc crashes. Override arches if needed: `-DCMAKE_CUDA_ARCHITECTURES=75;86;89`.

# Troubleshooting

  - **CMake Error: Cannot find pybind11**: Install it in your environment (`pip install pybind11>=3.0.1`) and/or re-run CMake with `-Dpybind11_DIR=$(python -m pybind11 --cmakedir)`.
  - CMake not found/too old: Prefer system install via apt (`sudo apt-get install cmake`) or the Kitware APT repo (see above). Verify with `cmake --version`.
  - Ninja not found: Prefer apt (`sudo apt-get install ninja-build`), or omit `-G Ninja` to use Makefiles/Visual Studio. As a last resort, install via `pip` inside your venv.
  - PyTorch missing: Build requires PyTorch. Install Torch/Torchvision first (see A) for CUDA/CPU indices).
  - CUDA toolkit not detected: Ensure `nvcc` is on `PATH` and the CUDA Toolkit 13.x is installed. You can also point CMake explicitly: `-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc`.
  - Unsupported GPU arch: If you see errors like `unsupported gpu architecture`, set `-DCMAKE_CUDA_ARCHITECTURES=75;86;89` (or a list matching your GPUs) or export `TORCH_CUDA_ARCH_LIST="75;86;89"`.
  - CPU-only Thrust include path: Without a CUDA Toolkit, provide Thrust headers or install a system Thrust and set `-DTHRUST_INCLUDE_DIR=/path/to/thrust`.
  - Build isolation quirks: If your toolchain is not discovered during isolated builds, use `pip install --no-build-isolation .` (or `uv pip install --no-build-isolation .`).

# Building in debug mode

Use scikit-build via pip and pass CMake debug flags.

- CUDA (default):
  - `CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Debug" pip install -v .`
- CPU-only:
  - `CMAKE_ARGS="-DDIFFVG_CUDA=0 -DCMAKE_BUILD_TYPE=Debug" pip install -v .`

Optional sanitizers (host code only): `CMAKE_ARGS="-DDIFFVG_SANITIZE=ON -DCMAKE_BUILD_TYPE=Debug" pip install -v .`

# Run

Activate the repository venv first:

```bash
source .venv/bin/activate
```

## Painterly Rendering

Pure random-init stroke optimization is the default path:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python apps/painterly_rendering.py \
  apps/imgs/fallingwater.jpg \
  --backend bezier_gsplat \
  --num-paths 2048 \
  --num-iter 256
```

Preconditioned stroke initialization is explicit and opt-in:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python apps/painterly_rendering.py \
  apps/imgs/flower.jpg \
  --backend bezier_gsplat \
  --precondition \
  --precond-mode teed \
  --num-paths 2048 \
  --num-iter 256
```

## Preconditioning Only

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python apps/precondition_vectorize.py \
  apps/imgs/flower.jpg \
  --backend bezier_gsplat
```

## Line-Art Vectorization

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python apps/lineart_vectorize.py \
  apps/imgs/flower.jpg \
  --backend bezier_gsplat
```

## Benchmarks

Painterly benchmark:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python apps/bench_painterly_backends.py \
  apps/imgs/flower.jpg \
  --backends bezier_gsplat \
  --path-counts 512,1024 \
  --num-iter 64 \
  --repeats 1
```

Renderer microbenchmark:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python apps/bench_renderer_micro.py \
  --backends bezier_gsplat \
  --path-counts 128,512,1024 \
  --repeats 5 \
  --warmup 2
```

## Minimal Runtime Smoke

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python apps/test_stroke_first_runtime.py
```

If you use diffvg in your academic work, please cite

```
@article{Li:2020:DVG,
    title = {Differentiable Vector Graphics Rasterization for Editing and Learning},
    author = {Li, Tzu-Mao and Luk\'{a}\v{c}, Michal and Gharbi Micha\"{e}l and Jonathan Ragan-Kelley},
    journal = {ACM Trans. Graph. (Proc. SIGGRAPH Asia)},
    volume = {39},
    number = {6},
    pages = {193:1--193:15},
    year = {2020}
}
```
