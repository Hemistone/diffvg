# diffvg

Differentiable Rasterizer for Vector Graphics
https://people.csail.mit.edu/tzumao/diffvg

diffvg is a differentiable rasterizer for 2D vector graphics. See the webpage for more info.

![teaser](https://user-images.githubusercontent.com/951021/92184822-2a0bc500-ee20-11ea-81a6-f26af2d120f4.jpg)

![circle](https://user-images.githubusercontent.com/951021/63556018-0b2ddf80-c4f8-11e9-849c-b4ecfcb9a865.gif)
![path](https://user-images.githubusercontent.com/951021/64070625-7a52b480-cc19-11e9-9380-eac02f56f693.gif)
![gradient](https://user-images.githubusercontent.com/951021/64898668-da475300-d63c-11e9-917a-825b94be0710.gif)
![ellipse_transform](https://user-images.githubusercontent.com/951021/67149013-06b54700-f25b-11e9-91eb-a61171c6d4a4.gif)

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

```
cd apps
```

Optimizing a single circle to a target.

```
python single_circle.py
```

Finite difference comparison.

```
finite_difference_comp.py [-h] [--size_scale SIZE_SCALE]
                                [--clamping_factor CLAMPING_FACTOR]
                                [--use_prefiltering USE_PREFILTERING]
                                svg_file
```

e.g.,

```
python finite_difference_comp.py imgs/tiger.svg
```

Interactive editor

```
python svg_brush.py

### High-level optimization driver

For quick experiments without writing a custom training loop, use the
`SvgOptimizationDriver` helper that wraps the legacy `OptimizableSvg` API:

```python
import torch
import pydiffvg

pydiffvg.set_use_gpu(torch.cuda.is_available())

settings = pydiffvg.SvgOptimizationSettings()
driver = pydiffvg.SvgOptimizationDriver(
    "apps/imgs/note_small.svg",
    settings=settings,
    optimize_background=False,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)

target = torch.ones((driver.document.canvas[1], driver.document.canvas[0], 4), dtype=torch.float32)

def mse_loss(image, iteration, drv):
    return torch.nn.functional.mse_loss(image, target)

history = driver.optimize(mse_loss, iterations=5)
driver.save_svg("results/note_small_optimized.svg")
```

See `scripts/test_optimize_driver.py` for a runnable smoke test.
```

Painterly rendering

```
painterly_rendering.py [-h] [--num_paths NUM_PATHS]
                           [--max_width MAX_WIDTH] [--use_lpips_loss]
                           [--num_iter NUM_ITER] [--use_blob]
                           target
```

e.g.,

```
python painterly_rendering.py imgs/fallingwater.jpg --num_paths 2048 --max_width 4.0 --use_lpips_loss
```

Note: `--use_lpips_loss` uses PIQ’s `LPIPS` implementation under the hood. Ensure `piq` is installed in your environment (`pip install piq`). Inputs are kept in [0,1] and internally normalized to [-1,1] for PIQ.

Image vectorization

```
python refine_svg.py [-h] [--use_lpips_loss] [--num_iter NUM_ITER] svg target
```

e.g.,

```
python refine_svg.py imgs/flower.svg imgs/flower.jpg
```

Note: When `--use_lpips_loss` is provided, PIQ’s `LPIPS` is used. Install with `pip install piq` if it is missing.

Seam carving

```
python seam_carving.py [-h] [--svg SVG] [--optim_steps OPTIM_STEPS]
```

e.g.,

```
python seam_carving.py imgs/hokusai.svg
```

Vector variational autoencoder & vector GAN:

For the GAN models, see `apps/generative_models/train_gan.py`. Generate samples from a pretrained using `apps/generative_models/eval_gan.py`.

For the VAE models, see `apps/generative_models/mnist_vae.py`.

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
