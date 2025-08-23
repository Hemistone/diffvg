# diffvg
Differentiable Rasterizer for Vector Graphics
https://people.csail.mit.edu/tzumao/diffvg

diffvg is a differentiable rasterizer for 2D vector graphics. See the webpage for more info.

![teaser](https://user-images.githubusercontent.com/951021/92184822-2a0bc500-ee20-11ea-81a6-f26af2d120f4.jpg)

![circle](https://user-images.githubusercontent.com/951021/63556018-0b2ddf80-c4f8-11e9-849c-b4ecfcb9a865.gif)
![ellipse](https://user-images.githubusercontent.com/951021/63556021-0ec16680-c4f8-11e9-8fc6-8b34de45b8be.gif)
![rect](https://user-images.githubusercontent.com/951021/63556028-12ed8400-c4f8-11e9-8072-81702c9193e1.gif)
![polygon](https://user-images.githubusercontent.com/951021/63980999-1e99f700-ca72-11e9-9786-1cba14d2d862.gif)
![curve](https://user-images.githubusercontent.com/951021/64042667-3d9e9480-cb17-11e9-88d8-2f7b9da8b8ab.gif)
![path](https://user-images.githubusercontent.com/951021/64070625-7a52b480-cc19-11e9-9380-eac02f56f693.gif)
![gradient](https://user-images.githubusercontent.com/951021/64898668-da475300-d63c-11e9-917a-825b94be0710.gif)
![circle_outline](https://user-images.githubusercontent.com/951021/65125594-84f7a280-d9aa-11e9-8bc4-669fd2eff2f4.gif)
![ellipse_transform](https://user-images.githubusercontent.com/951021/67149013-06b54700-f25b-11e9-91eb-a61171c6d4a4.gif)

## Install
PyTorch-only. TensorFlow support has been removed.

We use a PEP 517 build with CMake (scikit-build-core). Poetry is deprecated.

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
  - CUDA Toolkit 12.x: Install from NVIDIA’s apt repositories for your distro (recommended) or use the official runfile installer.

- Install uv without global pip
  - `curl -LsSf https://astral.sh/uv/install.sh | sh`
  - Ensure your shell PATH includes the installer’s suggested directory (e.g., `~/.local/bin`).

- Create and activate a venv:
  - `python3 -m venv .venv && source .venv/bin/activate`
  - `python -m pip install -U pip setuptools wheel`
  - `git submodule update --init --recursive`
    - Note: You do not need a Thrust/CCCL submodule. We use the CUDA Toolkit's CCCL (preferred) or a system-installed Thrust if present.

### A) Install runtime dependencies
- With uv: `uv pip install -r requirements.txt`
- With pip: `pip install -r requirements.txt`

Install PyTorch/torchvision first if you need CUDA wheels (recommended):
- CUDA 12.4+ wheels: `uv pip install --index-url https://download.pytorch.org/whl/cu124 'torch>=2.4,<2.6' 'torchvision>=0.19,<0.21'`
- CPU-only wheels: `uv pip install --index-url https://download.pytorch.org/whl/cpu 'torch>=2.4,<2.6' 'torchvision>=0.19,<0.21'`

### B) Build and install diffvg into venv (recommended)
- With pip
  - CUDA (default): `pip install .`
  - CPU-only: `CMAKE_ARGS="-DDIFFVG_CUDA=0" pip install .` or `DIFFVG_CUDA=0 pip install .`

- With uv (behaves like pip for local builds)
  - CUDA (default): `uv pip install .`
  - CPU-only: `DIFFVG_CUDA=0 uv pip install .`

Notes
- CUDA Toolkit 12.x+ must be installed and `nvcc` available for the CUDA build.
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
- Prereqs: `cmake` (>=3.25), `ninja`, a C++14 compiler, and for CUDA builds a CUDA Toolkit 12.x with `nvcc`.
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
- With CUDA 12+, very old GPU targets (e.g., 5.2) are avoided by default to prevent nvcc crashes. Override arches if needed: `-DCMAKE_CUDA_ARCHITECTURES=75;86;89`.

# Troubleshooting
- CMake not found/too old: Prefer system install via apt (`sudo apt-get install cmake`) or the Kitware APT repo (see above). Verify with `cmake --version`.
- Ninja not found: Prefer apt (`sudo apt-get install ninja-build`), or omit `-G Ninja` to use Makefiles/Visual Studio. As a last resort, install via `pip` inside your venv.
- PyTorch missing: Build requires PyTorch. Install Torch/Torchvision first (see A) for CUDA/CPU indices).
- CUDA toolkit not detected: Ensure `nvcc` is on `PATH` and the CUDA Toolkit 12.x is installed. You can also point CMake explicitly: `-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc`.
- Unsupported GPU arch: If you see errors like `unsupported gpu architecture`, set `-DCMAKE_CUDA_ARCHITECTURES=75;86;89` (or a list matching your GPUs) or export `TORCH_CUDA_ARCH_LIST="75;86;89"`.
- CPU-only Thrust include path: Without a CUDA Toolkit, provide Thrust headers or install a system Thrust and set `-DTHRUST_INCLUDE_DIR=/path/to/thrust`.
- Build isolation quirks: If your toolchain is not discovered during isolated builds, use `pip install --no-build-isolation .` (or `uv pip install --no-build-isolation .`).
- Submodules: If you see missing pybind headers, run `git submodule update --init --recursive`.

# Building in debug mode

```
python setup.py build --debug install
```

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

Image vectorization
```
python refine_svg.py [-h] [--use_lpips_loss] [--num_iter NUM_ITER] svg target
```
e.g.,
```
python refine_svg.py imgs/flower.svg imgs/flower.jpg
```

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
