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
- CUDA (default): `pip install .`
- CPU-only: `CMAKE_ARGS="-DDIFFVG_CUDA=0" pip install .` or `DIFFVG_CUDA=0 pip install .`

Notes
- CUDA Toolkit 12.x+ must be installed and `nvcc` available for the CUDA build.
- We source Thrust/CCCL headers from the CUDA Toolkit for both CPU and CUDA builds when available. For CPU-only builds without the Toolkit, install a system Thrust and ensure its include path is visible to CMake.
- GPU architectures: defaults to `89`. Set explicitly with `CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=75;86;89"`.
- You can also set `TORCH_CUDA_ARCH_LIST` or `DIFFVG_CUDA_ARCHS` (e.g. `80;86`).

### C) Build wheels (for distribution)
- CPU wheel: `python -m pip install build && python -m build`
- CUDA wheel: `CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=75;86;89" python -m build`

uv users
- Ensure uv targets the active venv (e.g. `uv pip install --python $(which python) .`).

Compatibility
- With CUDA 12+, very old GPU targets (e.g., 5.2) are avoided by default to prevent nvcc crashes. Override arches if needed: `-DCMAKE_CUDA_ARCHITECTURES=75;86;89`.

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
