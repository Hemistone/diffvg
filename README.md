# diffvg

This repository is no longer maintained as a general differentiable SVG renderer.
It is now a stroke-first raster-to-vector engine built around a compiled
open-stroke runtime and the `bezier_gsplat` backend.

Maintained scope:
- open stroke scenes only
- constant RGBA stroke colors
- scalar stroke widths
- `bezier_gsplat` as the only supported runtime backend
- painterly / precondition / line-art vectorization workflows

Final painterly artifacts:
- `final_splatted.png`: direct internal `bezier_gsplat` raster
- `final.svg`: canonical vector output
- `final.png`: preview rasterized back from `final.svg` on a white background

See:
- [`docs/stroke_first_reboot.md`](docs/stroke_first_reboot.md)
- [`docs/plotter_workflow_context.md`](docs/plotter_workflow_context.md)

## Install

Activate the repository virtualenv first:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
```

Install runtime dependencies:

```bash
pip install -r requirements.txt
```

Install the package itself:

```bash
pip install -e .
```

Notes:
- The maintained runtime is pure Python + PyTorch + `gsplat`.
- CUDA rendering requires a CUDA-capable PyTorch install and a compatible `gsplat` build in the active environment.
- Reference `gsplat` setup matches the one used by `../Bezier_splatting`.

## Run

### Painterly Rendering

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

### Preconditioning Only

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python apps/precondition_vectorize.py \
  apps/imgs/flower.jpg \
  --backend bezier_gsplat
```

### Line-Art Vectorization

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python apps/lineart_vectorize.py \
  apps/imgs/flower.jpg \
  --backend bezier_gsplat
```

### Benchmarks

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

### Minimal Runtime Smoke

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python apps/test_stroke_first_runtime.py
```

## Cite

If you use diffvg in your academic work, please cite:

```bibtex
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
