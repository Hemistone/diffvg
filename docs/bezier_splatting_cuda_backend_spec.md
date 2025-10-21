
# Bézier Splatting CUDA Backend — **Unified Spec (Option A) aligned with existing backend/registry**

**Audience:** code agent (English)  
**Purpose:** Single, self-contained document to implement the CUDA backend for Bézier Splatting and integrate it into the existing `diffvg` codebase using **Option A (runtime JIT)**. This document is aligned with the **already present backend selector and registry** in `pydiffvg/backend.py` and `pydiffvg/backends/registry.py`.

---

## 0) What already exists (and must be respected)

- **Backend selection & env config**: `pydiffvg/backend.py` already exposes `set_backend/get_backend/current_api`, reads `DIFFVG_BACKEND` (`baseline|splat`) and splat configs (`DIFFVG_SPLAT_*`, `DIFFVG_DEPTH_POLICY`) and returns a `RenderAPI` via the registry.
- **Backend registry**: `pydiffvg/backends/registry.py` defines `RenderAPI` (serialize_scene/apply/render_grad), returns either the **baseline** API or the **splat** API (wired to `render_splat.py`). Users are expected to select backends through `set_backend`/`get_backend`.
- **Backends package**: `pydiffvg/backends/__init__.py` exists and is internal.

**Implication:** We **do not** create a new `backend.py`. We keep the current `backend.py` + registry and integrate the CUDA path inside the **splat** backend’s code (`render_splat.py` and a new JIT loader inside `pydiffvg/splat/`).

---

## 1) High-level plan (Option A)

- Keep a **single** Python package name: `diffvg` (installed via `pip install .` as today).
- Leave **baseline** backend untouched.
- Implement CUDA kernels + PyTorch extension under `pydiffvg/splat/cuda/` and load them **JIT at first use** (only if `DIFFVG_BACKEND=splat`).
- Hook points are in the **splat** backend code path only (what `registry.get_api("splat")` already returns).

**Why this works cleanly:** the registry hands out a `RenderAPI` pointing to **splat** functions in `render_splat.py`. We just make the splat path call into our CUDA extension instead of the heavy Python/Torch compositing.

---

## 2) Repository layout additions

```
diffvg/
  pydiffvg/
    backend.py                  # keeps env, config, and selection (no changes)  (uses registry)  <-- keep
    backends/
      __init__.py               # internal                                              <-- keep
      registry.py               # RenderAPI bundling + get_api/list_backends            <-- keep
    render_splat.py             # splat backend orchestrator (to call CUDA paths)
    splat/
      __init__.py
      geometry.py               # existing Python geom sampling
      compositor.py             # existing Python compositing (kept as correctness ref/fallback)
      gauss.py                  # existing
      trace.py                  # existing
      types.py                  # existing
      vjp.py                    # existing
      runtime_cuda.py           # NEW: JIT loader & thin wrapper for CUDA op
      cuda/
        cuda_splat.cpp          # NEW: PyBind11 + ATen bindings (build_tile_csr, forward, backward)
        cuda_splat_kernel.cu    # NEW: core kernels
```

---

## 3) CUDA backend — functionality (unchanged from prior spec)

### 3.1 Data layout (SoA)
```cpp
struct GaussSOA {
  float* mu_x; float* mu_y; float* theta;
  float* sigma_x; float* sigma_y;
  float* color_r; float* color_g; float* color_b;
  float* opacity;
  int    N;
};
```
- Inputs can be FP16/BF16; **accumulators remain FP32** for stability.
- Pixel center coordinates `(x, y) = (j+0.5, i+0.5)`.

### 3.2 Kernels (summary)
1) **curve_to_gauss_kernel**: Bézier segment sampling (arc-length approx), fused computation of `(mu, theta, sigma_x, sigma_y, opacity, color)` into SoA.  
2) **build_tile_csr**: 2-pass CSR construction (count → prefix-sum → scatter) to map splats to tiles.  
3) **composite_tiled_fwd**: block-per-tile; shared memory accumulators (`acc_rgb`, `acc_alpha` FP32); rotated anisotropic Gaussian eval; α-blend with transmission.  
4) **composite_tiled_bwd**: reproduces forward; computes gradients w.r.t. `(mu_x,mu_y,theta,sigma_x,sigma_y,color,opacity)` and accumulates per-splat grads.

### 3.3 PyTorch extension interface (PyBind11)
```cpp
py::tuple build_tile_csr(Tensor mu_x, Tensor mu_y, Tensor theta,
                         Tensor sigma_x, Tensor sigma_y, Tensor opacity,
                         int width, int height, int tile);

Tensor forward_tiled(const GaussSOA& G, int W, int H, int tile);
Tensor forward_full (const GaussSOA& G, int W, int H);

std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,Tensor>
backward_tiled(const GaussSOA& G, int W, int H, int tile,
               const Tensor& grad_img);
```

---

## 4) JIT loader (`pydiffvg/splat/runtime_cuda.py`)

```python
# runtime_cuda.py
import os
from pathlib import Path
from torch.utils.cpp_extension import load

_mod = None

def _cuda_cflags():
    return [
        "-O3", "--use_fast_math",
        "-Xptxas", "-O3", "-Xptxas", "-dlcm=ca",
        "-gencode=arch=compute_89,code=sm_89"  # Ada 4090
    ]

def _cxx_cflags():
    # Match PyTorch 2.9 ABI
    return ["-O3","-DNDEBUG","-fPIC","-std=c++17","-D_GLIBCXX_USE_CXX11_ABI=1"]

def get_module():
    global _mod
    if _mod is None:
        root = Path(__file__).parent / "cuda"
        srcs = [str(root/"cuda_splat.cpp"), str(root/"cuda_splat_kernel.cu")]
        ccbin = os.environ.get("CUDA_HOST_CCBIN", "/usr/bin/g++-12")  # WSL2: GCC host preferred
        extra_cuda = _cuda_cflags() + [f"-ccbin={ccbin}"]
        _mod = load(
            name="cuda_splat",
            sources=srcs,
            extra_cflags=_cxx_cflags(),
            extra_cuda_cflags=extra_cuda,
            with_cuda=True, verbose=False
        )
    return _mod
```

- JIT builds once and caches under `~/.cache/torch_extensions`.
- You may expose optional knobs via env (e.g., `DIFFVG_SPLAT_TILE`, `CUDA_HOST_CCBIN`).

---

## 5) Wiring in `render_splat.py` (splat backend only)

Inside `render_splat.py`:
- Keep existing Python geometry sampling to produce `GaussSOA` tensors (or migrate to CUDA later).
- When composing:
  - Read tile size/depth policy from `pydiffvg.backend.get_backend_config()` (already provided).
  - Call JIT module functions:
    - `tile_ptr,tile_idx = cuda_mod.build_tile_csr(...)`
    - `img = cuda_mod.forward_tiled(...)`
  - Save `(GaussSOA, CSR)` in autograd context for backward.
  - In backward, call `cuda_mod.backward_tiled(...)` and map results to input grads.

The public API surface does **not** change; the registry still returns the `RenderAPI` that `backend.current_api()` supplies.

---

## 6) Runtime selection (unchanged)

- Users select at runtime: `DIFFVG_BACKEND=splat` (or via `pydiffvg.set_backend("splat")`), otherwise `baseline`.  
- `pydiffvg/backend.py` continues to own env parsing and the splat configuration (`K, R, rho, tile, depth_policy`).

---

## 7) Environment (validated, non-deprecated for your stack)

- **CUDA arch**: `-gencode=arch=compute_89,code=sm_89` (Ada 4090).  
- **Host compiler (WSL2)**: prefer **GCC-12** (`/usr/bin/g++-12`) for nvcc; if clang host is required, add `--allow-unsupported-compiler -ccbin=/usr/bin/clang++` (less tested).  
- **C++ ABI**: `-D_GLIBCXX_USE_CXX11_ABI=1` to match PyTorch 2.9.  
- **WSL2 FS**: keep repo/build under Linux FS (avoid `/mnt/c/...`).  
- **Precision**: inputs may be FP16/BF16; **accumulators FP32**.  
- **CUDA Graphs**: enable only with static shapes and preallocated buffers.  
- **PyBind11**: `3.0.1` should work; if ABI/link friction arises, upgrade to `>=2.12`.  
- **`torch.compile`**: allowed around Python glue (`fullgraph=True`, `mode="reduce-overhead"`, `enable_cudagraphs=True`); custom CUDA op remains as is.

---

## 8) Testing & Benchmarks

1) **Unit**: single splat forward/backward vs finite-diff.  
2) **Integration**: random scenes vs Python compositing — target PSNR/SSIM thresholds.  
3) **Perf**: 1080p/2K × {1k,2k,4k} splats; tile {32,64}; record FW/BW times and VRAM.  
4) **Edge cases**: extreme `σ`, `α`, many overlaps, different depth policies.  
5) **Graphs**: graph-captured steady-state run with fixed shapes.

---

## 9) Milestones

1. CSR binning kernel → integrate in `render_splat.py` (splat path).  
2. Forward tiled kernel → parity check vs Python, measure speedup.  
3. Backward tiled kernel → wire autograd, verify gradients.  
4. Mixed precision / CUDA Graphs / Top-K tuning.  
5. Document & baseline benchmarks.

---

## 10) Deliverables Checklist

- [ ] `pydiffvg/splat/cuda/cuda_splat.{cpp,cu}` with three entry points.  
- [ ] `pydiffvg/splat/runtime_cuda.py` (JIT loader).  
- [ ] `pydiffvg/render_splat.py` calls CUDA ops in the **splat** path.  
- [ ] Tests and benchmarks.  
- [ ] (Optional) prebuilt wheels later.

