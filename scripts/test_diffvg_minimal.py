#!/usr/bin/env python3
"""
Direct diffvg extension smoke test (no pydiffvg).

Build prerequisite: `pip install .` (or CPU-only `DIFFVG_CUDA=0 pip install .`).
This creates a tiny scene with one filled circle and calls diffvg.render on CPU.
Exits 0 on success. Prints min/max of the rendered buffer for quick sanity.
"""
import ctypes as ct
import os
import sys

# Encourage safe dev defaults
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

try:
    import diffvg as dv
except Exception as e:
    print(f"[direct-test] Failed to import diffvg: {e}")
    sys.exit(2)

W, H = 64, 64

def c_floats(n):
    return (ct.c_float * n)()

def ptr_float(buf):
    return dv.float_ptr(ct.addressof(buf))

def ptr_int(buf):
    return dv.int_ptr(ct.addressof(buf))

# Geometry: one circle at center
circle = dv.Circle(10.0, dv.Vector2f(W/2.0, H/2.0))
shape = dv.Shape(dv.ShapeType.circle, circle.get_ptr(), 0.0)

# ShapeGroup: constant red fill, no stroke, identity transform
fill = dv.Constant(dv.Vector4f(1.0, 0.0, 0.0, 1.0))
shape_ids = (ct.c_int * 1)(0)
M = (ct.c_float * 9)(1,0,0, 0,1,0, 0,0,1)
group = dv.ShapeGroup(
    ptr_int(shape_ids), 1,
    dv.ColorType.constant, fill.get_ptr(),
    dv.ColorType.constant, dv.void_ptr(0),
    False,
    ptr_float(M),
)

# Scene: CPU path, box filter
scene = dv.Scene(
    W, H,
    [shape], [group],
    dv.Filter(dv.FilterType.box, 0.5),
    False, -1,
)

# Output buffer (RGBA)
img = c_floats(W * H * 4)

# Call render: supply only render_image; leave optional buffers as null
dv.render(
    scene,
    dv.float_ptr(0),               # background_image (nullptr)
    ptr_float(img),                # render_image
    dv.float_ptr(0),               # render_sdf
    W, H,
    1, 1,                         # samples per pixel
    0,                            # seed
    dv.float_ptr(0), dv.float_ptr(0), dv.float_ptr(0), dv.float_ptr(0),
    False,                        # use_prefiltering
    dv.float_ptr(0), 0            # eval_positions, num_eval_positions
)

# Quick sanity check
vals = [img[i] for i in range(min(16, W*H*4))]
print("[direct-test] ok; first-vals:", " ".join(f"{v:.3f}" for v in vals))
print("[direct-test] min/max:", f"{min(img):.3f}", f"{max(img):.3f}")
