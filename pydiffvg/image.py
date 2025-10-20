"""Image IO helpers for pydiffvg (minimal, dependency-light).

This module provides a small PNG writer implemented with the Python standard
library (zlib/struct) and a tiny adapter for torch/NumPy inputs. It avoids
matplotlib/skimage hard dependencies and only uses Pillow if explicitly
available, falling back to the built-in writer otherwise.
"""

from __future__ import annotations

from typing import Union
import os
import struct
import zlib
import numpy as np
import torch


def _ensure_numpy(img: Union[np.ndarray, torch.Tensor, object]) -> np.ndarray:
    if isinstance(img, np.ndarray):
        return img
    if isinstance(img, torch.Tensor):
        return img.detach().cpu().numpy()
    # Torch-wrapped or array-like
    data = getattr(img, "data", img)
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.asarray(data)


def _write_png_raw(filename: str, img_uint8: np.ndarray) -> None:
    """Write an HxWxC uint8 image as PNG using only stdlib.

    Supports C=3 (RGB) and C=4 (RGBA), bit depth=8, no interlace, filter=0.
    """
    if img_uint8.ndim != 3 or img_uint8.shape[2] not in (3, 4):
        raise ValueError("PNG writer expects HxWx3 or HxWx4 uint8 array")
    H, W, C = img_uint8.shape
    color_type = 6 if C == 4 else 2  # RGBA or RGB

    def chunk(tag: bytes, data: bytes) -> bytes:
        length = struct.pack(">I", len(data))
        crc = struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        return length + tag + data + crc

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(
        ">IIBBBBB",
        W,
        H,
        8,          # bit depth
        color_type, # color type
        0,          # compression
        0,          # filter
        0,          # interlace
    )

    # No filtering (type 0) per row
    raw = img_uint8
    if not raw.flags.c_contiguous:
        raw = np.ascontiguousarray(raw)
    rows = []
    stride = W * C
    view = raw.reshape(H, stride)
    for y in range(H):
        rows.append(b"\x00")  # filter type 0
        rows.append(view[y].tobytes())
    idat = zlib.compress(b"".join(rows), level=6)

    with open(filename, "wb") as f:
        f.write(sig)
        f.write(chunk(b"IHDR", ihdr))
        # Optional: gAMA chunk to approximate sRGB gamma if desired. Omit to keep minimal.
        f.write(chunk(b"IDAT", idat))
        f.write(chunk(b"IEND", b""))


def imwrite(
    img: Union[np.ndarray, torch.Tensor],
    filename: str,
    gamma: float = 2.2,
    normalize: bool = False,
) -> None:
    """Save an image to disk as PNG.

    Parameters
    - img: HxWxC array or tensor in [0, 1]
    - filename: output path (directories created if missing)
    - gamma: gamma to apply to RGB channels (default 2.2)
    - normalize: if True, min-max normalizes the input before saving
    """
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    arr = _ensure_numpy(img)
    if arr.ndim == 2:
        arr = np.expand_dims(arr, 2)
    if arr.shape[-1] not in (3, 4):
        # Coerce to RGB
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=2)
        else:
            raise ValueError("imwrite expects last channel to be 3 or 4")
    if normalize:
        rng = float(arr.max() - arr.min())
        if rng > 0:
            arr = (arr - arr.min()) / rng
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    arr = np.clip(arr, 0.0, 1.0)
    # Apply gamma to RGB channels only
    arr = arr.copy()
    arr[:, :, :3] = np.power(arr[:, :, :3], 1.0 / float(gamma))
    arr_u8 = (arr * 255.0 + 0.5).astype(np.uint8)

    # Try Pillow if available (faster for some environments); else pure stdlib PNG
    try:
        from PIL import Image  # type: ignore

        Image.fromarray(arr_u8).save(filename)
        return
    except Exception:
        _write_png_raw(filename, arr_u8)


__all__ = ["imwrite"]
