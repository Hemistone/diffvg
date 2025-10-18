"""Image IO helpers for pydiffvg.

Save tensors/arrays with gamma correction using a backend that does not rely on
skimage plugin autoloading (which can be brittle in some Python 3.12 stacks).
Prefers matplotlib; falls back to skimage only if explicitly available.
"""

from typing import Union
import os
import numpy as np
import torch
from typing import Optional

def _save_with_matplotlib(img_uint8, filename: str) -> bool:
    try:
        from matplotlib import image as mpimg
        mpimg.imsave(filename, img_uint8)
        return True
    except Exception:
        return False

def _save_with_skimage(img_uint8, filename: str) -> bool:
    try:
        import skimage.io
        skimage.io.imsave(filename, img_uint8)
        return True
    except Exception:
        return False


def imwrite(
    img: Union[np.ndarray, torch.Tensor],
    filename: str,
    gamma: float = 2.2,
    normalize: bool = False,
) -> None:
    """Save an image to disk.

    Parameters
    - img: HxWxC array or tensor in [0, 1]
    - filename: output path (directories created if missing)
    - gamma: gamma to apply to RGB channels (default 2.2)
    - normalize: if True, min-max normalizes the input before saving
    """
    directory = os.path.dirname(filename)
    if directory != '' and not os.path.exists(directory):
        os.makedirs(directory)

    if not isinstance(img, np.ndarray):
        # Accept torch.Tensor or objects with .data.numpy()
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        else:
            img = img.data
            if isinstance(img, torch.Tensor):
                img = img.detach().cpu().numpy()
            else:
                img = np.asarray(img)
    if normalize:
        img_rng = np.max(img) - np.min(img)
        if img_rng > 0:
            img = (img - np.min(img)) / img_rng
    # Sanitize numerics to avoid warnings and file corruption
    img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
    img = np.clip(img, 0.0, 1.0)
    if img.ndim==2:
        #repeat along the third dimension
        img=np.expand_dims(img,2)
    img[:, :, :3] = np.power(img[:, :, :3], 1.0/gamma)
    img_uint8 = (img * 255).astype(np.uint8, copy=False)
    if _save_with_matplotlib(img_uint8, filename):
        return
    if _save_with_skimage(img_uint8, filename):
        return
    # Last resort: use PIL if present
    try:
        from PIL import Image
        Image.fromarray(img_uint8).save(filename)
        return
    except Exception:
        pass
    raise RuntimeError("No available image backend (matplotlib/skimage/PIL) to save image.")

__all__ = ["imwrite"]
