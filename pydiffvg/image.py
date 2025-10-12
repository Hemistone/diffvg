"""Image IO helpers for pydiffvg.

Provides a thin wrapper around skimage to save tensors/arrays with gamma.
"""

from typing import Union
import os
import numpy as np
import skimage
import skimage.io
import torch


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
    img = np.clip(img, 0.0, 1.0)
    if img.ndim==2:
        #repeat along the third dimension
        img=np.expand_dims(img,2)
    img[:, :, :3] = np.power(img[:, :, :3], 1.0/gamma)
    skimage.io.imsave(filename, (img * 255).astype(np.uint8))
