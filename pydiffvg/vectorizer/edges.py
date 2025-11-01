"""Edge detection utilities used by the vectorizer pipeline."""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - optional dependency
    import cv2
except ModuleNotFoundError:  # pragma: no cover - fallback when OpenCV is missing
    cv2 = None  # type: ignore[assignment]


_SOBEL_X = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
_SOBEL_Y = _SOBEL_X.T


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an ``H×W`` or ``H×W×C`` image to grayscale float array."""

    if image.ndim == 2:
        gray = image.astype(np.float32, copy=False)
    elif image.ndim == 3:
        # Weighted RGB average that roughly matches luminance perception.
        weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
        if image.shape[2] == 1:
            gray = image[..., 0].astype(np.float32, copy=False)
        else:
            channels = image[..., : weights.shape[0]].astype(np.float32, copy=False)
            gray = np.tensordot(channels, weights[: channels.shape[-1]], axes=((-1,), (0,)))
    else:  # pragma: no cover - defensive branch for unexpected input
        raise ValueError("Unsupported image rank for grayscale conversion")

    if gray.max(initial=0.0) > 1.0:
        gray = gray / 255.0
    return gray


def sobel_edge_map(image: np.ndarray) -> np.ndarray:
    """Return the Sobel gradient magnitude for *image* as a float map."""

    gray = _to_grayscale(image)
    padded = np.pad(gray, 1, mode="edge")
    gx = _convolve3x3(padded, _SOBEL_X)
    gy = _convolve3x3(padded, _SOBEL_Y)
    mag = np.sqrt(gx * gx + gy * gy)
    if mag.size:
        mag = mag / (mag.max() + 1e-8)
    return mag.astype(np.float32)


def canny_edge_map(image: np.ndarray, *, low_threshold: float = 0.1, high_threshold: float = 0.3) -> np.ndarray:
    """Return a Canny edge probability map, falling back to Sobel when needed."""

    gray = _to_grayscale(image)
    if cv2 is not None:  # pragma: no cover - exercised in environments with OpenCV
        scaled = (gray * 255.0).astype(np.uint8)
        edges = cv2.Canny(scaled, int(low_threshold * 255), int(high_threshold * 255))
        return edges.astype(np.float32) / 255.0

    # Fallback: approximate Canny by hysteresis on Sobel magnitude.
    mag = sobel_edge_map(gray)
    high = mag > high_threshold
    low = mag > low_threshold
    # Simple hysteresis – propagate high responses into their 8-neighbourhood.
    strong = np.copy(high)
    h, w = mag.shape
    for y in range(h):
        for x in range(w):
            if high[y, x]:
                y0 = max(y - 1, 0)
                y1 = min(y + 2, h)
                x0 = max(x - 1, 0)
                x1 = min(x + 2, w)
                strong[y0:y1, x0:x1] |= low[y0:y1, x0:x1]
    return strong.astype(np.float32)


def _convolve3x3(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Light-weight 3×3 convolution used for Sobel responses."""

    h, w = image.shape
    out = np.empty((h - 2, w - 2), dtype=np.float32)
    for y in range(out.shape[0]):
        for x in range(out.shape[1]):
            out[y, x] = float(np.sum(image[y : y + 3, x : x + 3] * kernel))
    return out
