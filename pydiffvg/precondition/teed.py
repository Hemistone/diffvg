"""TEED (Tiny and Efficient Edge Detector) backend for preconditioning.

This module vendors a minimal TED/TEED model definition (MIT-licensed upstream)
and exposes helper functions to produce an edge strength map and a boolean
edge mask compatible with the existing skeletonization/vectorization pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from skimage import filters, morphology

from .config import PreconditionConfig


@torch.jit.script
def _fsmish(input: torch.Tensor) -> torch.Tensor:
    return input * torch.tanh(torch.log(1 + torch.sigmoid(input)))


class _Smish(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        return _fsmish(input)


def _weight_init(m: nn.Module) -> None:
    if isinstance(m, (nn.Conv2d,)):
        torch.nn.init.xavier_normal_(m.weight, gain=1.0)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    if isinstance(m, (nn.ConvTranspose2d,)):
        torch.nn.init.xavier_normal_(m.weight, gain=1.0)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


class _DoubleFusion(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        # Keep attribute names aligned with upstream TEED implementations so
        # published weights load without remapping.
        self.DWconv1 = nn.Conv2d(in_ch, in_ch * 8, kernel_size=3, stride=1, padding=1, groups=in_ch)
        self.PSconv1 = nn.PixelShuffle(1)
        self.DWconv2 = nn.Conv2d(24, 24 * 1, kernel_size=3, stride=1, padding=1, groups=24)
        self.AF = _Smish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.PSconv1(self.DWconv1(self.AF(x)))
        attn2 = self.PSconv1(self.DWconv2(self.AF(attn)))
        return _fsmish(((attn2 + attn).sum(1)).unsqueeze(1))


class _DenseLayer(nn.Sequential):
    def __init__(self, input_features: int, out_features: int):
        super().__init__()
        self.add_module(
            "conv1",
            nn.Conv2d(
                input_features,
                out_features,
                kernel_size=3,
                stride=1,
                padding=2,
                bias=True,
            ),
        )
        self.add_module("smish1", _Smish())
        self.add_module("conv2", nn.Conv2d(out_features, out_features, kernel_size=3, stride=1, bias=True))

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x1, x2 = x
        new_features = super().forward(_fsmish(x1))
        return 0.5 * (new_features + x2), x2


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers: int, input_features: int, out_features: int):
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(input_features, out_features)
            self.add_module(f"denselayer{i + 1}", layer)
            input_features = out_features


class _UpConvBlock(nn.Module):
    def __init__(self, in_features: int, up_scale: int):
        super().__init__()
        self.constant_features = 16
        layers = self._make_deconv_layers(in_features, up_scale)
        self.features = nn.Sequential(*layers)

    def _make_deconv_layers(self, in_features: int, up_scale: int) -> list[nn.Module]:
        layers: list[nn.Module] = []
        all_pads = [0, 0, 1, 3, 7]
        for i in range(up_scale):
            kernel_size = 2**up_scale
            pad = all_pads[up_scale]
            out_features = 1 if i == up_scale - 1 else self.constant_features
            layers.append(nn.Conv2d(in_features, out_features, 1))
            layers.append(_Smish())
            layers.append(nn.ConvTranspose2d(out_features, out_features, kernel_size, stride=2, padding=pad))
            in_features = out_features
        return layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class _SingleConvBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, stride: int, use_ac: bool = False):
        super().__init__()
        self.use_ac = use_ac
        self.conv = nn.Conv2d(in_features, out_features, 1, stride=stride, bias=True)
        if self.use_ac:
            self.smish = _Smish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.use_ac:
            return self.smish(x)
        return x


class _DoubleConvBlock(nn.Module):
    def __init__(self, in_features: int, mid_features: int, out_features: int | None = None, stride: int = 1, use_act: bool = True):
        super().__init__()
        self.use_act = use_act
        if out_features is None:
            out_features = mid_features
        self.conv1 = nn.Conv2d(in_features, mid_features, 3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(mid_features, out_features, 3, padding=1)
        self.smish = _Smish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.smish(x)
        x = self.conv2(x)
        if self.use_act:
            x = self.smish(x)
        return x


class TED(nn.Module):
    """Tiny and Efficient Edge Detector (TED) network definition."""

    def __init__(self):
        super().__init__()
        self.block_1 = _DoubleConvBlock(3, 16, 16, stride=2)
        self.block_2 = _DoubleConvBlock(16, 32, use_act=False)
        self.dblock_3 = _DenseBlock(1, 32, 48)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.side_1 = _SingleConvBlock(16, 32, 2)
        self.pre_dense_3 = _SingleConvBlock(32, 48, 1)
        self.up_block_1 = _UpConvBlock(16, 1)
        self.up_block_2 = _UpConvBlock(32, 1)
        self.up_block_3 = _UpConvBlock(48, 2)
        self.block_cat = _DoubleFusion(3, 3)
        self.apply(_weight_init)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f"expected NCHW input, got {tuple(x.shape)}")

        block_1 = self.block_1(x)
        block_1_side = self.side_1(block_1)

        block_2 = self.block_2(block_1)
        block_2_down = self.maxpool(block_2)
        block_2_add = block_2_down + block_1_side

        block_3_pre_dense = self.pre_dense_3(block_2_down)
        block_3, _ = self.dblock_3([block_2_add, block_3_pre_dense])

        out_1 = self.up_block_1(block_1)
        out_2 = self.up_block_2(block_2)
        out_3 = self.up_block_3(block_3)
        results: list[torch.Tensor] = [out_1, out_2, out_3]

        block_cat = torch.cat(results, dim=1)
        block_cat = self.block_cat(block_cat)
        results.append(block_cat)
        return results


_MODEL_CACHE: Dict[Tuple[str, str], TED] = {}


def _strip_module_prefix(state: dict) -> dict:
    if not state:
        return state
    if any(k.startswith("module.") for k in state.keys()):
        return {k[len("module."):]: v for k, v in state.items()}
    return state


def _load_teed_model(weights_path: str | Path, device: torch.device) -> TED:
    weights_path = str(Path(weights_path))
    if not Path(weights_path).is_file():
        raise FileNotFoundError(f"TEED weights not found: {weights_path}")
    cache_key = (weights_path, str(device))
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    model = TED()
    raw = torch.load(weights_path, map_location="cpu")
    if isinstance(raw, dict) and "state_dict" in raw and isinstance(raw["state_dict"], dict):
        raw = raw["state_dict"]
    if not isinstance(raw, dict):
        raise ValueError(f"Unsupported TEED weights format: expected a state_dict dict, got {type(raw)}")
    raw = _strip_module_prefix(raw)
    missing, unexpected = model.load_state_dict(raw, strict=False)
    if missing:
        raise ValueError(f"TEED weights missing keys (first 10): {missing[:10]}")
    if unexpected:
        # Keep strict-ish: mismatched weights can silently degrade quality.
        raise ValueError(f"TEED weights have unexpected keys (first 10): {unexpected[:10]}")

    model.eval()
    model.to(device)
    _MODEL_CACHE[cache_key] = model
    return model


def _resize_like_controlnet_aux(img: np.ndarray, resolution: int) -> np.ndarray:
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"expected HxWx3 uint8 image, got shape {img.shape}")
    if img.dtype != np.uint8:
        raise ValueError(f"expected uint8 image, got dtype {img.dtype}")
    h, w = img.shape[:2]
    k = float(resolution) / float(min(h, w))
    new_h = int(np.round((h * k) / 64.0)) * 64
    new_w = int(np.round((w * k) / 64.0)) * 64
    new_h = max(64, new_h)
    new_w = max(64, new_w)
    if new_h == h and new_w == w:
        return img
    resample = Image.Resampling.LANCZOS if k > 1.0 else Image.Resampling.BOX
    pil = Image.fromarray(img, mode="RGB")
    pil = pil.resize((new_w, new_h), resample=resample)
    return np.asarray(pil, dtype=np.uint8)


def _safe_step(x: torch.Tensor, steps: int) -> torch.Tensor:
    if steps <= 0:
        return x
    y = x * float(steps + 1)
    y = y.to(torch.int32).to(torch.float32) / float(steps)
    return y.clamp(0.0, 1.0)


def teed_edge_strength(
    image_rgb: np.ndarray,
    cfg: PreconditionConfig,
    *,
    device: torch.device,
) -> np.ndarray:
    """Return a float32 edge strength map in [0, 1] (HxW), resized to the original image."""
    if cfg.teed_weights_path is None or str(cfg.teed_weights_path).strip() == "":
        raise ValueError("TEED requested but cfg.teed_weights_path is not set.")
    if image_rgb.ndim != 3 or image_rgb.shape[2] < 3:
        raise ValueError("image_rgb must be HxWx3")

    img = np.asarray(image_rgb[..., :3], dtype=np.float32)
    if img.max() <= 1.5:
        img = img * 255.0
    img_uint8 = np.clip(img, 0.0, 255.0).astype(np.uint8)

    orig_h, orig_w = img_uint8.shape[:2]
    resized = _resize_like_controlnet_aux(img_uint8, int(cfg.teed_detect_resolution))
    resized = resized.copy()
    h, w = resized.shape[:2]

    model = _load_teed_model(cfg.teed_weights_path, device=device)

    with torch.no_grad():
        x = torch.from_numpy(resized).to(device=device, dtype=torch.float32)
        x = x.permute(2, 0, 1).unsqueeze(0)  # NHWC -> NCHW
        outputs = model(x)
        up = [F.interpolate(o, size=(h, w), mode="bilinear", align_corners=False) for o in outputs]
        mean_logits = torch.stack(up, dim=0).mean(dim=0)
        prob = torch.sigmoid(mean_logits)
        prob = _safe_step(prob, int(cfg.teed_safe_steps))
        prob = F.interpolate(prob, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        out = prob[0, 0].detach().cpu().numpy().astype(np.float32, copy=False)
    return np.clip(out, 0.0, 1.0)

def teed_mask_from_strength(
    strength: np.ndarray,
    cfg: PreconditionConfig,
    *,
    threshold: float | None = None,
) -> np.ndarray:
    """Convert an edge strength map in [0,1] to a boolean edge mask."""
    mode = (cfg.teed_threshold_mode or "fixed").strip().lower()
    if mode == "fixed":
        thr = float(cfg.teed_threshold if threshold is None else threshold)
        edges = np.asarray(strength, dtype=np.float32) >= thr
    elif mode == "hysteresis":
        high = float(cfg.teed_threshold if threshold is None else threshold)
        low_ratio = float(cfg.teed_hysteresis_low_ratio)
        low = max(0.0, min(1.0, high * low_ratio))
        high = max(0.0, min(1.0, high))
        edges = filters.apply_hysteresis_threshold(np.asarray(strength, dtype=np.float32), low, high)
    else:
        raise ValueError(f"Unsupported teed_threshold_mode '{cfg.teed_threshold_mode}'. Choose from: fixed, hysteresis")

    if cfg.min_component_area > 0:
        edges = morphology.remove_small_objects(edges, cfg.min_component_area)
    if cfg.morph_open_radius > 0:
        edges = morphology.binary_opening(edges, morphology.disk(cfg.morph_open_radius))
    if cfg.morph_close_radius > 0:
        edges = morphology.binary_closing(edges, morphology.disk(cfg.morph_close_radius))

    return edges


def teed_edges(image_rgb: np.ndarray, cfg: PreconditionConfig, *, device: torch.device) -> np.ndarray:
    """Compute a boolean edge mask using TEED."""
    strength = teed_edge_strength(image_rgb, cfg, device=device)
    return teed_mask_from_strength(strength, cfg)


__all__ = ["TED", "teed_edge_strength", "teed_mask_from_strength", "teed_edges"]
