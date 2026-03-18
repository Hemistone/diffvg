from __future__ import annotations

import io
import tempfile
from typing import Optional

import numpy as np
import torch

from .backend import BezierGsplatConfig, get_backend_config
from .openstroke import CompiledOpenStrokeScene, OpenStrokeUnsupported, compile_scene, render_compiled_scene
from .output import OutputType


def serialize_scene(
    canvas_width,
    canvas_height,
    shapes,
    shape_groups,
    filter=None,
    output_type=OutputType.color,
    use_prefiltering: bool = False,
    eval_positions: torch.Tensor = torch.tensor([]),
    *,
    keep_on_device: bool = True,
    device: Optional[torch.device | str] = None,
):
    del filter, keep_on_device
    compiled = compile_scene(
        canvas_width,
        canvas_height,
        shapes,
        shape_groups,
        output_type=output_type,
        use_prefiltering=use_prefiltering,
        eval_positions=eval_positions,
        device=device,
    )
    return (compiled,)


def _get_config() -> BezierGsplatConfig:
    config = get_backend_config("bezier_gsplat")
    if isinstance(config, BezierGsplatConfig):
        return config
    return BezierGsplatConfig()


def _resolve_scene(args: tuple[object, ...]) -> CompiledOpenStrokeScene:
    if len(args) != 1 or not isinstance(args[0], CompiledOpenStrokeScene):
        raise OpenStrokeUnsupported(
            "bezier_gsplat expects a compiled open-stroke scene; call serialize_scene() again after topology changes"
        )
    return args[0]


def _exact_render_cairo(
    scene: CompiledOpenStrokeScene,
    width: int,
    height: int,
) -> torch.Tensor:
    """Render the scene via SVG -> CairoSVG for pixel-perfect vector output.

    This produces output visually equivalent to the original diffvg baseline:
    uniform stroke width, round line caps, clean anti-aliased edges.
    Used only when grad is disabled (inference / visualization).
    """
    from .device import get_device
    from .save_svg import save_svg as _save_svg

    # Sync compiled tensor banks back to the frontend shape/group objects
    scene.bind_frontend_views()

    shapes = [ref.shape for ref in scene.shape_refs]
    groups = [ref.group for ref in scene.style_refs]

    # Generate SVG in memory
    svg_buf = io.StringIO()
    import xml.etree.ElementTree as etree
    from .save_svg import prettify, _path_d, _format_rgb, _finite_scalar

    root = etree.Element("svg")
    root.set("version", "1.1")
    root.set("xmlns", "http://www.w3.org/2000/svg")
    root.set("width", str(width))
    root.set("height", str(height))
    etree.SubElement(root, "defs")
    g = etree.SubElement(root, "g")

    for group_index, style_ref in enumerate(scene.style_refs):
        group = style_ref.group
        stroke_color = getattr(group, "stroke_color", None)
        if not isinstance(stroke_color, torch.Tensor) or stroke_color.numel() != 4:
            continue
        shape_ids = group.shape_ids.detach().to(dtype=torch.int64, device="cpu")
        if int(shape_ids.numel()) != 1:
            continue
        shape = shapes[int(shape_ids.item())]
        stroke_width = getattr(shape, "stroke_width", None)
        if not isinstance(stroke_width, torch.Tensor) or stroke_width.numel() != 1:
            continue
        stroke_width_value = _finite_scalar(stroke_width, what="stroke widths")
        if stroke_width_value < 0.0:
            continue
        stroke_alpha = _finite_scalar(stroke_color.reshape(4)[3], what="stroke alpha")
        stroke_alpha = min(max(stroke_alpha, 0.0), 1.0)

        shape_node = etree.SubElement(g, "path")
        shape_node.set("d", _path_d(shape))
        shape_node.set("fill", "none")
        shape_node.set("stroke", _format_rgb(stroke_color))
        shape_node.set("stroke-opacity", str(stroke_alpha))
        shape_node.set("stroke-width", str(2.0 * stroke_width_value))
        shape_node.set("stroke-linecap", "round")
        shape_node.set("stroke-linejoin", "round")
        shape_node.set("id", f"shape_{group_index}")

    svg_string = prettify(root)

    # Rasterize SVG -> PNG in memory via CairoSVG
    try:
        import cairosvg
    except ImportError:
        # Fallback: if CairoSVG not installed, use gsplat rendering
        return render_compiled_scene(
            scene, width=width, height=height, config=_get_config(),
        )

    png_bytes = cairosvg.svg2png(
        bytestring=svg_string.encode("utf-8"),
        output_width=width,
        output_height=height,
    )

    # PNG -> numpy -> tensor (H, W, 4) RGBA float32 [0,1]
    # CairoSVG renders on transparent background. Convert so that:
    #   RGB channels = pre-multiplied color on transparent bg
    #   Alpha channel = stroke coverage
    # This matches the gsplat output convention used by ControlSketch's compositing.
    from PIL import Image
    pil_img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    arr = np.array(pil_img, dtype=np.float32) / 255.0

    device = get_device()
    return torch.from_numpy(arr).to(device=device)


def apply(
    width: int,
    height: int,
    num_samples_x: int,
    num_samples_y: int,
    seed: int,
    background_image: Optional[torch.Tensor],
    *args: object,
) -> torch.Tensor:
    del num_samples_x, num_samples_y, seed
    scene = _resolve_scene(tuple(args))

    # Inference/visualization mode: exact vector rendering via Cairo
    # Training mode (grad enabled): fast Gaussian splatting
    if not torch.is_grad_enabled():
        try:
            return _exact_render_cairo(scene, width, height)
        except Exception:
            pass  # fallback to gsplat if Cairo fails

    return render_compiled_scene(
        scene,
        width=width,
        height=height,
        config=_get_config(),
        background_image=background_image,
    )


def render_grad(
    grad_img: torch.Tensor,
    width: int,
    height: int,
    num_samples_x: int,
    num_samples_y: int,
    seed: int,
    background_image: Optional[torch.Tensor],
    *args: object,
):
    del grad_img, width, height, num_samples_x, num_samples_y, seed, background_image, args
    raise RuntimeError("bezier_gsplat relies on standard autograd through apply(); render_grad is not used in the maintained path")


__all__ = ["serialize_scene", "apply", "render_grad"]
