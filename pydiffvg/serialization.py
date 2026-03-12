from __future__ import annotations

from typing import Optional

import torch

def serialize_scene(
    canvas_width,
    canvas_height,
    shapes,
    shape_groups,
    filter=None,
    output_type=None,
    use_prefiltering=False,
    eval_positions=torch.tensor([]),
    *,
    keep_on_device: bool = False,
    device: Optional[torch.device | str] = None,
):
    """Serialize a supported stroke-first scene for the current backend."""
    from .render_function import RenderFunction

    return RenderFunction.serialize_scene(
        canvas_width,
        canvas_height,
        shapes,
        shape_groups,
        filter=filter,
        output_type=output_type,
        use_prefiltering=use_prefiltering,
        eval_positions=eval_positions,
        keep_on_device=keep_on_device,
        device=device,
    )


__all__ = ["serialize_scene"]
