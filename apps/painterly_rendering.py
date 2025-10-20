"""
Scream: python painterly_rendering.py imgs/scream.jpg --num_paths 2048 --max_width 4.0
Fallingwater: python painterly_rendering.py imgs/fallingwater.jpg --num_paths 2048 --max_width 4.0
Fallingwater: python painterly_rendering.py imgs/fallingwater.jpg --num_paths 2048 --max_width 4.0 --use_lpips_loss
Baboon: python painterly_rendering.py imgs/baboon.png --num_paths 1024 --max_width 4.0 --num_iter 250
Baboon Lpips: python painterly_rendering.py imgs/baboon.png --num_paths 1024 --max_width 4.0 --num_iter 500 --use_lpips_loss
Kitty: python painterly_rendering.py imgs/kitty.jpg --num_paths 1024 --use_blob
"""
import argparse
import math
import random
from pathlib import Path

import pydiffvg
import numpy as np
import skimage
import skimage.io
import torch

from single_utils import create_run_context, log_run_configuration


def _rgba_over_white(img_rgba: torch.Tensor) -> torch.Tensor:
    """Composite RGBA to RGB over white.

    Handles premultiplied vs straight-alpha differences between backends:
    - baseline returns straight RGB (needs multiply by A)
    - splat returns premultiplied RGB (do NOT multiply by A again)
    """
    backend = pydiffvg.get_backend()
    a = img_rgba[:, :, 3:4].clamp(0.0, 1.0)
    rgb = img_rgba[:, :, :3]
    if backend == "splat":
        premul = rgb  # RGB is already premultiplied in splat backend
    else:
        premul = rgb * a  # baseline provides straight (non-premultiplied) RGB
    return (premul + (1.0 - a)).clamp(0.0, 1.0)


gamma = 1.0


def main(args):
    use_gpu = torch.cuda.is_available()
    pydiffvg.set_use_gpu(use_gpu)

    # Initialize LPIPS loss (prefer PIQ if requested and available)
    perception_loss = None
    if args.use_lpips_loss:
        device = pydiffvg.get_device()
        try:
            import piq  # type: ignore

            class _LPIPSAdapter(torch.nn.Module):
                """Wrap PIQ's LPIPS to accept inputs in [0, 1].

                PIQ's LPIPS expects inputs normalized to [-1, 1]. This adapter
                keeps the rest of the pipeline unchanged by converting ranges.
                """

                def __init__(self):
                    super().__init__()
                    # Use default backbone (VGG) and mean reduction
                    self.loss = piq.LPIPS(reduction="mean")

                def forward(self, x, y):
                    x_n = x * 2.0 - 1.0
                    y_n = y * 2.0 - 1.0
                    return self.loss(x_n, y_n)

            perception_loss = _LPIPSAdapter().to(device)
        except Exception as e:
            raise RuntimeError(
                "--use_lpips_loss requested but 'piq' is not available. Install with 'pip install piq'."
            ) from e

    target_path = Path(args.target)
    # Robust image load: prefer imageio or PIL; fallback to skimage, then sanitize dtype
    img_np: np.ndarray
    try:
        import imageio.v3 as iio  # type: ignore
        img_np = iio.imread(str(target_path))
    except Exception:
        try:
            from PIL import Image  # type: ignore

            with Image.open(str(target_path)) as _im:
                if _im.mode not in ("RGB", "RGBA"):
                    # Convert paletted/LA/others to RGB
                    _im = _im.convert("RGB")
                img_np = np.array(_im)
        except Exception:
            img_np = skimage.io.imread(str(target_path))
    # Sanitize dtype/channels
    if img_np.dtype == object:
        # Some backends can return object arrays for odd encodings; coerce via PIL
        try:
            from PIL import Image  # type: ignore

            with Image.open(str(target_path)) as _im:
                if _im.mode not in ("RGB", "RGBA"):
                    _im = _im.convert("RGB")
                img_np = np.array(_im)
        except Exception:
            img_np = np.array(img_np, dtype=np.uint8)
    if img_np.ndim == 2:
        img_np = np.repeat(img_np[:, :, None], 3, axis=2)
    if img_np.shape[-1] == 4:
        img_np = img_np[:, :, :3]
    if img_np.dtype != np.uint8:
        img_np = np.clip(img_np, 0, 255).astype(np.uint8, copy=False)
    target = torch.from_numpy(img_np).to(torch.float32) / 255.0
    target = target.pow(gamma)
    target = target.to(pydiffvg.get_device())
    target = target.unsqueeze(0)
    target = target.permute(0, 3, 1, 2)  # NHWC -> NCHW
    canvas_width, canvas_height = target.shape[3], target.shape[2]
    num_paths = args.num_paths
    max_width = args.max_width

    def _sanitize_component(value: str) -> str:
        sanitized = [ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value]
        result = "".join(sanitized).strip("_")
        return result or "target"

    run_label = f"{_sanitize_component(target_path.stem)}_paths{num_paths}_"
    run_label += "blob" if args.use_blob else "strokes"

    run = create_run_context(
        run_label,
        args.num_iter,
        video_fps=24,
        video_bitrate="20M",
        results_root=Path("results") / "painterly_rendering",
    )
    device_str = str(pydiffvg.get_device())
    log_run_configuration(
        "painterly_rendering",
        {
            "device": device_str,
            "target": str(target_path),
            "canvas": f"{canvas_width}x{canvas_height}",
            "paths": num_paths,
            "iterations": args.num_iter,
            "max_width": max_width,
            "lpips": args.use_lpips_loss,
            "blob_mode": args.use_blob,
            "run_dir": run.results_dir,
        },
    )

    random.seed(1234)
    torch.manual_seed(1234)

    shapes = []
    shape_groups = []
    if args.use_blob:
        for i in range(num_paths):
            num_segments = random.randint(3, 5)
            num_control_points = torch.zeros(num_segments, dtype = torch.int32) + 2
            points = []
            p0 = (random.random(), random.random())
            points.append(p0)
            for j in range(num_segments):
                radius = 0.05
                p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
                p2 = (p1[0] + radius * (random.random() - 0.5), p1[1] + radius * (random.random() - 0.5))
                p3 = (p2[0] + radius * (random.random() - 0.5), p2[1] + radius * (random.random() - 0.5))
                points.append(p1)
                points.append(p2)
                if j < num_segments - 1:
                    points.append(p3)
                    p0 = p3
            points = torch.tensor(points)
            points[:, 0] *= canvas_width
            points[:, 1] *= canvas_height
            path = pydiffvg.Path(num_control_points = num_control_points,
                                 points = points,
                                 stroke_width = torch.tensor(1.0),
                                 is_closed = True)
            shapes.append(path)
            path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(shapes) - 1]),
                                             fill_color = torch.tensor([random.random(),
                                                                        random.random(),
                                                                        random.random(),
                                                                        random.random()]))
            shape_groups.append(path_group)
    else:
        for i in range(num_paths):
            num_segments = random.randint(1, 3)
            num_control_points = torch.zeros(num_segments, dtype = torch.int32) + 2
            points = []
            p0 = (random.random(), random.random())
            points.append(p0)
            for j in range(num_segments):
                radius = 0.05
                p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
                p2 = (p1[0] + radius * (random.random() - 0.5), p1[1] + radius * (random.random() - 0.5))
                p3 = (p2[0] + radius * (random.random() - 0.5), p2[1] + radius * (random.random() - 0.5))
                points.append(p1)
                points.append(p2)
                points.append(p3)
                p0 = p3
            points = torch.tensor(points)
            points[:, 0] *= canvas_width
            points[:, 1] *= canvas_height
            #points = torch.rand(3 * num_segments + 1, 2) * min(canvas_width, canvas_height)
            path = pydiffvg.Path(num_control_points = num_control_points,
                                 points = points,
                                 stroke_width = torch.tensor(1.0),
                                 is_closed = False)
            shapes.append(path)
            path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(shapes) - 1]),
                                             fill_color = None,
                                             stroke_color = torch.tensor([random.random(),
                                                                          random.random(),
                                                                          random.random(),
                                                                          random.random()]))
            shape_groups.append(path_group)
    
    scene_args = pydiffvg.RenderFunction.serialize_scene(
        canvas_width, canvas_height, shapes, shape_groups
    )

    render = pydiffvg.RenderFunction.apply
    img = render(
        canvas_width,
        canvas_height,
        2,
        2,
        0,
        None,
        *scene_args,
    )
    # Save raw RGBA for faithful visualization (viewer composites alpha)
    pydiffvg.imwrite(img.cpu(), str(run.results_dir / "init.png"), gamma=gamma)

    points_vars = []
    stroke_width_vars = []
    color_vars = []
    for path in shapes:
        path.points.requires_grad = True
        points_vars.append(path.points)
    if not args.use_blob:
        for path in shapes:
            path.stroke_width.requires_grad = True
            stroke_width_vars.append(path.stroke_width)
    if args.use_blob:
        for group in shape_groups:
            group.fill_color.requires_grad = True
            color_vars.append(group.fill_color)
    else:
        for group in shape_groups:
            group.stroke_color.requires_grad = True
            color_vars.append(group.stroke_color)
    
    # Optimize
    points_optim = torch.optim.Adam(points_vars, lr=1.0)
    width_optim = torch.optim.Adam(stroke_width_vars, lr=0.1) if stroke_width_vars else None
    color_optim = torch.optim.Adam(color_vars, lr=0.01)

    progress = run.progress
    t = -1
    try:
        for t in range(args.num_iter):
            points_optim.zero_grad()
            if width_optim is not None:
                width_optim.zero_grad()
            color_optim.zero_grad()
            scene_args = pydiffvg.RenderFunction.serialize_scene(
                canvas_width, canvas_height, shapes, shape_groups
            )
            img = render(
                canvas_width,
                canvas_height,
                2,
                2,
                t,
                None,
                *scene_args,
            )
            # Save raw RGBA for visualization
            pydiffvg.imwrite(img.cpu(), str(run.iter_path(t)), gamma=gamma)
            # Compose to RGB (over white) for loss only
            img_rgb = _rgba_over_white(img)
            img = img_rgb
            img = img.unsqueeze(0)
            img = img.permute(0, 3, 1, 2)
            if args.use_lpips_loss:
                loss = perception_loss(img, target) + (img.mean() - target.mean()).pow(2)
            else:
                loss = (img - target).pow(2).mean()

            loss.backward()

            points_optim.step()
            if width_optim is not None:
                width_optim.step()
            color_optim.step()
            if width_optim is not None:
                for path in shapes:
                    path.stroke_width.data.clamp_(1.0, max_width)
            if args.use_blob:
                for group in shape_groups:
                    group.fill_color.data.clamp_(0.0, 1.0)
            else:
                for group in shape_groups:
                    group.stroke_color.data.clamp_(0.0, 1.0)

            if t % 10 == 0 or t == args.num_iter - 1:
                pydiffvg.save_svg(
                    str(run.iter_dir / f"iter_{t:04d}.svg"),
                    canvas_width,
                    canvas_height,
                    shapes,
                    shape_groups,
                )

            progress.log(t, loss=loss.item())
    except KeyboardInterrupt:
        progress.interrupt(t)
    finally:
        progress.close()

    # Render the final result.
    scene_args = pydiffvg.RenderFunction.serialize_scene(
        canvas_width, canvas_height, shapes, shape_groups
    )
    img = render(
        target.shape[1],
        target.shape[0],
        2,
        2,
        0,
        None,
        *scene_args,
    )
    pydiffvg.imwrite(img.cpu(), str(run.results_dir / "final.png"), gamma=gamma)

    run.make_video()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("target", help="target image path")
    parser.add_argument("--num_paths", type=int, default=512)
    parser.add_argument("--max_width", type=float, default=2.0)
    parser.add_argument("--use_lpips_loss", dest='use_lpips_loss', action='store_true')
    parser.add_argument("--num_iter", type=int, default=500)
    parser.add_argument("--use_blob", dest='use_blob', action='store_true')
    args = parser.parse_args()
    main(args)
