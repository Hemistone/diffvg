import pydiffvg
import torch
import skimage
import numpy as np

from single_utils import create_run_context, log_run_configuration


def main() -> None:
    use_gpu = torch.cuda.is_available()
    pydiffvg.set_use_gpu(use_gpu)

    canvas_width, canvas_height = 256, 256
    num_samples = (2, 2)
    num_iterations = 100
    learning_rate = 1e-2

    run = create_run_context("single_polygon", num_iterations, video_fps=24)
    device_str = str(pydiffvg.get_device())
    log_run_configuration(
        "single_polygon",
        {
            "device": device_str,
            "canvas": f"{canvas_width}x{canvas_height}",
            "samples": f"{num_samples[0]}x{num_samples[1]}",
            "iterations": num_iterations,
            "lr": learning_rate,
        },
    )

    points = torch.tensor(
        [[120.0, 30.0], [60.0, 218.0], [210.0, 98.0], [30.0, 98.0], [180.0, 218.0]]
    )
    polygon = pydiffvg.Polygon(points=points, is_closed=True)
    shapes = [polygon]
    polygon_group = pydiffvg.ShapeGroup(
        shape_ids=torch.tensor([0]),
        fill_color=torch.tensor([0.3, 0.6, 0.3, 1.0]),
    )
    shape_groups = [polygon_group]
    scene_args = pydiffvg.RenderFunction.serialize_scene(
        canvas_width, canvas_height, shapes, shape_groups
    )

    render = pydiffvg.RenderFunction.apply
    img = render(
        canvas_width,
        canvas_height,
        num_samples[0],
        num_samples[1],
        0,
        None,
        *scene_args,
    )
    pydiffvg.imwrite(img.cpu(), str(run.results_dir / "target.png"), gamma=2.2)
    target = img.clone()

    points_n = torch.tensor(
        [
            [140.0 / 256.0, 20.0 / 256.0],
            [65.0 / 256.0, 228.0 / 256.0],
            [215.0 / 256.0, 100.0 / 256.0],
            [35.0 / 256.0, 90.0 / 256.0],
            [160.0 / 256.0, 208.0 / 256.0],
        ],
        requires_grad=True,
    )
    color = torch.tensor([0.3, 0.2, 0.5, 1.0], requires_grad=True)
    polygon.points = points_n * 256
    polygon_group.color = color
    scene_args = pydiffvg.RenderFunction.serialize_scene(
        canvas_width, canvas_height, shapes, shape_groups
    )
    img = render(
        canvas_width,
        canvas_height,
        num_samples[0],
        num_samples[1],
        1,
        None,
        *scene_args,
    )
    pydiffvg.imwrite(img.cpu(), str(run.results_dir / "init.png"), gamma=2.2)

    optimizer = torch.optim.Adam([points_n, color], lr=learning_rate)

    t = -1
    try:
        for t in range(num_iterations):
            optimizer.zero_grad()
            polygon.points = points_n * 256
            polygon_group.color = color
            scene_args = pydiffvg.RenderFunction.serialize_scene(
                canvas_width, canvas_height, shapes, shape_groups
            )
            img = render(
                canvas_width,
                canvas_height,
                num_samples[0],
                num_samples[1],
                t + 1,
                None,
                *scene_args,
            )
            pydiffvg.imwrite(img.cpu(), str(run.iter_path(t)), gamma=2.2)
            loss = (img - target).pow(2).sum()
            loss_value = loss.item()
            loss.backward()
            optimizer.step()
            run.progress.log(t, loss=loss_value)
    except KeyboardInterrupt:
        run.progress.interrupt(t if t >= 0 else -1)
    finally:
        run.progress.close()

    polygon.points = points_n * 256
    polygon_group.color = color
    scene_args = pydiffvg.RenderFunction.serialize_scene(
        canvas_width, canvas_height, shapes, shape_groups
    )
    img = render(
        canvas_width,
        canvas_height,
        num_samples[0],
        num_samples[1],
        num_iterations + 2,
        None,
        *scene_args,
    )
    pydiffvg.imwrite(img.cpu(), str(run.results_dir / "final.png"))

    run.make_video()


if __name__ == "__main__":
    main()
