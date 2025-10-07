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

    run = create_run_context("single_gradient", num_iterations, video_fps=24)
    log_run_configuration(
        "single_gradient",
        {
            "device": "cuda" if use_gpu else "cpu",
            "canvas": f"{canvas_width}x{canvas_height}",
            "samples": f"{num_samples[0]}x{num_samples[1]}",
            "iterations": num_iterations,
            "lr": learning_rate,
        },
    )

    color = pydiffvg.LinearGradient(
        begin=torch.tensor([50.0, 50.0]),
        end=torch.tensor([200.0, 200.0]),
        offsets=torch.tensor([0.0, 1.0]),
        stop_colors=torch.tensor([[0.2, 0.5, 0.7, 1.0], [0.7, 0.2, 0.5, 1.0]]),
    )
    circle = pydiffvg.Circle(radius=torch.tensor(40.0), center=torch.tensor([128.0, 128.0]))
    shapes = [circle]
    circle_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([0]), fill_color=color)
    shape_groups = [circle_group]
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

    radius_n = torch.tensor(20.0 / 256.0, requires_grad=True)
    center_n = torch.tensor([108.0 / 256.0, 138.0 / 256.0], requires_grad=True)
    begin_n = torch.tensor([100.0 / 256.0, 100.0 / 256.0], requires_grad=True)
    end_n = torch.tensor([150.0 / 256.0, 150.0 / 256.0], requires_grad=True)
    stop_colors = torch.tensor(
        [[0.1, 0.9, 0.2, 1.0], [0.5, 0.3, 0.6, 1.0]], requires_grad=True
    )
    color.begin = begin_n * 256
    color.end = end_n * 256
    color.stop_colors = stop_colors
    circle.radius = radius_n * 256
    circle.center = center_n * 256
    circle_group.fill_color = color
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

    optimizer = torch.optim.Adam(
        [radius_n, center_n, begin_n, end_n, stop_colors], lr=learning_rate
    )

    t = -1
    try:
        for t in range(num_iterations):
            optimizer.zero_grad()
            color.begin = begin_n * 256
            color.end = end_n * 256
            color.stop_colors = stop_colors
            circle.radius = radius_n * 256
            circle.center = center_n * 256
            circle_group.fill_color = color
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

    color.begin = begin_n * 256
    color.end = end_n * 256
    color.stop_colors = stop_colors
    circle.radius = radius_n * 256
    circle.center = center_n * 256
    circle_group.fill_color = color
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
