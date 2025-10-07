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
    num_iterations = 200
    learning_rate = 1e-2

    run = create_run_context("single_circle_outline", num_iterations, video_fps=24)
    log_run_configuration(
        "single_circle_outline",
        {
            "device": "cuda" if use_gpu else "cpu",
            "canvas": f"{canvas_width}x{canvas_height}",
            "samples": f"{num_samples[0]}x{num_samples[1]}",
            "iterations": num_iterations,
            "lr": learning_rate,
        },
    )

    circle = pydiffvg.Circle(
        radius=torch.tensor(40.0),
        center=torch.tensor([128.0, 128.0]),
        stroke_width=torch.tensor(5.0),
    )
    shapes = [circle]
    circle_group = pydiffvg.ShapeGroup(
        shape_ids=torch.tensor([0]),
        fill_color=torch.tensor([0.3, 0.6, 0.3, 1.0]),
        stroke_color=torch.tensor([0.6, 0.3, 0.6, 0.8]),
    )
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
    fill_color = torch.tensor([0.3, 0.2, 0.8, 1.0], requires_grad=True)
    stroke_color = torch.tensor([0.4, 0.7, 0.5, 0.5], requires_grad=True)
    stroke_width_n = torch.tensor(10.0 / 100.0, requires_grad=True)
    circle.radius = radius_n * canvas_width
    circle.center = center_n * canvas_width
    circle.stroke_width = stroke_width_n * 100
    circle_group.fill_color = fill_color
    circle_group.stroke_color = stroke_color
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
        [radius_n, center_n, fill_color, stroke_color, stroke_width_n], lr=learning_rate
    )

    t = -1
    try:
        for t in range(num_iterations):
            optimizer.zero_grad()
            circle.radius = radius_n * canvas_width
            circle.center = center_n * canvas_width
            circle.stroke_width = stroke_width_n * 100
            circle_group.fill_color = fill_color
            circle_group.stroke_color = stroke_color
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

    circle.radius = radius_n * canvas_width
    circle.center = center_n * canvas_width
    circle.stroke_width = stroke_width_n * 100
    circle_group.fill_color = fill_color
    circle_group.stroke_color = stroke_color
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
