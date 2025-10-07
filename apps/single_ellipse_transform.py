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
    num_iterations = 150
    learning_rate = 1e-2

    run = create_run_context("single_ellipse_transform", num_iterations, video_fps=24)
    log_run_configuration(
        "single_ellipse_transform",
        {
            "device": "cuda" if use_gpu else "cpu",
            "canvas": f"{canvas_width}x{canvas_height}",
            "samples": f"{num_samples[0]}x{num_samples[1]}",
            "iterations": num_iterations,
            "lr": learning_rate,
        },
    )

    ellipse = pydiffvg.Ellipse(
        radius=torch.tensor([60.0, 30.0]),
        center=torch.tensor([128.0, 128.0]),
    )
    shapes = [ellipse]
    ellipse_group = pydiffvg.ShapeGroup(
        shape_ids=torch.tensor([0]),
        fill_color=torch.tensor([0.3, 0.6, 0.3, 1.0]),
        shape_to_canvas=torch.eye(3, 3),
    )
    shape_groups = [ellipse_group]
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

    color = torch.tensor([0.3, 0.2, 0.8, 1.0], requires_grad=True)
    affine = torch.zeros(2, 3)
    affine[0, 0] = 1.3
    affine[0, 1] = 0.2
    affine[0, 2] = 0.1
    affine[1, 0] = 0.2
    affine[1, 1] = 0.6
    affine[1, 2] = 0.3
    affine.requires_grad = True
    shape_to_canvas = torch.cat((affine, torch.tensor([[0.0, 0.0, 1.0]])), axis=0)
    ellipse_group.fill_color = color
    ellipse_group.shape_to_canvas = shape_to_canvas
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

    optimizer = torch.optim.Adam([color, affine], lr=learning_rate)

    t = -1
    try:
        for t in range(num_iterations):
            optimizer.zero_grad()
            ellipse_group.fill_color = color
            ellipse_group.shape_to_canvas = torch.cat(
                (affine, torch.tensor([[0.0, 0.0, 1.0]])), axis=0
            )
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

    ellipse_group.fill_color = color
    ellipse_group.shape_to_canvas = torch.cat(
        (affine, torch.tensor([[0.0, 0.0, 1.0]])), axis=0
    )
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
