import pydiffvg
import torch
import skimage

from single_utils import create_run_context, log_run_configuration


def main() -> None:
    use_gpu = torch.cuda.is_available()
    pydiffvg.set_use_gpu(use_gpu)

    canvas_width, canvas_height = 256, 256
    num_samples = (2, 2)
    num_iterations = 200
    learning_rate = 1e-2

    run = create_run_context("single_open_curve", num_iterations, video_fps=24)
    log_run_configuration(
        "single_open_curve",
        {
            "device": "cuda" if use_gpu else "cpu",
            "canvas": f"{canvas_width}x{canvas_height}",
            "samples": f"{num_samples[0]}x{num_samples[1]}",
            "iterations": num_iterations,
            "lr": learning_rate,
        },
    )

    num_control_points = torch.tensor([2])
    points = torch.tensor(
        [[120.0, 30.0], [150.0, 60.0], [90.0, 198.0], [60.0, 218.0]]
    )
    path = pydiffvg.Path(
        num_control_points=num_control_points,
        points=points,
        is_closed=False,
        stroke_width=torch.tensor(5.0),
    )
    shapes = [path]
    path_group = pydiffvg.ShapeGroup(
        shape_ids=torch.tensor([0]),
        fill_color=None,
        stroke_color=torch.tensor([0.6, 0.3, 0.6, 0.8]),
    )
    shape_groups = [path_group]
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
            [100.0 / 256.0, 40.0 / 256.0],
            [155.0 / 256.0, 65.0 / 256.0],
            [100.0 / 256.0, 180.0 / 256.0],
            [65.0 / 256.0, 238.0 / 256.0],
        ],
        requires_grad=True,
    )
    stroke_color = torch.tensor([0.4, 0.7, 0.5, 0.5], requires_grad=True)
    stroke_width_n = torch.tensor(10.0 / 100.0, requires_grad=True)
    path.points = points_n * 256
    path.stroke_width = stroke_width_n * 100
    path_group.stroke_color = stroke_color
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

    optimizer = torch.optim.Adam([points_n, stroke_color, stroke_width_n], lr=learning_rate)

    t = -1
    try:
        for t in range(num_iterations):
            optimizer.zero_grad()
            path.points = points_n * 256
            path.stroke_width = stroke_width_n * 100
            path_group.stroke_color = stroke_color
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

    path.points = points_n * 256
    path.stroke_width = stroke_width_n * 100
    path_group.stroke_color = stroke_color
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
