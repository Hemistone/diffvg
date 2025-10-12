import pydiffvg
import torch
import skimage

from single_utils import create_run_context, log_run_configuration


def main() -> None:
    use_gpu = torch.cuda.is_available()
    pydiffvg.set_use_gpu(use_gpu)

    canvas_width, canvas_height = 510, 510
    num_samples = (1, 1)
    num_iterations = 100
    learning_rate = 1e-2

    run = create_run_context("single_path_sdf", num_iterations, video_fps=24)
    device_str = str(pydiffvg.get_device())
    log_run_configuration(
        "single_path_sdf",
        {
            "device": device_str,
            "canvas": f"{canvas_width}x{canvas_height}",
            "samples": f"{num_samples[0]}x{num_samples[1]}",
            "iterations": num_iterations,
            "lr": learning_rate,
            "output": "sdf",
        },
    )

    shapes = pydiffvg.from_svg_path(
        "M510,255c0-20.4-17.85-38.25-38.25-38.25H331.5L204,12.75h-51l63.75,204H76.5l-38.25-51H0L25.5,255L0,344.25h38.25l38.25-51h140.25l-63.75,204h51l127.5-204h140.25C492.15,293.25,510,275.4,510,255z"
    )
    path_group = pydiffvg.ShapeGroup(
        shape_ids=torch.tensor([0]),
        fill_color=torch.tensor([0.3, 0.6, 0.3, 1.0]),
    )
    shape_groups = [path_group]
    scene_args = pydiffvg.RenderFunction.serialize_scene(
        canvas_width,
        canvas_height,
        shapes,
        shape_groups,
        output_type=pydiffvg.OutputType.sdf,
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
    img = img / 510.0
    pydiffvg.imwrite(img.cpu(), str(run.results_dir / "target.png"), gamma=1.0)
    target = img.clone()

    noise = torch.FloatTensor(shapes[0].points.shape).uniform_(0.0, 1.0)
    points_n = (shapes[0].points.clone() + (noise * 60 - 30)) / 510.0
    points_n.requires_grad = True
    color = torch.tensor([0.3, 0.2, 0.5, 1.0], requires_grad=True)
    shapes[0].points = points_n * 510
    path_group.fill_color = color
    scene_args = pydiffvg.RenderFunction.serialize_scene(
        canvas_width,
        canvas_height,
        shapes,
        shape_groups,
        output_type=pydiffvg.OutputType.sdf,
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
    img = img / 510.0
    pydiffvg.imwrite(img.cpu(), str(run.results_dir / "init.png"), gamma=1.0)

    optimizer = torch.optim.Adam([points_n, color], lr=learning_rate)

    t = -1
    try:
        for t in range(num_iterations):
            optimizer.zero_grad()
            shapes[0].points = points_n * 510
            path_group.fill_color = color
            scene_args = pydiffvg.RenderFunction.serialize_scene(
                canvas_width,
                canvas_height,
                shapes,
                shape_groups,
                output_type=pydiffvg.OutputType.sdf,
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
            img = img / 510.0
            pydiffvg.imwrite(img.cpu(), str(run.iter_path(t)), gamma=1.0)
            loss = (img - target).pow(2).sum()
            loss_value = loss.item()
            loss.backward()
            optimizer.step()
            run.progress.log(t, loss=loss_value)
    except KeyboardInterrupt:
        run.progress.interrupt(t if t >= 0 else -1)
    finally:
        run.progress.close()

    shapes[0].points = points_n * 510
    path_group.fill_color = color
    scene_args = pydiffvg.RenderFunction.serialize_scene(
        canvas_width,
        canvas_height,
        shapes,
        shape_groups,
        output_type=pydiffvg.OutputType.sdf,
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
    img = img / 510.0
    pydiffvg.imwrite(img.cpu(), str(run.results_dir / "final.png"), gamma=1.0)

    run.make_video()


if __name__ == "__main__":
    main()
