#!/usr/bin/env python3
"""Renderer-only microbenchmark for stroke-first and legacy backends.

Measures scene compilation, forward-only render, backward-only autograd, and
one optimization step on synthetic open-stroke scenes. Each benchmark case runs
in its own subprocess so legacy backend crashes do not discard the entire run.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch


@dataclass(frozen=True)
class BenchRow:
    backend: str
    paths: int
    repeats: int
    warmup: int
    compile_ms: float
    forward_ms: float
    backward_ms: float
    step_ms: float
    width: int
    height: int
    segments: int
    samples_x: int
    samples_y: int
    device: str


@dataclass(frozen=True)
class BenchFailure:
    backend: str
    paths: int
    returncode: int
    stdout_log: str
    stderr_log: str


def _parse_csv_ints(raw: str) -> list[int]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def _parse_backends(raw: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one backend")
    for value in values:
        if value not in {"bezier_gsplat", "baseline", "splat"}:
            raise argparse.ArgumentTypeError(f"unsupported backend '{value}'")
    return values


def _sync_if_needed() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _build_random_scene(d, *, width: int, height: int, path_count: int, segments: int, device: torch.device):
    shapes = []
    shape_groups = []
    for _ in range(path_count):
        points = []
        x = random.random() * width
        y = random.random() * height
        points.append((x, y))
        for _ in range(segments):
            p1 = (min(max(x + random.uniform(-0.12, 0.12) * width, 0.0), width), min(max(y + random.uniform(-0.12, 0.12) * height, 0.0), height))
            p2 = (min(max(p1[0] + random.uniform(-0.12, 0.12) * width, 0.0), width), min(max(p1[1] + random.uniform(-0.12, 0.12) * height, 0.0), height))
            p3 = (min(max(p2[0] + random.uniform(-0.12, 0.12) * width, 0.0), width), min(max(p2[1] + random.uniform(-0.12, 0.12) * height, 0.0), height))
            points.extend([p1, p2, p3])
            x, y = p3
        path = d.Path(
            num_control_points=torch.full((segments,), 2, dtype=torch.int32, device=device),
            points=torch.tensor(points, dtype=torch.float32, device=device),
            stroke_width=torch.tensor(1.0, dtype=torch.float32, device=device),
            is_closed=False,
        )
        color = torch.tensor([
            0.15 + 0.7 * random.random(),
            0.15 + 0.7 * random.random(),
            0.15 + 0.7 * random.random(),
            1.0,
        ], dtype=torch.float32, device=device)
        group = d.ShapeGroup(
            shape_ids=torch.tensor([len(shapes)], dtype=torch.int32, device=device),
            fill_color=None,
            stroke_color=color,
        )
        shapes.append(path)
        shape_groups.append(group)
    return shapes, shape_groups


def _compile_scene(renderer, width: int, height: int, shapes, shape_groups, device: torch.device):
    _sync_if_needed()
    start = time.perf_counter()
    scene_args = renderer.serialize_scene(
        width,
        height,
        shapes,
        shape_groups,
        device=device,
        cache_key="bench",
        invalidate_cache=True,
    )
    _sync_if_needed()
    return scene_args, time.perf_counter() - start


def _forward_only(renderer, width: int, height: int, scene_args, samples_x: int, samples_y: int) -> torch.Tensor:
    return renderer.apply(width, height, samples_x, samples_y, 0, None, *scene_args)


def _time_forward(renderer, width: int, height: int, scene_args, samples_x: int, samples_y: int, repeats: int, warmup: int) -> float:
    for _ in range(max(warmup, 1)):
        _forward_only(renderer, width, height, scene_args, samples_x, samples_y)
    _sync_if_needed()
    start = time.perf_counter()
    for _ in range(repeats):
        _forward_only(renderer, width, height, scene_args, samples_x, samples_y)
    _sync_if_needed()
    return (time.perf_counter() - start) / repeats


def _make_trainable(shapes):
    params = []
    for shape in shapes:
        shape.points = shape.points.detach().clone().requires_grad_(True)
        params.append(shape.points)
    return params


def _time_backward(renderer, width: int, height: int, shapes, shape_groups, samples_x: int, samples_y: int, repeats: int, warmup: int, device: torch.device) -> float:
    if renderer.backend == "bezier_gsplat":
        scene_args = renderer.serialize_scene(
            width,
            height,
            shapes,
            shape_groups,
            device=device,
            cache_key=None,
            invalidate_cache=True,
        )
        compiled_scene = scene_args[0]
        compiled_scene.point_bank.requires_grad_(True)
        params = [compiled_scene.point_bank]
    else:
        params = _make_trainable(shapes)
        scene_args = renderer.serialize_scene(
            width,
            height,
            shapes,
            shape_groups,
            device=device,
            cache_key=None,
            invalidate_cache=True,
        )
    for _ in range(max(warmup, 1)):
        for param in params:
            if param.grad is not None:
                param.grad.zero_()
        if renderer.backend != "bezier_gsplat":
            scene_args = renderer.serialize_scene(
                width,
                height,
                shapes,
                shape_groups,
                device=device,
                cache_key=None,
                invalidate_cache=True,
            )
        img = renderer.apply(width, height, samples_x, samples_y, 0, None, *scene_args)
        img[..., :3].mean().backward()
    _sync_if_needed()
    start = time.perf_counter()
    for _ in range(repeats):
        for param in params:
            if param.grad is not None:
                param.grad.zero_()
        if renderer.backend != "bezier_gsplat":
            scene_args = renderer.serialize_scene(
                width,
                height,
                shapes,
                shape_groups,
                device=device,
                cache_key=None,
                invalidate_cache=True,
            )
        img = renderer.apply(width, height, samples_x, samples_y, 0, None, *scene_args)
        img[..., :3].mean().backward()
    _sync_if_needed()
    return (time.perf_counter() - start) / repeats


def _time_step(renderer, width: int, height: int, shapes, shape_groups, samples_x: int, samples_y: int, repeats: int, warmup: int, device: torch.device) -> float:
    if renderer.backend == "bezier_gsplat":
        scene_args = renderer.serialize_scene(
            width,
            height,
            shapes,
            shape_groups,
            device=device,
            cache_key=None,
            invalidate_cache=True,
        )
        compiled_scene = scene_args[0]
        compiled_scene.point_bank.requires_grad_(True)
        params = [compiled_scene.point_bank]
    else:
        params = _make_trainable(shapes)
        scene_args = renderer.serialize_scene(
            width,
            height,
            shapes,
            shape_groups,
            device=device,
            cache_key=None,
            invalidate_cache=True,
        )
    optim = torch.optim.Adam(params, lr=1e-1)
    for _ in range(max(warmup, 1)):
        optim.zero_grad(set_to_none=True)
        if renderer.backend != "bezier_gsplat":
            scene_args = renderer.serialize_scene(
                width,
                height,
                shapes,
                shape_groups,
                device=device,
                cache_key=None,
                invalidate_cache=True,
            )
        img = renderer.apply(width, height, samples_x, samples_y, 0, None, *scene_args)
        img[..., :3].mean().backward()
        optim.step()
    _sync_if_needed()
    start = time.perf_counter()
    for _ in range(repeats):
        optim.zero_grad(set_to_none=True)
        if renderer.backend != "bezier_gsplat":
            scene_args = renderer.serialize_scene(
                width,
                height,
                shapes,
                shape_groups,
                device=device,
                cache_key=None,
                invalidate_cache=True,
            )
        img = renderer.apply(width, height, samples_x, samples_y, 0, None, *scene_args)
        img[..., :3].mean().backward()
        optim.step()
    _sync_if_needed()
    return (time.perf_counter() - start) / repeats


def _run_worker(args: argparse.Namespace) -> BenchRow:
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["PYTHONPATH"] = str(REPO_ROOT)
    os.environ["DIFFVG_DEVICE"] = args.device
    if args.worker_backend in {"baseline", "splat"}:
        os.environ["DIFFVG_ENABLE_LEGACY"] = "1"

    import pydiffvg as d

    d.set_device(args.device)
    d.set_backend(args.worker_backend)
    device = d.get_device()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    shapes, shape_groups = _build_random_scene(
        d,
        width=args.width,
        height=args.height,
        path_count=args.worker_paths,
        segments=args.segments,
        device=device,
    )
    renderer = d.Renderer(backend=args.worker_backend)
    scene_args, compile_sec = _compile_scene(renderer, args.width, args.height, shapes, shape_groups, device)
    forward_sec = _time_forward(renderer, args.width, args.height, scene_args, args.samples_x, args.samples_y, args.repeats, args.warmup)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    backward_shapes, backward_groups = _build_random_scene(
        d,
        width=args.width,
        height=args.height,
        path_count=args.worker_paths,
        segments=args.segments,
        device=device,
    )
    backward_sec = _time_backward(renderer, args.width, args.height, backward_shapes, backward_groups, args.samples_x, args.samples_y, args.repeats, args.warmup, device)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    step_shapes, step_groups = _build_random_scene(
        d,
        width=args.width,
        height=args.height,
        path_count=args.worker_paths,
        segments=args.segments,
        device=device,
    )
    step_sec = _time_step(renderer, args.width, args.height, step_shapes, step_groups, args.samples_x, args.samples_y, args.repeats, args.warmup, device)

    return BenchRow(
        backend=args.worker_backend,
        paths=args.worker_paths,
        repeats=args.repeats,
        warmup=args.warmup,
        compile_ms=compile_sec * 1e3,
        forward_ms=forward_sec * 1e3,
        backward_ms=backward_sec * 1e3,
        step_ms=step_sec * 1e3,
        width=args.width,
        height=args.height,
        segments=args.segments,
        samples_x=args.samples_x,
        samples_y=args.samples_y,
        device=str(device),
    )


def _spawn_worker(args: argparse.Namespace, backend: str, path_count: int, report_dir: Path) -> BenchRow | BenchFailure:
    json_path = report_dir / f"{backend}_paths{path_count}.json"
    stdout_path = report_dir / f"{backend}_paths{path_count}.stdout.log"
    stderr_path = report_dir / f"{backend}_paths{path_count}.stderr.log"
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-json", str(json_path),
        "--worker-backend", backend,
        "--worker-paths", str(path_count),
        "--width", str(args.width),
        "--height", str(args.height),
        "--segments", str(args.segments),
        "--samples-x", str(args.samples_x),
        "--samples-y", str(args.samples_y),
        "--repeats", str(args.repeats),
        "--warmup", str(args.warmup),
        "--seed", str(args.seed),
        "--device", args.device,
    ]
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["DIFFVG_DEVICE"] = args.device
    if backend in {"baseline", "splat"}:
        env["DIFFVG_ENABLE_LEGACY"] = "1"
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, text=True, capture_output=True)
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0 or not json_path.exists():
        return BenchFailure(
            backend=backend,
            paths=path_count,
            returncode=proc.returncode,
            stdout_log=str(stdout_path),
            stderr_log=str(stderr_path),
        )
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    return BenchRow(**payload)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backends",
        type=_parse_backends,
        default=["bezier_gsplat"],
        help="comma-separated backends to benchmark (baseline/splat are legacy comparison paths)",
    )
    parser.add_argument("--path-counts", type=_parse_csv_ints, default=[128, 512, 1024])
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--segments", type=int, default=3)
    parser.add_argument("--samples-x", type=int, default=2)
    parser.add_argument("--samples-y", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--worker-json", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-backend", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-paths", type=int, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.worker_json is not None:
        row = _run_worker(args)
        args.worker_json.write_text(json.dumps(asdict(row), indent=2), encoding="utf-8")
        print(
            f"[micro] backend={row.backend:13s} paths={row.paths:4d} compile={row.compile_ms:8.2f}ms "
            f"forward={row.forward_ms:8.2f}ms backward={row.backward_ms:8.2f}ms step={row.step_ms:8.2f}ms"
        )
        return

    report_dir = args.report_dir or (REPO_ROOT / "results" / "benchmarks" / "renderer_micro" / datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    report_dir.mkdir(parents=True, exist_ok=True)

    rows: list[BenchRow] = []
    failures: list[BenchFailure] = []
    for backend in args.backends:
        for path_count in args.path_counts:
            result = _spawn_worker(args, backend, path_count, report_dir)
            if isinstance(result, BenchFailure):
                failures.append(result)
                print(f"[micro] backend={backend:13s} paths={path_count:4d} failed rc={result.returncode}")
            else:
                rows.append(result)
                print(
                    f"[micro] backend={result.backend:13s} paths={result.paths:4d} compile={result.compile_ms:8.2f}ms "
                    f"forward={result.forward_ms:8.2f}ms backward={result.backward_ms:8.2f}ms step={result.step_ms:8.2f}ms"
                )

    csv_path = report_dir / "results.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
            writer.writeheader()
            for row in rows:
                writer.writerow(asdict(row))

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "rows": [asdict(row) for row in rows],
        "failures": [asdict(item) for item in failures],
    }
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = ["# Renderer Microbenchmark", ""]
    if rows:
        device_values = sorted({row.device for row in rows})
        lines.append(f"device: `{', '.join(device_values)}`")
        lines.append("")
        for row in rows:
            lines.append(
                f"- `{row.backend}` paths={row.paths}: compile={row.compile_ms:.2f}ms, forward={row.forward_ms:.2f}ms, backward={row.backward_ms:.2f}ms, step={row.step_ms:.2f}ms"
            )
    if failures:
        if rows:
            lines.append("")
        lines.append("## Failures")
        lines.append("")
        for failure in failures:
            lines.append(
                f"- `{failure.backend}` paths={failure.paths}: rc={failure.returncode}, stdout=`{failure.stdout_log}`, stderr=`{failure.stderr_log}`"
            )
    (report_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
