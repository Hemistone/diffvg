#!/usr/bin/env python3
"""Benchmark painterly_rendering.py across multiple backends and path counts.

Examples:
  python apps/bench_painterly_backends.py apps/imgs/flower.jpg
  python apps/bench_painterly_backends.py apps/imgs/flower.jpg --backends baseline,splat,bezier_gsplat --path-counts 128,512,1024 --num-iter 8 --repeats 3
  python apps/bench_painterly_backends.py apps/imgs/flower.jpg --precondition --config configs/precondition/teed_detail_quantile.toml --backends bezier_gsplat --path-counts 16
  python apps/bench_painterly_backends.py apps/imgs/flower.jpg --plotter-mode teed --plotter-report --plotter-cleanup --device cpu
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import statistics
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import diffvg
import torch

import pydiffvg
from single_utils import format_duration

PAINTERLY_APP = REPO_ROOT / "apps" / "painterly_rendering.py"
BENCH_ROOT = REPO_ROOT / "results" / "benchmarks" / "painterly_backends"
LOSS_RE = re.compile(r"loss=([0-9.eE+-]+)")
RUN_DIR_RE = re.compile(r"^\[config\] run_dir: (.+)$", re.MULTILINE)


@dataclass(frozen=True)
class RunSpec:
    backend: str
    paths: int
    repeat: int
    warmup: bool


@dataclass(frozen=True)
class RunResult:
    backend: str
    paths: int
    repeat: int
    warmup: bool
    success: bool
    elapsed_sec: float
    returncode: int
    log_stdout: str
    log_stderr: str
    run_dir: Optional[str]
    final_loss: Optional[float]
    stdout_tail: str
    stderr_tail: str
    final_svg: Optional[str] = None
    cleaned_svg: Optional[str] = None
    raw_strokes: Optional[int] = None
    raw_colors: Optional[int] = None
    raw_travel_ratio: Optional[float] = None
    cleaned_strokes: Optional[int] = None
    cleaned_colors: Optional[int] = None
    cleaned_travel_ratio: Optional[float] = None


def _parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_int_csv(raw: str) -> list[int]:
    values = []
    for item in _parse_csv_list(raw):
        try:
            values.append(int(item))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid integer list item '{item}'") from exc
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def _normalize_backend_name(name: str) -> str:
    key = (name or "").strip().lower()
    if key in ("baseline", "default"):
        return "baseline"
    if key == "splat":
        return "splat"
    if key in ("bezier_gsplat", "bezier-gsplat"):
        return "bezier_gsplat"
    raise ValueError(f"unsupported backend '{name}'")


def _normalize_plotter_mode(name: str | None) -> str | None:
    if name is None:
        return None
    key = name.strip().lower()
    if key == "":
        return None
    if key not in {"teed", "lineart", "flowline"}:
        raise ValueError(f"unsupported plotter mode '{name}'")
    return key


def _tail_lines(text: str, limit: int = 12) -> str:
    normalized = text.replace("\r", "\n")
    lines = [line for line in normalized.splitlines() if line.strip()]
    return "\n".join(lines[-limit:])


def _extract_final_loss(text: str) -> Optional[float]:
    normalized = text.replace("\r", "\n")
    matches = LOSS_RE.findall(normalized)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def _extract_run_dir(text: str) -> Optional[str]:
    match = RUN_DIR_RE.search(text.replace("\r", "\n"))
    if match is None:
        return None
    return match.group(1).strip()


def _detect_cuda_device_name() -> Optional[str]:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name",
                "--format=csv,noheader",
            ],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    return lines[0] if lines else None


def _git_output(args: list[str]) -> str:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return proc.stdout.strip()
    except Exception:
        return "<unavailable>"


def _gather_metadata(requested_device: str) -> dict:
    metadata = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "requested_device": requested_device,
        "diffvg_cuda_compiled": bool(getattr(diffvg, "is_cuda_compiled", lambda: False)()),
        "available_backends": list(pydiffvg.list_backends()),
        "git_branch": _git_output(["rev-parse", "--abbrev-ref", "HEAD"]),
        "git_commit": _git_output(["rev-parse", "HEAD"]),
    }
    if requested_device == "cuda":
        metadata["cuda_device"] = _detect_cuda_device_name()
    try:
        import gsplat  # type: ignore

        metadata["gsplat"] = getattr(gsplat, "__version__", "<unknown>")
    except Exception:
        metadata["gsplat"] = None
    return metadata


def _build_temp_config(contents: str, *, prefix: str) -> tuple[tempfile.TemporaryDirectory[str], Path]:
    temp_dir = tempfile.TemporaryDirectory(prefix="painterly-bench-", dir="/tmp")
    config_path = Path(temp_dir.name) / f"{prefix}.toml"
    config_path.write_text(contents, encoding="utf-8")
    return temp_dir, config_path


def _default_plotter_config(plotter_mode: str) -> Path:
    mapping = {
        "teed": REPO_ROOT / "configs" / "precondition" / "teed_detail_quantile.toml",
        "lineart": REPO_ROOT / "configs" / "precondition" / "lineart_quantile.toml",
        "flowline": REPO_ROOT / "configs" / "precondition" / "flowline.toml",
    }
    return mapping[plotter_mode]


def _resolve_palette(args: argparse.Namespace) -> Optional[str]:
    if args.palette:
        return args.palette
    if args.plotter_mode is not None:
        return "single_black_pen"
    return None


def _resolve_config(args: argparse.Namespace) -> tuple[Optional[tempfile.TemporaryDirectory[str]], Optional[Path]]:
    if args.config:
        config_path = (REPO_ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
        if not config_path.is_file():
            raise FileNotFoundError(f"config not found: {config_path}")
        return None, config_path
    if args.plotter_mode is not None:
        return None, _default_plotter_config(args.plotter_mode)
    if args.precondition:
        return None, None
    return _build_temp_config("precondition = false\n", prefix="no_precondition")


def _build_command(
    args: argparse.Namespace,
    target: Path,
    backend: str,
    paths: int,
    config_path: Optional[Path],
    palette: Optional[str],
) -> list[str]:
    cmd = [
        sys.executable,
        str(PAINTERLY_APP),
        str(target),
        "--backend",
        backend,
        "--num-paths",
        str(paths),
        "--num-iter",
        str(args.num_iter),
        "--save-every",
        "0",
        "--save-svg-every",
        "0",
        "--max-width",
        str(args.max_width),
        "--loss",
        args.loss,
    ]
    if config_path is not None:
        cmd.extend(["--config", str(config_path)])
    if palette:
        cmd.extend(["--palette", palette])
    if args.use_blob:
        cmd.append("--use-blob")
    if args.plotter_mode is not None:
        cmd.append("--precondition")
        if args.plotter_mode == "lineart":
            if args.plotter_lineart_mask_count is not None:
                cmd.extend(["--precond-lineart-mask-count", str(args.plotter_lineart_mask_count)])
            if args.plotter_lineart_mask_mode is not None:
                cmd.extend(["--precond-lineart-mask-mode", args.plotter_lineart_mask_mode])
    for extra in args.extra_arg:
        cmd.extend(shlex.split(extra))
    return cmd


def _run_once(
    spec: RunSpec,
    cmd: list[str],
    env: dict[str, str],
    logs_dir: Path,
    timeout_sec: int,
) -> RunResult:
    stem = f"{spec.backend}_paths{spec.paths}_{'warmup' if spec.warmup else f'repeat{spec.repeat}'}"
    stdout_path = logs_dir / f"{stem}.stdout.log"
    stderr_path = logs_dir / f"{stem}.stderr.log"
    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )
    elapsed = time.perf_counter() - start
    stdout_path.write_text(proc.stdout, encoding="utf-8", errors="replace")
    stderr_path.write_text(proc.stderr, encoding="utf-8", errors="replace")
    return RunResult(
        backend=spec.backend,
        paths=spec.paths,
        repeat=spec.repeat,
        warmup=spec.warmup,
        success=proc.returncode == 0,
        elapsed_sec=elapsed,
        returncode=proc.returncode,
        log_stdout=str(stdout_path),
        log_stderr=str(stderr_path),
        run_dir=_extract_run_dir(proc.stdout),
        final_loss=_extract_final_loss(proc.stdout),
        stdout_tail=_tail_lines(proc.stdout),
        stderr_tail=_tail_lines(proc.stderr),
    )


def _collect_plotter_metrics(
    result: RunResult,
    *,
    plotter_dir: Path,
    cleanup: bool,
) -> RunResult:
    if not result.success or not result.run_dir:
        return result
    final_svg = Path(result.run_dir) / "final.svg"
    if not final_svg.is_file():
        return result
    try:
        raw = pydiffvg.analyze_svg(final_svg)
    except Exception:
        return replace(result, final_svg=str(final_svg))

    cleaned_svg: Optional[Path] = None
    cleaned_metrics = None
    if cleanup:
        plotter_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(result.log_stdout).stem.replace(".stdout", "")
        cleaned_svg = plotter_dir / f"{stem}.cleaned.svg"
        try:
            cleanup_result = pydiffvg.cleanup_svg(final_svg, cleaned_svg)
            cleaned_metrics = cleanup_result.after_metrics
        except Exception:
            cleaned_svg = None
            cleaned_metrics = None

    return replace(
        result,
        final_svg=str(final_svg),
        cleaned_svg=str(cleaned_svg) if cleaned_svg is not None else None,
        raw_strokes=raw.stroke_count,
        raw_colors=raw.color_count,
        raw_travel_ratio=raw.travel_ratio,
        cleaned_strokes=cleaned_metrics.stroke_count if cleaned_metrics is not None else None,
        cleaned_colors=cleaned_metrics.color_count if cleaned_metrics is not None else None,
        cleaned_travel_ratio=cleaned_metrics.travel_ratio if cleaned_metrics is not None else None,
    )


def _write_results_csv(path: Path, results: Iterable[RunResult]) -> None:
    rows = [asdict(result) for result in results]
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _summarize(results: list[RunResult]) -> list[dict]:
    groups: dict[tuple[str, int], list[RunResult]] = {}
    for result in results:
        if result.warmup:
            continue
        groups.setdefault((result.backend, result.paths), []).append(result)

    summaries: list[dict] = []
    for (backend, paths), items in sorted(groups.items(), key=lambda item: (item[0][1], item[0][0])):
        ok = [item for item in items if item.success]
        row = {
            "backend": backend,
            "paths": paths,
            "repeats": len(items),
            "successes": len(ok),
            "failures": len(items) - len(ok),
            "mean_sec": None,
            "median_sec": None,
            "min_sec": None,
            "max_sec": None,
            "last_loss": ok[-1].final_loss if ok and ok[-1].final_loss is not None else None,
            "raw_travel_ratio": ok[-1].raw_travel_ratio if ok else None,
            "cleaned_travel_ratio": ok[-1].cleaned_travel_ratio if ok else None,
            "raw_strokes": ok[-1].raw_strokes if ok else None,
            "cleaned_strokes": ok[-1].cleaned_strokes if ok else None,
        }
        if ok:
            elapsed = [item.elapsed_sec for item in ok]
            row["mean_sec"] = statistics.mean(elapsed)
            row["median_sec"] = statistics.median(elapsed)
            row["min_sec"] = min(elapsed)
            row["max_sec"] = max(elapsed)
        summaries.append(row)
    return summaries


def _write_summary_markdown(path: Path, metadata: dict, args: argparse.Namespace, summaries: list[dict]) -> None:
    lines = [
        "# Painterly Backend Benchmark",
        "",
        f"- target: `{args.target}`",
        f"- backends: `{','.join(args.backends)}`",
        f"- path_counts: `{','.join(str(v) for v in args.path_counts)}`",
        f"- num_iter: `{args.num_iter}`",
        f"- repeats: `{args.repeats}`",
        f"- warmup: `{args.warmup}`",
        f"- precondition: `{args.precondition}`",
        f"- plotter_mode: `{args.plotter_mode}`",
        f"- palette: `{metadata.get('palette')}`",
        f"- plotter_report: `{args.plotter_report}`",
        f"- plotter_cleanup: `{args.plotter_cleanup}`",
        f"- torch: `{metadata.get('torch')}`",
        f"- cuda: `{metadata.get('torch_cuda')}`",
        f"- requested_device: `{metadata.get('requested_device')}`",
        f"- device: `{metadata.get('cuda_device') or '<unknown>' if metadata.get('requested_device') == 'cuda' else '<cpu>'}`",
        f"- gsplat: `{metadata.get('gsplat')}`",
        f"- git: `{metadata.get('git_branch')} @ {metadata.get('git_commit')}`",
        "",
        "| backend | paths | ok/repeats | mean_s | median_s | min_s | max_s | last_loss | raw_travel | cleaned_travel | raw_strokes | cleaned_strokes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        def _fmt(value: Optional[float]) -> str:
            return "-" if value is None else f"{value:.4f}"

        def _fmt_int(value: Optional[int]) -> str:
            return "-" if value is None else str(value)

        lines.append(
            f"| {row['backend']} | {row['paths']} | {row['successes']}/{row['repeats']} | "
            f"{_fmt(row['mean_sec'])} | {_fmt(row['median_sec'])} | {_fmt(row['min_sec'])} | "
            f"{_fmt(row['max_sec'])} | {_fmt(row['last_loss'])} | {_fmt(row['raw_travel_ratio'])} | "
            f"{_fmt(row['cleaned_travel_ratio'])} | {_fmt_int(row['raw_strokes'])} | {_fmt_int(row['cleaned_strokes'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_summary(summaries: list[dict]) -> None:
    if not summaries:
        print("[bench] no results")
        return
    print("[bench] summary")
    for row in summaries:
        if row["successes"] == 0:
            print(f"[bench] {row['backend']:>13} paths={row['paths']:>4} success=0/{row['repeats']}")
            continue
        extra = ""
        if row["raw_travel_ratio"] is not None:
            extra += f" raw_travel={row['raw_travel_ratio']:.3f}"
        if row["cleaned_travel_ratio"] is not None:
            extra += f" clean_travel={row['cleaned_travel_ratio']:.3f}"
        print(
            f"[bench] {row['backend']:>13} paths={row['paths']:>4} "
            f"median={row['median_sec']:.4f}s mean={row['mean_sec']:.4f}s "
            f"ok={row['successes']}/{row['repeats']}{extra}"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", help="target image path")
    parser.add_argument(
        "--backends",
        type=_parse_csv_list,
        default=list(pydiffvg.list_backends()),
        help="comma-separated backends to benchmark",
    )
    parser.add_argument(
        "--path-counts",
        type=_parse_int_csv,
        default=[128, 512, 1024],
        help="comma-separated path counts",
    )
    parser.add_argument("--num-iter", type=int, default=8, help="optimization iterations per run")
    parser.add_argument("--repeats", type=int, default=3, help="timed repeats per backend/path pair")
    parser.add_argument("--warmup", type=int, default=1, help="warmup runs per backend/path pair")
    parser.add_argument("--timeout-sec", type=int, default=3600, help="subprocess timeout")
    parser.add_argument("--max-width", type=float, default=2.0, help="forwarded to painterly_rendering.py")
    parser.add_argument("--loss", type=str, default="mse", help="forwarded to painterly_rendering.py")
    parser.add_argument("--palette", type=str, default=None, help="forwarded to painterly_rendering.py")
    parser.add_argument("--config", type=str, default=None, help="explicit painterly config file")
    parser.add_argument("--precondition", action="store_true", help="use painterly preconditioning config path instead of auto no-precondition config")
    parser.add_argument("--use-blob", action="store_true", help="forwarded to painterly_rendering.py")
    parser.add_argument("--device", type=str, default="cuda", help="DIFFVG_DEVICE for subprocesses")
    parser.add_argument(
        "--plotter-mode",
        type=str,
        default=None,
        choices=["teed", "lineart", "flowline"],
        help="plotter-style fixed-palette/fixed-width benchmark mode; implies --precondition and defaults palette to single_black_pen",
    )
    parser.add_argument(
        "--plotter-lineart-mask-count",
        type=int,
        default=None,
        help="forwarded only when --plotter-mode lineart is active",
    )
    parser.add_argument(
        "--plotter-lineart-mask-mode",
        type=str,
        default=None,
        choices=["auto", "fixed"],
        help="forwarded only when --plotter-mode lineart is active",
    )
    parser.add_argument(
        "--plotter-report",
        action="store_true",
        help="analyze final.svg for plotter metrics on successful runs",
    )
    parser.add_argument(
        "--plotter-cleanup",
        action="store_true",
        help="when --plotter-report is enabled, also run conservative cleanup and record cleaned metrics",
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="extra arg string forwarded to painterly_rendering.py; may be repeated",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.plotter_mode = _normalize_plotter_mode(args.plotter_mode)
    if args.plotter_mode is not None:
        args.precondition = True

    target = Path(args.target)
    if not target.is_absolute():
        target = (REPO_ROOT / target).resolve()
    if not target.is_file():
        raise FileNotFoundError(f"target image not found: {target}")

    backends = [_normalize_backend_name(name) for name in args.backends]
    unknown = [name for name in backends if name not in pydiffvg.list_backends()]
    if unknown:
        raise ValueError(f"unknown backends requested: {unknown}")
    args.backends = backends

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = BENCH_ROOT / timestamp
    logs_dir = report_dir / "logs"
    plotter_dir = report_dir / "plotter"
    report_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    temp_config_ctx, config_path = _resolve_config(args)
    palette = _resolve_palette(args)

    metadata = _gather_metadata(args.device)
    metadata["target"] = str(target)
    metadata["config"] = str(config_path) if config_path is not None else None
    metadata["palette"] = palette
    metadata["precondition"] = bool(args.precondition)
    metadata["plotter_mode"] = args.plotter_mode
    metadata["plotter_report"] = bool(args.plotter_report)
    metadata["plotter_cleanup"] = bool(args.plotter_cleanup)
    metadata["path_counts"] = args.path_counts
    metadata["backends"] = args.backends
    metadata["num_iter"] = args.num_iter
    metadata["repeats"] = args.repeats
    metadata["warmup"] = args.warmup
    (report_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["DIFFVG_DEVICE"] = args.device

    print(f"[bench] target={target}")
    print(f"[bench] report_dir={report_dir}")
    print(f"[bench] backends={','.join(args.backends)} paths={','.join(str(v) for v in args.path_counts)}")
    if config_path is not None:
        print(f"[bench] config={config_path}")
    if palette is not None:
        print(f"[bench] palette={palette}")
    if args.plotter_mode is not None:
        print(f"[bench] plotter_mode={args.plotter_mode}")

    results: list[RunResult] = []
    try:
        for paths in args.path_counts:
            for backend in args.backends:
                cmd = _build_command(args, target, backend, paths, config_path, palette)
                print(f"[bench] case backend={backend} paths={paths}")
                for warmup_idx in range(args.warmup):
                    spec = RunSpec(backend=backend, paths=paths, repeat=warmup_idx + 1, warmup=True)
                    result = _run_once(spec, cmd, env, logs_dir, args.timeout_sec)
                    results.append(result)
                    status = "ok" if result.success else f"fail(rc={result.returncode})"
                    print(f"[bench]   warmup {warmup_idx + 1}/{args.warmup}: {status} {format_duration(result.elapsed_sec)}")
                    if not result.success:
                        print(textwrap.indent(result.stdout_tail or result.stderr_tail, prefix="        "))
                        break
                for repeat_idx in range(args.repeats):
                    spec = RunSpec(backend=backend, paths=paths, repeat=repeat_idx + 1, warmup=False)
                    result = _run_once(spec, cmd, env, logs_dir, args.timeout_sec)
                    if args.plotter_report:
                        result = _collect_plotter_metrics(result, plotter_dir=plotter_dir, cleanup=args.plotter_cleanup)
                    results.append(result)
                    status = "ok" if result.success else f"fail(rc={result.returncode})"
                    loss_text = "" if result.final_loss is None else f" loss={result.final_loss:.4f}"
                    plotter_text = ""
                    if result.raw_travel_ratio is not None:
                        plotter_text += f" raw_travel={result.raw_travel_ratio:.3f}"
                    if result.cleaned_travel_ratio is not None:
                        plotter_text += f" clean_travel={result.cleaned_travel_ratio:.3f}"
                    print(
                        f"[bench]   repeat {repeat_idx + 1}/{args.repeats}: {status} "
                        f"{format_duration(result.elapsed_sec)}{loss_text}{plotter_text}"
                    )
                    if not result.success:
                        print(textwrap.indent(result.stdout_tail or result.stderr_tail, prefix="        "))
    finally:
        if temp_config_ctx is not None:
            temp_config_ctx.cleanup()

    _write_results_csv(report_dir / "results.csv", results)
    summaries = _summarize(results)
    _write_summary_markdown(report_dir / "summary.md", metadata, args, summaries)
    (report_dir / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    _print_summary(summaries)


if __name__ == "__main__":
    main()
