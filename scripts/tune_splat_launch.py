#!/usr/bin/env python3
"""Grid-search tuner for Bézier splat Triton launch parameters.

The script executes `apps/painterly_rendering.py` repeatedly under the splat
backend while sweeping environment-variable knobs (e.g. TILE/WARPS/STAGES).
For each workload (target image × paths × loss) it reports the
configuration that minimizes the mean per-iteration wall-clock time parsed
from the painterly log lines.

Example:
    # 1. Edit CONFIG below to set targets, sweeps, repeats, etc.
    # 2. Run the tuner:
    python scripts/tune_splat_launch.py

    # Preview the plan without running:
    python scripts/tune_splat_launch.py --dry-run

    # Override the JSON output path for a single run:
    python scripts/tune_splat_launch.py --out tuning.json

Use `--` to pass extra arguments straight to painterly_rendering.py, e.g.
`-- --max_width 3.0 --blob_mode`.

Because each run launches the full painterly optimization, expect the sweep
to take significant time. Run on the target GPU you are tuning for.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from itertools import product
from statistics import mean
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

PainterlyResult = Dict[str, object]

_DELTA_PATTERN = re.compile(r"Δt=(?P<seconds>[0-9]+(?:\.[0-9]+)?)s")


@dataclass
class TunerConfig:
    """Edit these defaults to control the sweep."""

    targets: Sequence[str]
    path_counts: Sequence[int]
    num_iter: int
    losses: Sequence[str]
    sweep: Mapping[str, Sequence[object]]
    repeat: int = 1
    timeout: float | None = None
    device: str | None = None
    impl: str = "triton"
    painterly_args: Sequence[str] = ()
    output_path: str | None = None


# Edit CONFIG to adjust what the tuner runs by default.
CONFIG = TunerConfig(
    targets=("apps/imgs/fallingwater.jpg",),
    path_counts=(256, 512),
    num_iter=32,
    losses=("lpips"),
    sweep={
        "DIFFVG_SPLAT_TILE": (16, 32, 64),
        "DIFFVG_SPLAT_WARPS": (4, 8),
        "DIFFVG_SPLAT_STAGES": (2, 3),
    },
    repeat=4,
    painterly_args=(),
)


@dataclass(frozen=True)
class NormalizedConfig:
    targets: Tuple[str, ...]
    path_counts: Tuple[int, ...]
    num_iter: int
    losses: Tuple[str, ...]
    sweep: Dict[str, Tuple[str, ...]]
    repeat: int
    timeout: float | None
    device: str | None
    impl: str
    painterly_args: Tuple[str, ...]
    output_path: str | None


def _ensure_tuple(value: Sequence[object] | object) -> Tuple[object, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(value)
    return (value,)


def _normalize_config(config: TunerConfig) -> NormalizedConfig:
    targets = tuple(str(v) for v in _ensure_tuple(config.targets))
    path_counts = tuple(int(v) for v in _ensure_tuple(config.path_counts))
    losses = tuple(str(v) for v in _ensure_tuple(config.losses))
    painterly_args = tuple(str(v) for v in _ensure_tuple(config.painterly_args) if str(v))

    sweep_normalized: Dict[str, Tuple[str, ...]] = {}
    for key, raw_values in config.sweep.items():
        sweep_normalized[key] = tuple(str(v) for v in _ensure_tuple(raw_values))

    return NormalizedConfig(
        targets=targets,
        path_counts=path_counts,
        num_iter=int(config.num_iter),
        losses=losses,
        sweep=sweep_normalized,
        repeat=int(config.repeat),
        timeout=config.timeout,
        device=config.device,
        impl=config.impl,
        painterly_args=painterly_args,
        output_path=config.output_path,
    )


@dataclass(frozen=True)
class Workload:
    """Key identifying a painterly workload configuration."""

    target: str
    num_paths: int
    num_iter: int
    loss: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "target": self.target,
            "num_paths": self.num_paths,
            "num_iter": self.num_iter,
            "loss": self.loss,
        }


@dataclass
class SweepResult:
    """Recorded metrics for a single env configuration under one workload."""

    workload: Workload
    env: Mapping[str, str]
    run_index: int
    mean_delta_t: float
    elapsed: float
    returncode: int
    stdout: str
    stderr: str

    def ok(self) -> bool:
        return self.returncode == 0


def _format_env(env: Mapping[str, str]) -> str:
    return " ".join(f"{k}={v}" for k, v in sorted(env.items()))


def _run_painterly(
    workload: Workload,
    painterly_args: Sequence[str],
    env_overrides: Mapping[str, str],
    repeat_index: int,
    timeout: float | None,
) -> SweepResult:
    cmd: List[str] = [
        sys.executable,
        "apps/painterly_rendering.py",
        workload.target,
        "--num_paths",
        str(workload.num_paths),
        "--num_iter",
        str(workload.num_iter),
        "--loss",
        workload.loss,
        *painterly_args,
    ]

    env: MutableMapping[str, str] = dict(os.environ)
    env.update(
        {
            "DIFFVG_BACKEND": "splat",
            "DIFFVG_SPLAT_IMPL": env_overrides.get("DIFFVG_SPLAT_IMPL", "triton"),
        }
    )
    env.setdefault("PYTHONWARNINGS", "ignore")
    for key, value in env_overrides.items():
        env[key] = value

    start = time.time()
    completed = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    elapsed = time.time() - start

    times = [float(match.group("seconds")) for match in _DELTA_PATTERN.finditer(completed.stdout)]

    calc_mean = completed.returncode == 0 and workload.num_iter > 0
    if calc_mean:
        per_iter = elapsed / float(workload.num_iter)
    else:
        per_iter = float("inf")

    result = SweepResult(
        workload=workload,
        env=dict(env_overrides),
        run_index=repeat_index,
        mean_delta_t=per_iter,
        elapsed=elapsed,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
    return result


def _summarize_workload(results: Sequence[SweepResult]) -> PainterlyResult:
    best = min(
        (res for res in results if res.ok()),
        key=lambda res: res.mean_delta_t,
        default=None,
    )
    summary: PainterlyResult = {
        "workload": results[0].workload.to_dict() if results else {},
        "trials": [
            {
                "env": dict(res.env),
                "repeat": res.run_index,
                "mean_delta_t": res.mean_delta_t,
                "elapsed": res.elapsed,
                "returncode": res.returncode,
                "stderr_head": " / ".join(
                    line.strip()
                    for line in res.stderr.strip().splitlines()[:2]
                    if line.strip()
                )
                if res.stderr.strip()
                else "",
            }
            for res in results
        ],
    }
    if best is not None:
        summary["best_env"] = dict(best.env)
        summary["best_mean_delta_t"] = best.mean_delta_t
    else:
        summary["best_env"] = None
        summary["best_mean_delta_t"] = None
    return summary


def _print_summary(summary: Mapping[str, object]) -> None:
    workload = summary.get("workload", {})
    best_env = summary.get("best_env")
    best_time = summary.get("best_mean_delta_t")
    print("-" * 80)
    print(
        f"Workload target={workload.get('target')} paths={workload.get('num_paths')} "
        f"iters={workload.get('num_iter')} loss={workload.get('loss')}"
    )
    if best_env and best_time is not None:
        print(f"  Best mean Δt: {best_time:.3f}s with env {_format_env(best_env)}")
    else:
        print("  No successful runs recorded.")
    for trial in summary.get("trials", []):
        env_str = _format_env(trial["env"])
        rc = trial["returncode"]
        mean_dt = trial["mean_delta_t"]
        if rc == 0:
            print(
                f"    repeat {trial['repeat']:02d}: mean Δt={mean_dt:.3f}s "
                f"(elapsed={trial['elapsed']:.1f}s) env={env_str}"
            )
        else:
            line = f"    repeat {trial['repeat']:02d}: FAILED rc={rc}"
            if trial.get("stderr_head"):
                line += f" stderr: {trial['stderr_head']}"
            line += f" env={env_str}"
            print(line)


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep DIFFVG_SPLAT_* launch parameters and report fastest combos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional path to write JSON results.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the sweep plan without executing painterly.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_argument_parser()
    args = parser.parse_args(argv)

    config = _normalize_config(CONFIG)
    sweep_keys = list(config.sweep.keys())
    sweep_values = list(config.sweep.values())

    workloads = [
        Workload(target=target, num_paths=paths, num_iter=config.num_iter, loss=loss)
        for target in config.targets
        for paths in config.path_counts
        for loss in config.losses
    ]

    plan: List[Tuple[Workload, Mapping[str, str]]] = []
    plans_by_workload: List[Tuple[Workload, List[Mapping[str, str]]]] = []
    for workload in workloads:
        env_list: List[Mapping[str, str]] = []
        for combo in product(*sweep_values):
            env_combo = {key: value for key, value in zip(sweep_keys, combo)}
            if config.device and "DIFFVG_DEVICE" not in env_combo:
                env_combo["DIFFVG_DEVICE"] = config.device
            if "DIFFVG_SPLAT_IMPL" not in env_combo and config.impl:
                env_combo["DIFFVG_SPLAT_IMPL"] = config.impl
            plan.append((workload, env_combo))
            env_list.append(env_combo)
        plans_by_workload.append((workload, env_list))

    repeat = config.repeat
    plan_count = len(plan)
    workload_count = len(plans_by_workload)
    total_runs = plan_count * repeat
    print(
        f"Prepared {plan_count} plans across {workload_count} workloads "
        f"with repeat x{repeat}: total {total_runs} runs"
    )
    if args.dry_run:
        width = len(str(plan_count))
        plan_idx = 1
        for workload_idx, (workload, env_list) in enumerate(plans_by_workload, start=1):
            header = (
                f"[Workload {workload_idx}/{workload_count}] "
                f"target={workload.target} paths={workload.num_paths} "
                f"iters={workload.num_iter} loss={workload.loss} "
                f"variants={len(env_list)}"
            )
            print(header)
            for env_combo in env_list:
                line = (
                    f"    [Plan {plan_idx:0{width}d}/{plan_count}] "
                    f"env={_format_env(env_combo)}"
                )
                print(line)
                plan_idx += 1
        return 0

    all_results: Dict[Workload, List[SweepResult]] = {workload: [] for workload in workloads}

    for workload, env_combo in plan:
        for repeat_idx in range(repeat):
            print(
                f"Running workload target={workload.target} paths={workload.num_paths} "
                f"iters={workload.num_iter} loss={workload.loss} repeat={repeat_idx+1}/{repeat} "
                f"env={_format_env(env_combo)}"
            )
            try:
                result = _run_painterly(
                    workload=workload,
                    painterly_args=tuple(config.painterly_args),
                    env_overrides=env_combo,
                    repeat_index=repeat_idx,
                    timeout=config.timeout,
                )
            except subprocess.TimeoutExpired as exc:
                print(f"  TIMEOUT after {exc.timeout}s for env={_format_env(env_combo)}", file=sys.stderr)
                continue

            if result.ok():
                print(
                    f"  -> ok; mean Δt={result.mean_delta_t:.3f}s "
                    f"(elapsed={result.elapsed:.1f}s)"
                )
            else:
                status = f"FAILED rc={result.returncode}"
                snippet = ""
                if result.stderr.strip():
                    lines = [line.strip() for line in result.stderr.strip().splitlines() if line.strip()]
                    snippet = " / ".join(lines[:2])
                if snippet:
                    status += f"; stderr: {snippet}"
                print(f"  -> {status} (elapsed={result.elapsed:.1f}s)")

            all_results[result.workload].append(result)

    summaries = [_summarize_workload(results) for results in all_results.values()]
    for summary in summaries:
        _print_summary(summary)

    output_path = args.out or config.output_path
    if output_path:
        payload = {
            "generated_at": time.time(),
            "cmd": " ".join(shlex.quote(arg) for arg in sys.argv),
            "summaries": summaries,
        }
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"Wrote results to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
