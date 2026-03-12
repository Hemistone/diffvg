from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Union

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pydiffvg
from pydiffvg.backends.registry import get_api

Number = Union[int, float]


def format_duration(seconds: float) -> str:
    """Return a short human readable duration string."""
    seconds = max(seconds, 0.0)
    if seconds >= 3600:
        hours = seconds / 3600.0
        return f"{hours:.1f}h"
    if seconds >= 60:
        minutes = int(seconds // 60)
        rem = seconds - minutes * 60
        return f"{minutes}m{rem:04.1f}s"
    if seconds >= 1:
        return f"{seconds:.2f}s"
    return f"{seconds * 1000:.1f}ms"


def _format_metric(value: Union[Number, str]) -> str:
    if isinstance(value, (int, float)):
        abs_value = abs(value)
        if 0 < abs_value < 1e-3 or abs_value >= 1e3:
            return f"{value:.3e}"
        return f"{value:.4f}"
    return str(value)


class ProgressLogger:
    """Render concise progress lines with per-iteration timings and ETA."""

    def __init__(self, total_iters: int, label: str):
        self.total_iters = total_iters
        self.label = label
        self._start_time = time.perf_counter()
        self._last_time = self._start_time
        self._printed_chars = 0
        self._completed = 0
        self._finished = False

    def log(self, iteration: int, *, loss: Optional[float] = None, metrics: Optional[Dict[str, Union[Number, str]]] = None) -> None:
        now = time.perf_counter()
        iter_duration = now - self._last_time if self._completed else now - self._start_time
        self._completed = iteration + 1
        elapsed = now - self._start_time
        avg = elapsed / max(self._completed, 1)
        remaining = max(self.total_iters - self._completed, 0)
        eta = remaining * avg

        parts = [f"[{self.label} {self._completed:>3}/{self.total_iters:<3}]"]
        if loss is not None:
            parts.append(f"loss={_format_metric(loss)}")
        if metrics:
            for key, value in metrics.items():
                parts.append(f"{key}={_format_metric(value)}")
        parts.append(f"Δt={format_duration(iter_duration)}")
        parts.append(f"elapsed={format_duration(elapsed)}")
        if remaining:
            parts.append(f"ETA={format_duration(eta)}")

        line = " ".join(parts)
        self._render(line)
        self._last_time = now
        if self._completed == self.total_iters and not self._finished:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self._printed_chars = 0
            self._finished = True

    def interrupt(self, iteration: int) -> None:
        if not self._finished:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self._printed_chars = 0
            self._finished = True
        elapsed = time.perf_counter() - self._start_time
        if iteration < 0:
            iteration = 0
        print(
            f"[progress] interrupted at iteration {iteration + 1}/{self.total_iters} "
            f"after {format_duration(elapsed)}"
        )

    def close(self) -> None:
        if not self._finished:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self._printed_chars = 0
            self._finished = True

    def _render(self, line: str) -> None:
        trace_env = os.environ.get("DIFFVG_SPLAT_TRACE", "").strip().lower()
        trace_enabled = bool(trace_env) and trace_env not in ("0", "false", "no", "off")
        if trace_enabled:
            sys.stdout.write(line + "\n")
            sys.stdout.flush()
            self._printed_chars = 0
        else:
            padding = max(self._printed_chars - len(line), 0)
            sys.stdout.write("\r" + line + (" " * padding))
            sys.stdout.flush()
            self._printed_chars = len(line)


@dataclass
class RunContext:
    task_name: str
    total_iters: int
    results_dir: Path
    iter_dir: Path
    video_fps: int
    video_bitrate: str
    progress: ProgressLogger
    frame_prefix: str = "iter_"
    frame_padding: int = 4

    def iter_path(self, iteration: int) -> Path:
        return self.iter_dir / f"{self.frame_prefix}{iteration:0{self.frame_padding}d}.png"

    def frame_pattern(self) -> str:
        return str(self.iter_dir / f"{self.frame_prefix}%0{self.frame_padding}d.png")

    def video_path(self, filename: str = "out.mp4") -> Path:
        return self.results_dir / filename

    def available_frames(self) -> Iterable[Path]:
        return self.iter_dir.glob(f"{self.frame_prefix}*.png")

    def make_video(self, filename: str = "out.mp4") -> bool:
        try:
            next(self.available_frames())
        except StopIteration:
            print("[video] no frames were generated; skipping video export")
            return False
        return build_video(self.frame_pattern(), self.video_path(filename), fps=self.video_fps, bitrate=self.video_bitrate)


def create_run_context(
    task_name: str,
    total_iters: int,
    *,
    results_root: Union[Path, str] = Path("results"),
    video_fps: int = 24,
    video_bitrate: str = "20M",
    label: Optional[str] = None,
) -> RunContext:
    results_dir = Path(results_root) / task_name
    iter_dir = results_dir / "iter"
    results_dir.mkdir(parents=True, exist_ok=True)
    if iter_dir.exists():
        shutil.rmtree(iter_dir)
    iter_dir.mkdir()
    progress = ProgressLogger(total_iters, label=label or task_name)
    return RunContext(task_name, total_iters, results_dir, iter_dir, video_fps, video_bitrate, progress)


def build_video(frame_pattern: str, output_path: Path, *, fps: int, bitrate: str) -> bool:
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-framerate",
        str(fps),
        "-i",
        frame_pattern,
        "-vb",
        bitrate,
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"[video] wrote {output_path}")
        return True
    except FileNotFoundError:
        print("[video] ffmpeg not found; skipping video export", file=sys.stderr)
        return False
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.decode("utf-8", "ignore").strip()
        if message:
            print(f"[video] ffmpeg error: {message}", file=sys.stderr)
        else:
            print(f"[video] ffmpeg exited with code {exc.returncode}", file=sys.stderr)
        return False


def log_run_configuration(
    task_name: str,
    config: Dict[str, Union[str, Number]],
    *,
    env_keys: Optional[Iterable[str]] = None,
) -> None:
    print(f"[config] {task_name}")
    for key, value in config.items():
        print(f"[config] {key}: {value}")
    backend = pydiffvg.get_backend()
    device = pydiffvg.get_device()
    try:
        api = get_api(backend)
        backend_impl = api.apply.__module__
    except Exception:  # pragma: no cover - defensive path
        backend_impl = "<unavailable>"
    backend_line = f"[config] backend: {backend} ({backend_impl})"
    suffix_parts = []
    if getattr(device, "type", None):
        suffix_parts.append(f"device={device}")
    if suffix_parts:
        backend_line += " [" + ", ".join(suffix_parts) + "]"
    print(backend_line)
    # Report env vars that contain "CUDA", "TORCH", or "DIFFVG" (case-insensitive).
    # If env_keys is provided, include those keys first (even if they don't match substrings).
    substrings = ("cuda", "torch", "diffvg")

    initial_keys = tuple(env_keys) if env_keys is not None else ()
    keys_list = list(initial_keys)
    keys_set = set(keys_list)

    for name in os.environ:
        lname = name.lower()
        if any(sub in lname for sub in substrings) and name not in keys_set:
            keys_list.append(name)
            keys_set.add(name)

    found_any = False
    for key in keys_list:
        value = os.environ.get(key)
        if value is not None:
            if not found_any:
                print("[config] env:")
                found_any = True
            print(f"[config]   {key}={value}")
    print("[config] --")
