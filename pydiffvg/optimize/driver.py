"""High-level orchestration helpers for SVG optimization."""

from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Sequence, Union, TYPE_CHECKING

import torch

from .settings import SvgOptimizationSettings

if TYPE_CHECKING:
    from ..optimize_svg import OptimizableSvg


Schedule = Union[int, float, Sequence[Union[int, float]], Callable[[int], Union[int, float]]]
LossFn = Callable[[torch.Tensor, int, "SvgOptimizationDriver"], torch.Tensor]
Callback = Callable[[int, torch.Tensor, torch.Tensor, "SvgOptimizationDriver"], None]


def _resolve_schedule(schedule: Optional[Schedule], index: int, default):
    if schedule is None:
        return default
    if callable(schedule):
        return schedule(index)
    if isinstance(schedule, Sequence):
        return schedule[index]
    return schedule


class SvgOptimizationDriver:
    """Thin wrapper around :class:`OptimizableSvg` to coordinate optimization loops."""

    def __init__(
        self,
        svg_path: str,
        *,
        settings: Optional[SvgOptimizationSettings] = None,
        optimize_background: bool = False,
        verbose: bool = False,
        device: Optional[torch.device] = None,
    ) -> None:
        self.settings = settings or SvgOptimizationSettings()
        self.device = device or torch.device("cpu")
        from ..optimize_svg import OptimizableSvg as _OptimizableSvg

        self.document = _OptimizableSvg(
            svg_path,
            settings=self.settings,
            optimize_background=optimize_background,
            verbose=verbose,
            device=self.device,
        )
        self.iteration: int = 0
        self.loss_history: List[float] = []

    # -- Passthrough helpers -------------------------------------------------
    def build_scene(self):
        return self.document.build_scene()

    def zero_grad(self) -> None:
        self.document.zero_grad()

    def render(self, *, scale: Optional[float] = None, seed: Optional[int] = None) -> torch.Tensor:
        return self.document.render(scale=scale, seed=self.iteration if seed is None else seed)

    def step(self) -> None:
        self.document.step()
        self.iteration += 1

    def write_xml(self) -> str:
        return self.document.write_xml()

    def write_defs(self, root) -> None:
        return self.document.write_defs(root)

    def save_svg(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.write_xml())

    # -- Optimization loop ---------------------------------------------------
    def optimize(
        self,
        loss_fn: LossFn,
        iterations: int,
        *,
        seed_schedule: Optional[Schedule] = None,
        scale_schedule: Optional[Schedule] = None,
        callback: Optional[Callback] = None,
    ) -> List[float]:
        """Run a simple first-order optimization loop.

        Args:
            loss_fn: Callable that receives the rendered image, iteration index, and driver,
                returning a scalar tensor loss.
            iterations: Number of optimization steps to execute.
            seed_schedule: Optional schedule (callable, sequence, or constant) providing seeds
                for stochastic rendering. Defaults to the iteration index.
            scale_schedule: Optional schedule providing scale factors per iteration. Defaults to
                rendering at native resolution.
            callback: Optional hook invoked with ``(iteration, image, loss, driver)`` after each
                optimization step.

        Returns:
            List of scalar loss values recorded per iteration.
        """

        history: List[float] = []
        for t in range(iterations):
            self.zero_grad()
            seed = _resolve_schedule(seed_schedule, t, default=t)
            scale = _resolve_schedule(scale_schedule, t, default=None)

            image = self.render(scale=scale, seed=seed)
            loss = loss_fn(image, t, self)
            if not isinstance(loss, torch.Tensor):
                raise TypeError("loss_fn must return a torch.Tensor")
            if loss.dim() != 0:
                loss = loss.mean()

            loss.backward()
            self.step()

            numeric_loss = float(loss.detach().cpu())
            history.append(numeric_loss)
            self.loss_history.append(numeric_loss)

            if callback is not None:
                callback(t, image, loss, self)

        return history


__all__ = ["SvgOptimizationDriver"]
