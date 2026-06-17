from __future__ import annotations

import time
from collections import deque
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from rich.table import Table
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Rich logging requires the train dependency group: `uv sync --group train`."
    ) from exc


class RichTrainingCallback(BaseCallback):
    """Clean terminal progress UI with ETA, reward stats, and checkpoint status."""

    def __init__(
        self,
        *,
        total_timesteps: int,
        output_dir: str | Path,
        checkpoint_freq: int,
        log_interval_seconds: float = 5.0,
    ) -> None:
        super().__init__()
        self.total_timesteps = total_timesteps
        self.output_dir = Path(output_dir)
        self.checkpoint_freq = checkpoint_freq
        self.log_interval_seconds = log_interval_seconds
        self.console = Console()
        self.progress: Progress | None = None
        self.task_id = None
        self.started_at = 0.0
        self.start_timesteps = 0
        self.last_update_at = 0.0
        self.episode_rewards: deque[float] = deque(maxlen=100)
        self.episode_lengths: deque[int] = deque(maxlen=100)
        self.episode_successes: deque[float] = deque(maxlen=100)
        self.episode_truncations: deque[float] = deque(maxlen=100)
        self.final_distances: deque[float] = deque(maxlen=100)
        self.best_mean_reward = -np.inf
        self.best_model_path = self.output_dir / "best_model.zip"

    def _on_training_start(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.started_at = time.monotonic()
        self.start_timesteps = int(self.model.num_timesteps)
        self.last_update_at = self.started_at
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]Training SAC mouse policy"),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("elapsed •"),
            TimeRemainingColumn(),
            TextColumn("ETA"),
            console=self.console,
            transient=False,
        )
        self.progress.start()
        self.task_id = self.progress.add_task("sac", total=self.total_timesteps)
        self.console.print(
            Panel.fit(
                f"[bold]Output[/bold]: {self.output_dir}\n"
                f"[bold]Checkpoints[/bold]: every {self.checkpoint_freq:,} steps\n"
                f"[bold]TensorBoard[/bold]: {self.output_dir / 'logs'}",
                title="Bumblebee RL Training",
                border_style="cyan",
            )
        )

    def _on_step(self) -> bool:
        assert self.progress is not None and self.task_id is not None

        for info in self.locals.get("infos", []):
            episode = info.get("episode")
            if episode:
                self.episode_rewards.append(float(episode["r"]))
                self.episode_lengths.append(int(episode["l"]))
                self.episode_successes.append(float(bool(info.get("success", False))))
                self.episode_truncations.append(
                    float(bool(info.get("truncated", False)))
                )
                if "distance_to_target" in info:
                    self.final_distances.append(float(info["distance_to_target"]))

        now = time.monotonic()
        if now - self.last_update_at >= self.log_interval_seconds:
            self.progress.update(
                self.task_id, completed=min(self.num_timesteps, self.total_timesteps)
            )
            self._render_status()
            self.last_update_at = now
        return True

    def _render_status(self) -> None:
        elapsed = max(time.monotonic() - self.started_at, 1e-9)
        run_timesteps = max(int(self.num_timesteps) - self.start_timesteps, 0)
        steps_per_second = run_timesteps / elapsed
        mean_reward = (
            float(np.mean(self.episode_rewards)) if self.episode_rewards else 0.0
        )
        mean_length = (
            float(np.mean(self.episode_lengths)) if self.episode_lengths else 0.0
        )
        success_rate = (
            100.0 * float(np.mean(self.episode_successes))
            if self.episode_successes
            else 0.0
        )
        truncation_rate = (
            100.0 * float(np.mean(self.episode_truncations))
            if self.episode_truncations
            else 0.0
        )
        mean_final_distance = (
            float(np.mean(self.final_distances)) if self.final_distances else 0.0
        )

        if self.episode_rewards and mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            self.model.save(self.best_model_path)

        table = Table.grid(expand=True)
        table.add_column(justify="left")
        table.add_column(justify="right")
        table.add_row("Steps", f"{self.num_timesteps:,}/{self.total_timesteps:,}")
        table.add_row("Run steps", f"{run_timesteps:,}")
        table.add_row("Throughput", f"{steps_per_second:,.1f} steps/s")
        table.add_row("Episodes", f"{len(self.episode_rewards):,} recent")
        table.add_row("Mean reward (100 ep)", f"{mean_reward:,.4f}")
        table.add_row("Mean episode length", f"{mean_length:,.1f}")
        table.add_row("Success rate (100 ep)", f"{success_rate:,.1f}%")
        table.add_row("Truncation rate (100 ep)", f"{truncation_rate:,.1f}%")
        table.add_row("Mean final distance", f"{mean_final_distance:,.1f}px")
        table.add_row("Best mean reward", f"{self.best_mean_reward:,.4f}")
        table.add_row("Best model", str(self.best_model_path))
        self.console.print(Panel(table, border_style="green", title="Live metrics"))

    def _on_training_end(self) -> None:
        if self.progress is not None:
            self.progress.update(
                self.task_id, completed=min(self.num_timesteps, self.total_timesteps)
            )
            self.progress.stop()
        self._render_status()
