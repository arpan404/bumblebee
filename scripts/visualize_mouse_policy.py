#!/usr/bin/env python
from __future__ import annotations

import argparse
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox

import numpy as np
import torch
from stable_baselines3 import SAC

from bumblebee.rl.envs.mouse import MouseEnvConfig, VirtualScreen


@dataclass
class Rollout:
    points: np.ndarray
    actions: np.ndarray
    speeds: np.ndarray


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_observation(
    position: np.ndarray,
    destination: np.ndarray,
    previous_velocity: np.ndarray,
    step: int,
    config: MouseEnvConfig,
) -> np.ndarray:
    screen_scale = np.array(
        [config.screen.width, config.screen.height], dtype=np.float64
    )
    delta = destination - position
    distance = float(np.linalg.norm(delta))
    return np.concatenate(
        [
            position / screen_scale,
            destination / screen_scale,
            delta / screen_scale,
            np.array([distance / config.screen.diagonal], dtype=np.float64),
            previous_velocity / config.max_velocity_px_s,
            np.array([step / config.max_steps], dtype=np.float64),
        ]
    ).astype(np.float32)


def rollout_policy(
    model: SAC,
    start: np.ndarray,
    destination: np.ndarray,
    config: MouseEnvConfig,
    *,
    deterministic: bool,
) -> Rollout:
    position = start.astype(np.float64).copy()
    previous_velocity = np.zeros(2, dtype=np.float64)
    points = [position.copy()]
    actions = []
    speeds = []

    for step in range(config.max_steps):
        obs = make_observation(position, destination, previous_velocity, step, config)
        action, _ = model.predict(obs, deterministic=deterministic)
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        action = np.clip(action, -1.0, 1.0)

        movement_action = action[:2]
        movement_norm = float(np.linalg.norm(movement_action))
        if movement_norm > 1.0:
            movement_action = movement_action / movement_norm
        velocity = movement_action * config.max_velocity_px_s
        position = position + velocity * config.dt
        position[0] = np.clip(position[0], 0, config.screen.width - 1)
        position[1] = np.clip(position[1], 0, config.screen.height - 1)

        previous_velocity = velocity
        points.append(position.copy())
        actions.append(action.copy())
        speeds.append(float(np.linalg.norm(velocity)))

        if np.linalg.norm(position - destination) <= config.target_radius_px:
            break

    return Rollout(
        points=np.asarray(points),
        actions=np.asarray(actions),
        speeds=np.asarray(speeds),
    )


class PolicyVisualizer:
    path_colors = [
        "#38bdf8",
        "#f59e0b",
        "#a78bfa",
        "#34d399",
        "#fb7185",
        "#facc15",
        "#60a5fa",
        "#f472b6",
    ]

    def __init__(self, root: tk.Tk, args: argparse.Namespace) -> None:
        self.root = root
        self.root.title("Bumblebee RL Mouse Policy Visualizer")

        self.config = MouseEnvConfig(
            screen=VirtualScreen(width=args.screen_width, height=args.screen_height),
            max_steps=args.max_steps,
            dt=args.dt,
            max_velocity_px_s=args.max_velocity_px_s,
            target_radius_px=args.target_radius,
        )
        self.device = resolve_device(args.device)
        self.model_path = Path(args.model)
        self.model = self._load_model(self.model_path)

        self.canvas_width = args.canvas_width
        self.canvas_height = args.canvas_height
        self.start: np.ndarray | None = None
        self.destination: np.ndarray | None = None
        self.rollout: Rollout | None = None
        self.rollouts: list[Rollout] = []

        self._build_ui()
        self._draw_empty_screen()

    def _load_model(self, path: Path) -> SAC:
        if not path.exists():
            raise FileNotFoundError(f"model not found: {path}")
        return SAC.load(path, device=self.device)

    def _build_ui(self) -> None:
        header = tk.Frame(self.root, padx=10, pady=8)
        header.pack(fill=tk.X)

        self.status = tk.StringVar(
            value=(
                f"Model: {self.model_path} | Device: {self.device} | "
                "Click once for start, again for destination."
            )
        )
        tk.Label(header, textvariable=self.status, anchor="w").pack(fill=tk.X)

        controls = tk.Frame(self.root, padx=10)
        controls.pack(fill=tk.X)

        self.deterministic = tk.BooleanVar(value=False)
        tk.Checkbutton(
            controls,
            text="Deterministic path",
            variable=self.deterministic,
            command=self._replace_if_ready,
        ).pack(side=tk.LEFT)

        tk.Button(
            controls,
            text="Regenerate same path",
            command=self.regenerate_same_path,
        ).pack(side=tk.LEFT, padx=6)
        tk.Button(controls, text="Replace path", command=self._replace_if_ready).pack(
            side=tk.LEFT
        )
        tk.Button(controls, text="Clear", command=self.clear).pack(side=tk.LEFT)
        tk.Button(controls, text="Load model…", command=self.load_model_dialog).pack(
            side=tk.LEFT, padx=6
        )

        self.metrics = tk.StringVar(value="No path generated yet.")
        tk.Label(controls, textvariable=self.metrics, anchor="e").pack(side=tk.RIGHT)

        self.canvas = tk.Canvas(
            self.root,
            width=self.canvas_width,
            height=self.canvas_height,
            bg="#0f172a",
            highlightthickness=0,
        )
        self.canvas.pack(padx=10, pady=10)
        self.canvas.bind("<Button-1>", self.on_click)

    def _draw_empty_screen(self) -> None:
        self.canvas.delete("all")
        grid_color = "#1e293b"
        for x in range(0, self.canvas_width, 80):
            self.canvas.create_line(x, 0, x, self.canvas_height, fill=grid_color)
        for y in range(0, self.canvas_height, 80):
            self.canvas.create_line(0, y, self.canvas_width, y, fill=grid_color)
        self.canvas.create_text(
            self.canvas_width // 2,
            24,
            text="Click start point, then destination point",
            fill="#94a3b8",
            font=("Helvetica", 14),
        )

    def screen_to_canvas(self, point: np.ndarray) -> tuple[float, float]:
        return (
            float(point[0] / self.config.screen.width * self.canvas_width),
            float(point[1] / self.config.screen.height * self.canvas_height),
        )

    def canvas_to_screen(self, x: float, y: float) -> np.ndarray:
        return np.array(
            [
                x / self.canvas_width * self.config.screen.width,
                y / self.canvas_height * self.config.screen.height,
            ],
            dtype=np.float64,
        )

    def on_click(self, event: tk.Event) -> None:
        point = self.canvas_to_screen(event.x, event.y)
        if self.start is None or (
            self.start is not None and self.destination is not None
        ):
            self.start = point
            self.destination = None
            self.rollout = None
            self.rollouts = []
            self._draw_empty_screen()
            self._draw_marker(point, "#22c55e", "START")
            self.status.set("Start selected. Now click destination.")
            return

        self.destination = point
        self.generate_and_draw(append=False)

    def generate_and_draw(self, *, append: bool) -> None:
        if self.start is None or self.destination is None:
            return
        self.rollout = rollout_policy(
            self.model,
            self.start,
            self.destination,
            self.config,
            deterministic=self.deterministic.get(),
        )
        if append:
            self.rollouts.append(self.rollout)
        else:
            self.rollouts = [self.rollout]
        self.draw_rollout()

    def draw_rollout(self) -> None:
        assert (
            self.start is not None
            and self.destination is not None
            and self.rollout is not None
        )
        self._draw_empty_screen()
        self._draw_marker(self.start, "#22c55e", "START")
        self._draw_marker(self.destination, "#ef4444", "DEST")

        for rollout_index, rollout in enumerate(self.rollouts):
            color = self.path_colors[rollout_index % len(self.path_colors)]
            is_latest = rollout is self.rollout
            canvas_points = [self.screen_to_canvas(point) for point in rollout.points]
            if len(canvas_points) < 2:
                continue

            flat = [coord for point in canvas_points for coord in point]
            self.canvas.create_line(
                *flat,
                fill=color,
                width=4 if is_latest else 2,
                smooth=True,
            )
            end_x, end_y = canvas_points[-1]
            self.canvas.create_oval(
                end_x - 4,
                end_y - 4,
                end_x + 4,
                end_y + 4,
                fill=color,
                outline="white" if is_latest else "",
            )
            self.canvas.create_text(
                end_x + 8,
                end_y,
                text=str(rollout_index + 1),
                fill=color,
                anchor="w",
                font=("Helvetica", 10, "bold"),
            )

            if is_latest:
                sample_step = max(1, len(canvas_points) // 20)
                for x, y in canvas_points[::sample_step]:
                    self.canvas.create_oval(
                        x - 2, y - 2, x + 2, y + 2, fill="#e0f2fe", outline=""
                    )

        self._draw_path_legend()
        self._update_metrics()

    def _update_metrics(self) -> None:
        assert self.destination is not None and self.rollout is not None

        final_distance = float(
            np.linalg.norm(self.rollout.points[-1] - self.destination)
        )
        duration = len(self.rollout.speeds) * self.config.dt
        mean_speed = (
            float(np.mean(self.rollout.speeds)) if len(self.rollout.speeds) else 0.0
        )
        all_errors = [
            float(np.linalg.norm(rollout.points[-1] - self.destination))
            for rollout in self.rollouts
        ]
        error_range = (
            f"best={min(all_errors):.1f}px | worst={max(all_errors):.1f}px"
            if all_errors
            else "best=0.0px | worst=0.0px"
        )
        self.metrics.set(
            f"paths={len(self.rollouts)} | latest steps={len(self.rollout.points)-1} | "
            f"duration={duration:.3f}s | mean speed={mean_speed:.1f}px/s | "
            f"latest error={final_distance:.1f}px | {error_range}"
        )
        self.status.set(
            "Path generated. Use Regenerate same path to overlay another rollout, "
            "or click to choose a new start."
        )

    def _draw_path_legend(self) -> None:
        if not self.rollouts:
            return
        x = 12
        y = self.canvas_height - 18
        for rollout_index, _rollout in enumerate(self.rollouts[-8:]):
            absolute_index = (
                len(self.rollouts) - min(len(self.rollouts), 8) + rollout_index
            )
            color = self.path_colors[absolute_index % len(self.path_colors)]
            self.canvas.create_line(x, y, x + 22, y, fill=color, width=4)
            self.canvas.create_text(
                x + 28,
                y,
                text=f"#{absolute_index + 1}",
                fill=color,
                anchor="w",
                font=("Helvetica", 10, "bold"),
            )
            x += 58

    def regenerate_same_path(self) -> None:
        if self.start is None or self.destination is None:
            self.status.set("Select a start and destination before regenerating.")
            return
        self.generate_and_draw(append=True)

    def _replace_if_ready(self) -> None:
        if self.start is not None and self.destination is not None:
            self.generate_and_draw(append=False)

    def clear(self) -> None:
        self.start = None
        self.destination = None
        self.rollout = None
        self.rollouts = []
        self.metrics.set("No path generated yet.")
        self.status.set("Cleared. Click once for start, again for destination.")
        self._draw_empty_screen()

    def _draw_marker(self, point: np.ndarray, color: str, label: str) -> None:
        x, y = self.screen_to_canvas(point)
        self.canvas.create_oval(
            x - 7, y - 7, x + 7, y + 7, fill=color, outline="white", width=2
        )
        self.canvas.create_text(x + 12, y - 12, text=label, fill=color, anchor="w")

    def load_model_dialog(self) -> None:
        filename = filedialog.askopenfilename(
            title="Load SAC model",
            filetypes=[("Stable-Baselines model", "*.zip"), ("All files", "*.*")],
        )
        if not filename:
            return
        try:
            self.model_path = Path(filename)
            self.model = self._load_model(self.model_path)
            self.status.set(f"Loaded model: {self.model_path} | Device: {self.device}")
            self._replace_if_ready()
        except Exception as exc:  # noqa: BLE001 - GUI should display any load failure
            messagebox.showerror("Failed to load model", str(exc))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize generated mouse paths from a SAC model."
    )
    parser.add_argument("--model", default="artifacts/rl/sac_mouse/best_model.zip")
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda", "mps"]
    )
    parser.add_argument("--screen-width", type=int, default=4096)
    parser.add_argument("--screen-height", type=int, default=2304)
    parser.add_argument("--canvas-width", type=int, default=1200)
    parser.add_argument("--canvas-height", type=int, default=675)
    parser.add_argument("--max-steps", type=int, default=96)
    parser.add_argument("--dt", type=float, default=1 / 120)
    parser.add_argument("--max-velocity-px-s", type=float, default=6500.0)
    parser.add_argument("--target-radius", type=float, default=8.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = tk.Tk()
    try:
        PolicyVisualizer(root, args)
    except Exception as exc:  # noqa: BLE001 - GUI startup error
        messagebox.showerror("Failed to start visualizer", str(exc))
        raise
    root.mainloop()


if __name__ == "__main__":
    main()
