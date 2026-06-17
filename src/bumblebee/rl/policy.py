from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bumblebee.models import packaged_mouse_model_file

from .envs.mouse import MouseEnvConfig


def resolve_device(requested: str) -> str:
    """Resolve `auto` to cuda, mps, or cpu without importing torch at module import."""

    if requested != "auto":
        return requested
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "RL policy loading requires torch. Install with `the-bumblebee[rl]` "
            "or run `uv sync --group train` in a checkout."
        ) from exc

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass(frozen=True)
class MousePolicyRollout:
    points: np.ndarray
    actions: np.ndarray
    speeds: np.ndarray


class SB3MousePolicyPathProvider:
    """Callable path provider that rolls out a packaged or external SB3 SAC model."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        config: MouseEnvConfig = MouseEnvConfig(),
        device: str = "auto",
        deterministic: bool = False,
    ) -> None:
        try:
            from stable_baselines3 import SAC
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "Stable-Baselines3 is required to load RL mouse policies. "
                "Install with `the-bumblebee[rl]` or run `uv sync --group train`."
            ) from exc

        self.config = config
        self.device = resolve_device(device)
        self.deterministic = deterministic
        self.model = SAC.load(model_path, device=self.device)

    @classmethod
    def from_packaged(
        cls,
        *,
        config: MouseEnvConfig = MouseEnvConfig(),
        device: str = "auto",
        deterministic: bool = False,
    ) -> "SB3MousePolicyPathProvider":
        """Load the default model bundled in the Bumblebee package."""

        with packaged_mouse_model_file() as model_path:
            return cls(
                model_path,
                config=config,
                device=device,
                deterministic=deterministic,
            )

    def make_observation(
        self,
        position: np.ndarray,
        destination: np.ndarray,
        previous_velocity: np.ndarray,
        step: int,
    ) -> np.ndarray:
        screen_scale = np.array(
            [self.config.screen.width, self.config.screen.height], dtype=np.float64
        )
        delta = destination - position
        distance = float(np.linalg.norm(delta))
        return np.concatenate(
            [
                position / screen_scale,
                destination / screen_scale,
                delta / screen_scale,
                np.array([distance / self.config.screen.diagonal], dtype=np.float64),
                previous_velocity / self.config.max_velocity_px_s,
                np.array([step / self.config.max_steps], dtype=np.float64),
            ]
        ).astype(np.float32)

    def rollout(
        self,
        start: np.ndarray,
        destination: np.ndarray,
        *,
        deterministic: bool | None = None,
    ) -> MousePolicyRollout:
        deterministic = self.deterministic if deterministic is None else deterministic
        position = np.asarray(start, dtype=np.float64).copy()
        destination = np.asarray(destination, dtype=np.float64).copy()
        previous_velocity = np.zeros(2, dtype=np.float64)
        points = [position.copy()]
        actions = []
        speeds = []

        for step in range(self.config.max_steps):
            obs = self.make_observation(position, destination, previous_velocity, step)
            action, _ = self.model.predict(obs, deterministic=deterministic)
            action = np.asarray(action, dtype=np.float64).reshape(-1)
            action = np.clip(action, -1.0, 1.0)

            movement = action[:2]
            movement_norm = float(np.linalg.norm(movement))
            if movement_norm > 1.0:
                movement = movement / movement_norm
            velocity = movement * self.config.max_velocity_px_s
            position = position + velocity * self.config.dt
            position[0] = np.clip(position[0], 0, self.config.screen.width - 1)
            position[1] = np.clip(position[1], 0, self.config.screen.height - 1)

            previous_velocity = velocity
            points.append(position.copy())
            actions.append(action.copy())
            speeds.append(float(np.linalg.norm(velocity)))

            if np.linalg.norm(position - destination) <= self.config.target_radius_px:
                break

        return MousePolicyRollout(
            points=np.asarray(points, dtype=np.float64),
            actions=np.asarray(actions, dtype=np.float64),
            speeds=np.asarray(speeds, dtype=np.float64),
        )

    def __call__(self, start: np.ndarray, destination: np.ndarray) -> np.ndarray:
        """Return an `N x 2` path suitable for `Mouse(path_provider=...)`."""

        return self.rollout(start, destination).points
