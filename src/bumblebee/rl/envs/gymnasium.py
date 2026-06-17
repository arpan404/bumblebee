from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..data import MouseDemonstrationDataset
from .mouse import MouseEnvConfig, MouseImitationEnv

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover - exercised only without train deps
    raise ImportError(
        "Gymnasium is required for RL training. Install with `uv sync --group train`."
    ) from exc

_OBSERVATION_LOW = np.array([0, 0, 0, 0, -1, -1, 0, -1, -1, 0], dtype=np.float32)
_OBSERVATION_HIGH = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32)


class GymMouseImitationEnv(gym.Env):
    """Gymnasium wrapper around :class:`MouseImitationEnv` for SB3/SAC training."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        dataset_path: str | Path,
        config: MouseEnvConfig = MouseEnvConfig(),
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.dataset = MouseDemonstrationDataset.load(dataset_path)
        self.env = MouseImitationEnv(self.dataset, config=config, seed=seed)

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(MouseImitationEnv.action_size,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=_OBSERVATION_LOW,
            high=_OBSERVATION_HIGH,
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        _ = options
        super().reset(seed=seed)
        observation = self.env.reset(seed=seed)
        info: dict[str, Any] = {}
        return observation, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        observation, reward, done, info = self.env.step(action)
        terminated = bool(info.get("reached", False))
        truncated = bool(info.get("truncated", False))
        if done and not terminated and not truncated:
            truncated = True
        return observation, reward, terminated, truncated, info
