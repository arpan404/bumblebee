from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import Demonstration, MouseDemonstrationDataset
from .reward import ImitationReward


@dataclass(frozen=True)
class VirtualScreen:
    width: int = 4096
    height: int = 2304


@dataclass(frozen=True)
class MouseEnvConfig:
    screen: VirtualScreen = VirtualScreen()
    max_steps: int = 64
    dt: float = 1 / 120
    max_velocity_px_s: float = 6500.0
    min_start_dest_distance_px: float = 20.0
    target_radius_px: float = 5.0


class MouseImitationEnv:
    """Small RL environment for stochastic mouse trajectory imitation.

    Observation:
        [x, y, dest_x, dest_y, previous_vx, previous_vy, progress]
        normalized into roughly ``[0, 1]`` / velocity scale.

    Action:
        Two continuous values in ``[-1, 1]`` representing velocity direction and
        magnitude components. The environment integrates them on a virtual screen.

    Episode target:
        On reset, a random start/destination is sampled and one cleaned real
        demonstration signature is transformed to that pair. Different resets for
        the same coordinates may sample different demonstrations, preserving the
        stochastic path and speed behavior present in the real data.
    """

    def __init__(
        self,
        demonstrations: MouseDemonstrationDataset,
        config: MouseEnvConfig = MouseEnvConfig(),
        reward_fn: ImitationReward | None = None,
        seed: int | None = None,
    ) -> None:
        self.demonstrations = demonstrations
        self.config = config
        self.reward_fn = reward_fn or ImitationReward()
        self.rng = np.random.default_rng(seed)
        self.position = np.zeros(2, dtype=np.float64)
        self.destination = np.zeros(2, dtype=np.float64)
        self.previous_velocity = np.zeros(2, dtype=np.float64)
        self.rollout_points: list[np.ndarray] = []
        self.rollout_speeds: list[float] = []
        self.demo: Demonstration | None = None
        self.step_count = 0

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.position, self.destination = self._sample_task()
        self.previous_velocity = np.zeros(2, dtype=np.float64)
        self.rollout_points = [self.position.copy()]
        self.rollout_speeds = []
        self.demo = self.demonstrations.sample(
            self.position, self.destination, self.rng
        )
        self.step_count = 0
        return self._observation()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (2,):
            raise ValueError("action must have shape (2,)")

        action = np.clip(action, -1.0, 1.0)
        velocity = action * self.config.max_velocity_px_s
        self.position = self.position + velocity * self.config.dt
        self.position[0] = np.clip(self.position[0], 0, self.config.screen.width - 1)
        self.position[1] = np.clip(self.position[1], 0, self.config.screen.height - 1)

        speed = float(np.linalg.norm(velocity))
        self.previous_velocity = velocity
        self.rollout_points.append(self.position.copy())
        self.rollout_speeds.append(speed)
        self.step_count += 1

        reached = (
            np.linalg.norm(self.position - self.destination)
            <= self.config.target_radius_px
        )
        truncated = self.step_count >= self.config.max_steps
        done = bool(reached or truncated)

        reward = self._step_reward(reached)
        info = {
            "reached": reached,
            "truncated": truncated,
            "destination": self.destination.copy(),
            "demo_path": None if self.demo is None else self.demo.path.copy(),
        }
        return self._observation(), reward, done, info

    def _sample_task(self) -> tuple[np.ndarray, np.ndarray]:
        screen = self.config.screen
        while True:
            start = np.array(
                [
                    self.rng.uniform(0, screen.width - 1),
                    self.rng.uniform(0, screen.height - 1),
                ],
                dtype=np.float64,
            )
            destination = np.array(
                [
                    self.rng.uniform(0, screen.width - 1),
                    self.rng.uniform(0, screen.height - 1),
                ],
                dtype=np.float64,
            )
            if (
                np.linalg.norm(destination - start)
                >= self.config.min_start_dest_distance_px
            ):
                return start, destination

    def _step_reward(self, reached: bool) -> float:
        # Dense shaping: move toward destination and discourage speedless dithering.
        distance = float(np.linalg.norm(self.position - self.destination))
        max_distance = float(
            np.hypot(self.config.screen.width, self.config.screen.height)
        )
        reward = -0.01 - 0.05 * (distance / max_distance)

        if reached or self.step_count >= self.config.max_steps:
            assert self.demo is not None
            reward += self.reward_fn(
                np.asarray(self.rollout_points),
                np.asarray(self.rollout_speeds),
                self.demo.path,
                self.demo.speed_profile,
                self.rollout_points[0],
                self.destination,
            )
        return float(reward)

    def _observation(self) -> np.ndarray:
        screen_scale = np.array(
            [self.config.screen.width, self.config.screen.height], dtype=np.float64
        )
        return np.concatenate(
            [
                self.position / screen_scale,
                self.destination / screen_scale,
                self.previous_velocity / self.config.max_velocity_px_s,
                np.array([self.step_count / self.config.max_steps], dtype=np.float64),
            ]
        ).astype(np.float32)
