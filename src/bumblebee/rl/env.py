from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .data import MouseDemonstrationDataset
from .reward import ImitationReward


@dataclass(frozen=True)
class VirtualScreen:
    width: int = 4096
    height: int = 2304

    @property
    def diagonal(self) -> float:
        return float(np.hypot(self.width, self.height))


@dataclass(frozen=True)
class MouseEnvConfig:
    screen: VirtualScreen = VirtualScreen()
    max_steps: int = 96
    dt: float = 1 / 120
    max_velocity_px_s: float = 6500.0
    min_start_dest_distance_px: float = 20.0
    max_start_dest_distance_px: float | None = None
    target_radius_px: float = 8.0
    success_reward: float = 12.0
    imitation_bonus: float = 2.0
    failure_penalty: float = -3.0
    step_penalty: float = -0.01
    progress_reward_scale: float = 0.35
    backward_penalty_scale: float = 3.0
    acceleration_penalty_scale: float = 0.00002
    jitter_penalty_scale: float = 0.10
    small_move_penalty: float = -0.04
    small_move_speed_px_s: float = 75.0
    local_loop_penalty: float = -0.12
    local_loop_window: int = 12
    local_loop_min_travel_px: float = 50.0
    local_loop_efficiency_threshold: float = 0.25
    task_reachability_margin: float = 0.85
    record_reward_terms: bool = True

    @property
    def max_reachable_distance_px(self) -> float:
        physical_max = (
            self.max_velocity_px_s
            * self.dt
            * self.max_steps
            * self.task_reachability_margin
        )
        configured_max = self.max_start_dest_distance_px
        if configured_max is None:
            return min(self.screen.diagonal, physical_max)
        return min(configured_max, self.screen.diagonal, physical_max)


class MouseImitationEnv:
    """High-throughput virtual-screen environment for mouse trajectory imitation.

    Observation shape is 10:
        ``position_xy, destination_xy, delta_xy, distance, previous_velocity_xy, progress``.

    Action shape is 2:
        ``vx, vy`` in ``[-1, 1]``. The vector is clipped to the unit circle so
        ``max_velocity_px_s`` is a real speed limit. Episodes terminate as soon
        as the cursor reaches/crosses the target; ``max_steps`` is only a safety
        cutoff, not the intended path length.
    """

    observation_size = 10
    action_size = 2

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

        self.position = np.zeros(2, dtype=np.float32)
        self.destination = np.zeros(2, dtype=np.float32)
        self.previous_velocity = np.zeros(2, dtype=np.float32)
        self.previous_previous_velocity = np.zeros(2, dtype=np.float32)
        self.rollout_points = np.zeros((config.max_steps + 1, 2), dtype=np.float32)
        self.rollout_speeds = np.zeros(config.max_steps, dtype=np.float32)
        self.cumulative_travel = np.zeros(config.max_steps + 1, dtype=np.float32)
        self.observation = np.zeros(self.observation_size, dtype=np.float32)

        self.demo_index = 0
        self.step_count = 0
        self.previous_distance = 0.0
        self.start_distance = 1.0
        self.last_reward_terms: dict[str, float] = {}

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.position, self.destination = self._sample_task()
        self.previous_velocity.fill(0.0)
        self.previous_previous_velocity.fill(0.0)
        self.rollout_points[0] = self.position
        self.rollout_speeds.fill(0.0)
        self.cumulative_travel.fill(0.0)
        self.demo_index = self.demonstrations.sample_index(self.rng)
        self.step_count = 0
        self.previous_distance = self._distance_to_target()
        self.start_distance = max(self.previous_distance, 1.0)
        self.last_reward_terms = {}
        return self._observation(self.previous_distance)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape != (self.action_size,):
            raise ValueError(f"action must have shape ({self.action_size},)")

        vx, vy = self._action_to_velocity_xy(action)
        previous_x = float(self.position[0])
        previous_y = float(self.position[1])
        new_x = previous_x + vx * self.config.dt
        new_y = previous_y + vy * self.config.dt
        new_x = min(max(new_x, 0.0), self.config.screen.width - 1)
        new_y = min(max(new_y, 0.0), self.config.screen.height - 1)

        reached_target = self._segment_reaches_target_xy(
            previous_x, previous_y, new_x, new_y
        )
        if reached_target:
            new_x = float(self.destination[0])
            new_y = float(self.destination[1])

        self.position[0] = new_x
        self.position[1] = new_y

        self.previous_previous_velocity[:] = self.previous_velocity
        self.previous_velocity[0] = vx
        self.previous_velocity[1] = vy
        speed = math.sqrt(vx * vx + vy * vy)
        movement_x = new_x - previous_x
        movement_y = new_y - previous_y
        movement_distance = math.sqrt(movement_x * movement_x + movement_y * movement_y)

        self.rollout_speeds[self.step_count] = speed
        self.cumulative_travel[self.step_count + 1] = (
            self.cumulative_travel[self.step_count] + movement_distance
        )
        self.step_count += 1
        self.rollout_points[self.step_count] = self.position

        truncated = self.step_count >= self.config.max_steps and not reached_target
        done = bool(reached_target or truncated)
        distance = 0.0 if reached_target else self._distance_to_target()

        reward, reward_terms = self._step_reward(
            reached=reached_target,
            truncated=truncated,
            distance=distance,
            speed=speed,
        )
        self.last_reward_terms = reward_terms

        info = {
            "success": reached_target,
            "reached": reached_target,
            "truncated": truncated,
            "distance_to_target": distance,
        }
        if reward_terms:
            info["reward_terms"] = reward_terms
        return self._observation(distance), reward, done, info

    def _action_to_velocity_xy(
        self, movement_action: np.ndarray
    ) -> tuple[float, float]:
        ax = min(max(float(movement_action[0]), -1.0), 1.0)
        ay = min(max(float(movement_action[1]), -1.0), 1.0)
        norm_sq = ax * ax + ay * ay
        if norm_sq > 1.0:
            scale = 1.0 / math.sqrt(norm_sq)
            ax *= scale
            ay *= scale
        max_velocity = self.config.max_velocity_px_s
        return ax * max_velocity, ay * max_velocity

    def _distance_to_target(self) -> float:
        dx = float(self.position[0] - self.destination[0])
        dy = float(self.position[1] - self.destination[1])
        return math.sqrt(dx * dx + dy * dy)

    def _segment_reaches_target_xy(
        self, start_x: float, start_y: float, end_x: float, end_y: float
    ) -> bool:
        segment_x = end_x - start_x
        segment_y = end_y - start_y
        length_sq = segment_x * segment_x + segment_y * segment_y
        if length_sq <= 1e-9:
            closest_x = start_x
            closest_y = start_y
        else:
            target_dx = float(self.destination[0]) - start_x
            target_dy = float(self.destination[1]) - start_y
            t = (target_dx * segment_x + target_dy * segment_y) / length_sq
            t = min(max(t, 0.0), 1.0)
            closest_x = start_x + t * segment_x
            closest_y = start_y + t * segment_y
        radius = self.config.target_radius_px
        dx = closest_x - float(self.destination[0])
        dy = closest_y - float(self.destination[1])
        return dx * dx + dy * dy <= radius * radius

    def _sample_task(self) -> tuple[np.ndarray, np.ndarray]:
        screen = self.config.screen
        min_distance = self.config.min_start_dest_distance_px
        max_distance = self.config.max_reachable_distance_px
        if min_distance > max_distance:
            raise ValueError(
                "min_start_dest_distance_px is larger than the reachable max distance. "
                "Increase max_steps/max_velocity or lower the minimum distance."
            )

        for _ in range(512):
            start = np.array(
                [
                    self.rng.uniform(0, screen.width - 1),
                    self.rng.uniform(0, screen.height - 1),
                ],
                dtype=np.float32,
            )
            distance = float(self.rng.uniform(min_distance, max_distance))
            angle = float(self.rng.uniform(0.0, 2.0 * np.pi))
            offset = np.array(
                [np.cos(angle) * distance, np.sin(angle) * distance], dtype=np.float32
            )
            destination = start + offset
            if (
                0 <= destination[0] < screen.width
                and 0 <= destination[1] < screen.height
            ):
                return start, destination.astype(np.float32)

        # Robust fallback for edge cases with very large reachability radii.
        while True:
            start = np.array(
                [
                    self.rng.uniform(0, screen.width - 1),
                    self.rng.uniform(0, screen.height - 1),
                ],
                dtype=np.float32,
            )
            destination = np.array(
                [
                    self.rng.uniform(0, screen.width - 1),
                    self.rng.uniform(0, screen.height - 1),
                ],
                dtype=np.float32,
            )
            distance = float(np.linalg.norm(destination - start))
            if min_distance <= distance <= max_distance:
                return start, destination

    def _step_reward(
        self, reached: bool, truncated: bool, distance: float, speed: float
    ) -> tuple[float, dict[str, float]]:
        progress = (self.previous_distance - distance) / self.start_distance
        self.previous_distance = distance

        step_penalty = self.config.step_penalty
        progress_reward = self.config.progress_reward_scale * progress
        backward_penalty = (
            -self.config.backward_penalty_scale * abs(progress) if progress < 0 else 0.0
        )

        small_move_penalty = 0.0
        if (
            distance > self.config.target_radius_px * 2
            and speed < self.config.small_move_speed_px_s
        ):
            small_move_penalty = self.config.small_move_penalty

        acceleration_x = float(
            self.previous_velocity[0] - self.previous_previous_velocity[0]
        )
        acceleration_y = float(
            self.previous_velocity[1] - self.previous_previous_velocity[1]
        )
        acceleration_penalty = -self.config.acceleration_penalty_scale * math.sqrt(
            acceleration_x * acceleration_x + acceleration_y * acceleration_y
        )
        jitter_penalty = (
            -self.config.jitter_penalty_scale * self._direction_change_penalty()
        )
        local_loop_penalty = self._local_loop_penalty()

        success_reward = 0.0
        imitation_reward = 0.0
        early_finish_reward = 0.0
        terminal_penalty = 0.0

        if reached:
            points = self.rollout_points[: self.step_count + 1]
            speeds = self.rollout_speeds[: self.step_count]
            demo = self.demonstrations.get(self.demo_index, points[0], self.destination)
            imitation_score = self.reward_fn(
                points,
                speeds,
                demo.path,
                demo.speed_profile,
                points[0],
                self.destination,
            )
            success_reward = self.config.success_reward
            imitation_reward = self.config.imitation_bonus * imitation_score
            early_finish_reward = max(
                0.0, 1.0 - (self.step_count / self.config.max_steps)
            )
        elif truncated:
            terminal_penalty = self.config.failure_penalty

        reward = (
            step_penalty
            + progress_reward
            + backward_penalty
            + small_move_penalty
            + acceleration_penalty
            + jitter_penalty
            + local_loop_penalty
            + success_reward
            + imitation_reward
            + early_finish_reward
            + terminal_penalty
        )
        if not self.config.record_reward_terms:
            return float(reward), {}

        terms = {
            "step": step_penalty,
            "progress": progress_reward,
            "backward": backward_penalty,
            "small_move": small_move_penalty,
            "acceleration": acceleration_penalty,
            "jitter": jitter_penalty,
            "local_loop": local_loop_penalty,
            "success": success_reward,
            "imitation": imitation_reward,
            "early_finish": early_finish_reward,
            "terminal": terminal_penalty,
        }
        return float(reward), terms

    def _direction_change_penalty(self) -> float:
        previous_x = float(self.previous_previous_velocity[0])
        previous_y = float(self.previous_previous_velocity[1])
        current_x = float(self.previous_velocity[0])
        current_y = float(self.previous_velocity[1])
        previous_norm = math.sqrt(previous_x * previous_x + previous_y * previous_y)
        current_norm = math.sqrt(current_x * current_x + current_y * current_y)
        if previous_norm < 1e-6 or current_norm < 1e-6:
            return 0.0
        cosine = (previous_x * current_x + previous_y * current_y) / (
            previous_norm * current_norm
        )
        return min(max((0.25 - cosine) / 1.25, 0.0), 1.0)

    def _local_loop_penalty(self) -> float:
        window = self.config.local_loop_window
        if self.step_count + 1 < window:
            return 0.0
        start = self.step_count + 1 - window
        traveled = float(
            self.cumulative_travel[self.step_count] - self.cumulative_travel[start]
        )
        if traveled < self.config.local_loop_min_travel_px:
            return 0.0
        net_x = float(
            self.rollout_points[self.step_count, 0] - self.rollout_points[start, 0]
        )
        net_y = float(
            self.rollout_points[self.step_count, 1] - self.rollout_points[start, 1]
        )
        net = math.sqrt(net_x * net_x + net_y * net_y)
        efficiency = net / traveled
        if efficiency < self.config.local_loop_efficiency_threshold:
            severity = 1.0 - (efficiency / self.config.local_loop_efficiency_threshold)
            return self.config.local_loop_penalty * severity
        return 0.0

    def _observation(self, distance: float | None = None) -> np.ndarray:
        screen = self.config.screen
        delta_x = float(self.destination[0] - self.position[0])
        delta_y = float(self.destination[1] - self.position[1])
        if distance is None:
            distance = math.sqrt(delta_x * delta_x + delta_y * delta_y)

        self.observation[0] = self.position[0] / screen.width
        self.observation[1] = self.position[1] / screen.height
        self.observation[2] = self.destination[0] / screen.width
        self.observation[3] = self.destination[1] / screen.height
        self.observation[4] = delta_x / screen.width
        self.observation[5] = delta_y / screen.height
        self.observation[6] = distance / screen.diagonal
        self.observation[7] = self.previous_velocity[0] / self.config.max_velocity_px_s
        self.observation[8] = self.previous_velocity[1] / self.config.max_velocity_px_s
        self.observation[9] = self.step_count / self.config.max_steps
        return self.observation.copy()
