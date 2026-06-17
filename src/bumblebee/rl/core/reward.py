from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import curvature, resample_polyline, to_local_frame


@dataclass(frozen=True)
class ImitationReward:
    """Terminal imitation bonus for successful trajectories only.

    Reaching the destination is the primary objective. Path/velocity/turn matching is
    only used as a bonus after the agent has reached the target. This prevents the
    policy from earning meaningful reward for drawing a beautiful human-like path
    that never arrives at the destination.
    """

    path_weight: float = 0.40
    speed_weight: float = 0.25
    turn_weight: float = 0.20
    efficiency_weight: float = 0.15

    def __post_init__(self) -> None:
        weights = {
            "path_weight": self.path_weight,
            "speed_weight": self.speed_weight,
            "turn_weight": self.turn_weight,
            "efficiency_weight": self.efficiency_weight,
        }
        for name, value in weights.items():
            if value < 0:
                raise ValueError(f"{name} cannot be negative")
        if sum(weights.values()) <= 0:
            raise ValueError("at least one imitation reward weight must be positive")

    def __call__(
        self,
        rollout_points: np.ndarray,
        rollout_speeds: np.ndarray,
        demo_path: np.ndarray,
        demo_speed_profile: np.ndarray,
        start: np.ndarray,
        destination: np.ndarray,
    ) -> float:
        n = len(demo_path)
        if len(rollout_points) < 2:
            return -1.0

        rollout_path = resample_polyline(rollout_points, n)
        rollout_local = to_local_frame(rollout_path, start, destination)
        demo_local = to_local_frame(demo_path, start, destination)

        path_error = float(np.mean(np.linalg.norm(rollout_local - demo_local, axis=1)))
        path_score = float(np.exp(-4.0 * path_error))

        rollout_speed_profile = np.interp(
            np.linspace(0, 1, n),
            np.linspace(0, 1, max(len(rollout_speeds), 1)),
            rollout_speeds if len(rollout_speeds) else np.zeros(1),
        )
        max_speed = float(np.max(rollout_speed_profile))
        if max_speed > 0:
            rollout_speed_profile = rollout_speed_profile / max_speed
        speed_error = float(np.mean(np.abs(rollout_speed_profile - demo_speed_profile)))
        speed_score = float(np.exp(-3.0 * speed_error))

        turn_error = float(
            np.mean(np.abs(curvature(rollout_local) - curvature(demo_local)))
        )
        turn_score = float(np.exp(-2.0 * turn_error))

        direct_distance = max(float(np.linalg.norm(destination - start)), 1.0)
        traveled_distance = float(
            np.sum(np.linalg.norm(np.diff(rollout_points, axis=0), axis=1))
        )
        efficiency_score = float(
            np.clip(direct_distance / max(traveled_distance, 1.0), 0.0, 1.0)
        )

        return (
            self.path_weight * path_score
            + self.speed_weight * speed_score
            + self.turn_weight * turn_score
            + self.efficiency_weight * efficiency_score
        )
