from __future__ import annotations

import numpy as np

from .geometry import curvature, resample_polyline, to_local_frame


class ImitationReward:
    """Reward path, turn/curvature, and velocity similarity to a real demo.

    This intentionally does not reward only endpoint distance. A rollout receives high
    reward when it reaches the target with a human-like path shape, stochastic turns,
    and speed pattern similar to the sampled demonstration.
    """

    def __init__(
        self,
        *,
        path_weight: float = 0.45,
        speed_weight: float = 0.30,
        turn_weight: float = 0.20,
        endpoint_weight: float = 0.05,
    ) -> None:
        self.path_weight = path_weight
        self.speed_weight = speed_weight
        self.turn_weight = turn_weight
        self.endpoint_weight = endpoint_weight

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

        endpoint_error = float(np.linalg.norm(rollout_points[-1] - destination))
        target_distance = max(float(np.linalg.norm(destination - start)), 1.0)
        endpoint_score = float(np.exp(-endpoint_error / target_distance))

        return (
            self.path_weight * path_score
            + self.speed_weight * speed_score
            + self.turn_weight * turn_score
            + self.endpoint_weight * endpoint_score
        )
