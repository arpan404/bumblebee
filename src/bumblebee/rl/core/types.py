from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MouseTrace:
    """A cleaned cursor movement demonstration.

    Attributes:
        points: Array with columns ``x``, ``y`` and ``timestamp``.
    """

    points: np.ndarray

    @property
    def start(self) -> np.ndarray:
        return self.points[0, :2]

    @property
    def destination(self) -> np.ndarray:
        return self.points[-1, :2]

    @property
    def duration(self) -> float:
        return float(self.points[-1, 2] - self.points[0, 2])

    @property
    def displacement(self) -> np.ndarray:
        return self.destination - self.start

    @property
    def distance(self) -> float:
        return float(np.linalg.norm(self.displacement))

    def relative_points(self) -> np.ndarray:
        """Return the path translated so the first point is at the origin."""

        return self.points[:, :2] - self.start

    def velocities(self) -> np.ndarray:
        """Return per-sample cursor velocities in pixels/second."""

        dt = np.diff(self.points[:, 2], prepend=self.points[0, 2])
        dt = np.maximum(dt, 1e-6)
        deltas = np.diff(self.points[:, :2], axis=0, prepend=self.points[:1, :2])
        return deltas / dt[:, None]

    def normalized_signature(self, num_points: int) -> tuple[np.ndarray, np.ndarray]:
        """Return normalized path shape and speed profile.

        The path is represented in a local coordinate frame where x is progress along
        the start→destination vector and y is perpendicular deviation. This allows a
        real demonstration to be reused for any sampled virtual-screen start/end pair
        while keeping stochastic human-like curvature and velocity patterns.
        """

        from .geometry import resample_polyline, to_local_frame

        sampled = resample_polyline(self.points[:, :2], num_points)
        local = to_local_frame(sampled, self.start, self.destination)
        speed = np.linalg.norm(self.velocities(), axis=1)
        elapsed = self.points[:, 2] - self.points[0, 2]
        if elapsed[-1] > 0:
            normalized_time = elapsed / elapsed[-1]
        else:
            normalized_time = np.linspace(0, 1, len(speed))
        speed_profile = np.interp(
            np.linspace(0, 1, num_points),
            normalized_time,
            speed,
        )
        max_speed = float(np.max(speed_profile))
        if max_speed > 0:
            speed_profile = speed_profile / max_speed
        return local, speed_profile
