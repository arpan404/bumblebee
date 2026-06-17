from __future__ import annotations

import numpy as np


def resample_polyline(points: np.ndarray, num_points: int) -> np.ndarray:
    """Resample a 2D polyline by arc length."""

    if num_points < 1:
        raise ValueError("num_points must be at least 1")

    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        raise ValueError("points cannot be empty")
    if len(points) == 1:
        return np.repeat(points, num_points, axis=0)
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total = cumulative[-1]
    if total <= 1e-9:
        return np.repeat(points[:1], num_points, axis=0)

    targets = np.linspace(0.0, total, num_points)
    x = np.interp(targets, cumulative, points[:, 0])
    y = np.interp(targets, cumulative, points[:, 1])
    return np.column_stack([x, y])


def _basis(
    start: np.ndarray, destination: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    vector = destination - start
    distance = float(np.linalg.norm(vector))
    if distance <= 1e-9:
        unit = np.array([1.0, 0.0], dtype=np.float64)
    else:
        unit = vector / distance
    perpendicular = np.array([-unit[1], unit[0]], dtype=np.float64)
    return unit, perpendicular, max(distance, 1e-9)


def to_local_frame(
    points: np.ndarray, start: np.ndarray, destination: np.ndarray
) -> np.ndarray:
    """Convert screen coordinates into normalized local trajectory coordinates."""

    unit, perpendicular, distance = _basis(start, destination)
    relative = points - start
    along = relative @ unit / distance
    lateral = relative @ perpendicular / distance
    return np.column_stack([along, lateral])


def from_local_frame(
    local: np.ndarray, start: np.ndarray, destination: np.ndarray
) -> np.ndarray:
    """Convert normalized local trajectory coordinates back to screen coordinates."""

    unit, perpendicular, distance = _basis(start, destination)
    return (
        start
        + (local[:, :1] * distance * unit)
        + (local[:, 1:2] * distance * perpendicular)
    )


def curvature(points: np.ndarray) -> np.ndarray:
    """Approximate unsigned turn angle at each point."""

    if len(points) < 3:
        return np.zeros(len(points), dtype=np.float64)
    prev_vec = points[1:-1] - points[:-2]
    next_vec = points[2:] - points[1:-1]
    prev_norm = np.linalg.norm(prev_vec, axis=1)
    next_norm = np.linalg.norm(next_vec, axis=1)
    denom = np.maximum(prev_norm * next_norm, 1e-9)
    cos_angle = np.sum(prev_vec * next_vec, axis=1) / denom
    angles = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.concatenate([[0.0], angles, [0.0]])
