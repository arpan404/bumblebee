from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .geometry import from_local_frame
from .types import MouseTrace


@dataclass(frozen=True)
class CleaningConfig:
    min_points: int = 8
    min_distance_px: float = 20.0
    min_duration_seconds: float = 0.05
    pause_seconds: float = 0.75
    max_jump_px: float = 500.0
    stationary_epsilon_px: float = 0.5
    max_duration_seconds: float = 8.0
    min_avg_speed_px_s: float = 50.0
    max_avg_speed_px_s: float = 8000.0
    max_instant_speed_px_s: float = 20_000.0


@dataclass(frozen=True)
class Demonstration:
    path: np.ndarray
    speed_profile: np.ndarray
    duration: float


class MouseDemonstrationDataset:
    """In-memory stochastic demonstration bank for imitation-style RL rewards."""

    def __init__(
        self, signatures: np.ndarray, speed_profiles: np.ndarray, durations: np.ndarray
    ):
        if len(signatures) == 0:
            raise ValueError("dataset must contain at least one demonstration")
        self.signatures = signatures.astype(np.float32)
        self.speed_profiles = speed_profiles.astype(np.float32)
        self.durations = durations.astype(np.float32)

    @classmethod
    def load(cls, path: str | Path) -> "MouseDemonstrationDataset":
        data = np.load(path)
        return cls(data["signatures"], data["speed_profiles"], data["durations"])

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            signatures=self.signatures,
            speed_profiles=self.speed_profiles,
            durations=self.durations,
        )

    def sample_index(self, rng: np.random.Generator) -> int:
        """Sample a demonstration id without transforming the path yet."""

        return int(rng.integers(0, len(self.signatures)))

    def get(
        self, index: int, start: np.ndarray, destination: np.ndarray
    ) -> Demonstration:
        """Transform a stored path signature to the requested coordinates."""

        path = from_local_frame(self.signatures[index], start, destination)
        return Demonstration(
            path=path,
            speed_profile=self.speed_profiles[index],
            duration=float(self.durations[index]),
        )

    def sample(
        self, start: np.ndarray, destination: np.ndarray, rng: np.random.Generator
    ) -> Demonstration:
        """Sample one real path style and transform it to the requested coordinates."""

        return self.get(self.sample_index(rng), start, destination)


def iter_tracker_files(source: str | Path) -> Iterable[Path]:
    source = Path(source).expanduser()
    yield from sorted(source.glob("mouse_positions_*.json"))


def load_tracker_file(path: str | Path) -> np.ndarray:
    """Load raw tracker JSON into an ``Nx3`` array of ``x, y, timestamp``."""

    with Path(path).open() as file:
        raw = json.load(file)

    points = np.array(
        [
            [float(item["x"]), float(item["y"]), float(item["timestamp"])]
            for item in raw
        ],
        dtype=np.float64,
    )
    if len(points) == 0:
        return points.reshape(0, 3)
    order = np.argsort(points[:, 2])
    return points[order]


def clean_and_segment(
    points: np.ndarray, config: CleaningConfig = CleaningConfig()
) -> list[MouseTrace]:
    """Split raw cursor stream into clean movement traces.

    Cuts happen on invalid timestamps, impossible jumps, long gaps, and sustained
    stationary pauses. The sustained-pause logic is important for the tracker data:
    repeated identical samples every few milliseconds should still split a movement
    once the cursor has been idle for ``pause_seconds``.
    """

    if len(points) < config.min_points:
        return []

    traces: list[MouseTrace] = []
    current: list[np.ndarray] = [points[0]]
    idle_started_at: float | None = None
    last_stationary_point = points[0]

    for previous, point in zip(points[:-1], points[1:]):
        dt = float(point[2] - previous[2])
        if dt <= 0:
            _append_trace(traces, current, config)
            current = [point]
            idle_started_at = None
            continue

        dist = float(np.linalg.norm(point[:2] - previous[:2]))
        instant_speed = dist / dt
        should_cut = (
            dt > config.pause_seconds
            or dist > config.max_jump_px
            or instant_speed > config.max_instant_speed_px_s
        )

        if should_cut:
            _append_trace(traces, current, config)
            current = [point]
            idle_started_at = None
            last_stationary_point = point
            continue

        is_stationary = dist <= config.stationary_epsilon_px
        if is_stationary:
            if idle_started_at is None:
                idle_started_at = float(previous[2])
            last_stationary_point = point
            if float(point[2]) - idle_started_at >= config.pause_seconds:
                _append_trace(traces, current, config)
                current = [point]
                idle_started_at = float(point[2])
            continue

        if idle_started_at is not None:
            # Movement resumed after a short pause. Keep the last idle coordinate as
            # the start of the resumed movement, but split if the pause was long.
            if float(previous[2]) - idle_started_at >= config.pause_seconds:
                _append_trace(traces, current, config)
                current = [last_stationary_point]
            idle_started_at = None

        current.append(point)

    _append_trace(traces, current, config)
    return traces


def _append_trace(
    traces: list[MouseTrace], points: list[np.ndarray], config: CleaningConfig
) -> None:
    if len(points) < config.min_points:
        return
    arr = np.asarray(points, dtype=np.float64)
    trace = MouseTrace(arr)
    if trace.distance < config.min_distance_px or trace.duration <= 0:
        return
    avg_speed = trace.distance / trace.duration
    if trace.duration < config.min_duration_seconds:
        return
    if trace.duration > config.max_duration_seconds:
        return
    if avg_speed < config.min_avg_speed_px_s or avg_speed > config.max_avg_speed_px_s:
        return
    traces.append(trace)


def _process_tracker_file(
    args: tuple[Path, int, CleaningConfig],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    file_path, num_points, config = args
    signatures: list[np.ndarray] = []
    speeds: list[np.ndarray] = []
    durations: list[float] = []

    for trace in clean_and_segment(load_tracker_file(file_path), config):
        signature, speed_profile = trace.normalized_signature(num_points)
        signatures.append(signature)
        speeds.append(speed_profile)
        durations.append(trace.duration)

    return np.asarray(signatures), np.asarray(speeds), np.asarray(durations)


def build_demonstration_dataset(
    source: str | Path,
    *,
    num_points: int = 64,
    config: CleaningConfig = CleaningConfig(),
    max_traces: int | None = None,
    workers: int | None = None,
) -> MouseDemonstrationDataset:
    """Build a cleaned demonstration dataset from raw tracker files.

    ``workers`` defaults to all available CPUs. Set it to ``1`` for deterministic
    single-process debugging.
    """

    files = list(iter_tracker_files(source))
    if not files:
        raise FileNotFoundError(f"no mouse_positions_*.json files found in {source}")

    if workers is None:
        workers = os.cpu_count() or 1
    workers = max(1, workers)

    jobs = [(file_path, num_points, config) for file_path in files]
    if workers == 1:
        results = [_process_tracker_file(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_process_tracker_file, jobs))

    signatures = [result[0] for result in results if len(result[0])]
    speeds = [result[1] for result in results if len(result[1])]
    durations = [result[2] for result in results if len(result[2])]

    if not signatures:
        raise ValueError("no valid mouse trajectories remained after cleaning")

    signature_arr = np.concatenate(signatures, axis=0)
    speed_arr = np.concatenate(speeds, axis=0)
    duration_arr = np.concatenate(durations, axis=0)

    if max_traces is not None:
        signature_arr = signature_arr[:max_traces]
        speed_arr = speed_arr[:max_traces]
        duration_arr = duration_arr[:max_traces]

    return MouseDemonstrationDataset(signature_arr, speed_arr, duration_arr)
