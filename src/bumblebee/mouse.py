from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, replace
from numbers import Real
from typing import Any, Callable

import numpy as np
import pyautogui

_VALID_BUTTONS = {"left", "middle", "right", "primary", "secondary"}


def _finite_number(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _validate_range(name: str, value: tuple[float, float]) -> None:
    if len(value) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    start = _finite_number(f"{name}[0]", value[0])
    end = _finite_number(f"{name}[1]", value[1])
    if start < 0 or end < 0:
        raise ValueError(f"{name} cannot contain negative values")
    if end < start:
        raise ValueError(f"{name} end must be >= start")


@dataclass(frozen=True)
class MouseMoveStep:
    x: int
    y: int
    move_duration: float
    sleep_duration: float


@dataclass(frozen=True)
class MouseBounds:
    """Clickable rectangle represented as left/top/right/bottom coordinates."""

    left: float
    top: float
    right: float
    bottom: float

    def __post_init__(self) -> None:
        left = _finite_number("left", self.left)
        top = _finite_number("top", self.top)
        right = _finite_number("right", self.right)
        bottom = _finite_number("bottom", self.bottom)
        if right < left:
            raise ValueError("right must be >= left")
        if bottom < top:
            raise ValueError("bottom must be >= top")
        object.__setattr__(self, "left", left)
        object.__setattr__(self, "top", top)
        object.__setattr__(self, "right", right)
        object.__setattr__(self, "bottom", bottom)

    @classmethod
    def from_xywh(
        cls, x: int | float, y: int | float, width: int | float, height: int | float
    ) -> "MouseBounds":
        x = _finite_number("x", x)
        y = _finite_number("y", y)
        width = _finite_number("width", width)
        height = _finite_number("height", height)
        if width < 0:
            raise ValueError("width must be non-negative")
        if height < 0:
            raise ValueError("height must be non-negative")
        return cls(x, y, x + width, y + height)

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.bottom - self.top

    def padded(self, padding: int | float) -> "MouseBounds":
        padding = _finite_number("padding", padding)
        if padding < 0:
            raise ValueError("padding cannot be negative")
        return MouseBounds(
            self.left + padding,
            self.top + padding,
            self.right - padding,
            self.bottom - padding,
        )

    def clamped_to_screen(self, width: int, height: int) -> "MouseBounds":
        if width <= 0 or height <= 0:
            raise ValueError("screen width and height must be positive")
        left = max(self.left, 0.0)
        right = min(self.right, width - 1)
        top = max(self.top, 0.0)
        bottom = min(self.bottom, height - 1)
        if right < left or bottom < top:
            raise ValueError("bounds do not intersect the screen")
        return MouseBounds(left, top, right, bottom)

    def sample(self, rng: random.Random) -> tuple[int, int]:
        return int(round(rng.uniform(self.left, self.right))), int(
            round(rng.uniform(self.top, self.bottom))
        )


@dataclass(frozen=True)
class MouseProfile:
    speed_px_s: float = 2000.0
    target_segment_length_px: float = 12.0
    min_path_points: int = 18
    max_path_points: int = 120
    min_segment_distance_px: float = 5.0
    max_segment_distance_px: float = 8.0
    short_move_duration_threshold: float = 0.1
    fast_segment_distance_px: float = 40.0
    fast_segment_speed_multiplier: float = 1.2
    speed_variation: float = 0.05
    speed_factor_variation: float = 0.03
    min_speed_factor: float = 0.65
    max_speed_factor: float = 1.25
    curve_strength: float = 0.0
    max_curve_px: float = 120.0
    jitter_px: float = 0.0
    target_radius_px: float = 8.0
    force_destination: bool = True
    trim_after_target_reached: bool = True
    click_pause_range: tuple[float, float] = (0.05, 0.10)
    post_move_click_pause_range: tuple[float, float] = (0.10, 1.40)

    def __post_init__(self) -> None:
        positive_fields = {
            "speed_px_s": self.speed_px_s,
            "target_segment_length_px": self.target_segment_length_px,
            "max_segment_distance_px": self.max_segment_distance_px,
            "short_move_duration_threshold": self.short_move_duration_threshold,
            "fast_segment_distance_px": self.fast_segment_distance_px,
            "fast_segment_speed_multiplier": self.fast_segment_speed_multiplier,
            "min_speed_factor": self.min_speed_factor,
            "max_speed_factor": self.max_speed_factor,
            "max_curve_px": self.max_curve_px,
            "target_radius_px": self.target_radius_px,
        }
        for name, value in positive_fields.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.min_path_points < 2:
            raise ValueError("min_path_points must be at least 2")
        if self.max_path_points < self.min_path_points:
            raise ValueError("max_path_points must be >= min_path_points")
        if self.min_segment_distance_px < 0:
            raise ValueError("min_segment_distance_px cannot be negative")
        if self.speed_variation < 0:
            raise ValueError("speed_variation cannot be negative")
        if self.speed_factor_variation < 0:
            raise ValueError("speed_factor_variation cannot be negative")
        if self.curve_strength < 0:
            raise ValueError("curve_strength cannot be negative")
        if self.jitter_px < 0:
            raise ValueError("jitter_px cannot be negative")
        if self.max_speed_factor < self.min_speed_factor:
            raise ValueError("max_speed_factor must be >= min_speed_factor")
        _validate_range("click_pause_range", self.click_pause_range)
        _validate_range("post_move_click_pause_range", self.post_move_click_pause_range)


MOUSE_PROFILES = {
    # Default intentionally avoids extra jitter/rounding. If an RL policy supplies a
    # complete path, Bumblebee executes that path directly instead of decorating it.
    "default": MouseProfile(),
    "precise": MouseProfile(speed_px_s=1200.0, speed_variation=0.02),
    "fast": MouseProfile(speed_px_s=3200.0, min_path_points=12, max_path_points=80),
    "natural": MouseProfile(curve_strength=0.06, jitter_px=1.0),
    "messy": MouseProfile(
        speed_px_s=1800.0,
        speed_variation=0.12,
        speed_factor_variation=0.08,
        curve_strength=0.12,
        jitter_px=3.0,
    ),
}

MousePathProvider = Callable[[np.ndarray, np.ndarray], np.ndarray]

__all__ = ["Mouse", "MouseBounds", "MouseMoveStep", "MouseProfile", "MOUSE_PROFILES"]


class Mouse:
    def __init__(
        self,
        speed: int | float | None = None,
        *,
        profile: str | MouseProfile = "default",
        controller: Any | None = None,
        rng: random.Random | None = None,
        sleep: Callable[[float], None] = time.sleep,
        path_provider: MousePathProvider | None = None,
        fail_safe: bool = True,
        pyautogui_pause: float = 0.001,
        minimum_duration: float = 0.0,
        configure_pyautogui: bool = True,
    ) -> None:
        self._controller = controller or pyautogui
        self._rng = rng or random.Random()
        self._sleep = sleep
        self._path_provider = path_provider
        self._profile = self._resolve_profile(profile)
        self._speed_px_s = self._profile.speed_px_s
        if speed is not None:
            self.set_speed(speed)
        if configure_pyautogui:
            self._setup_controller(
                fail_safe=fail_safe,
                pause=pyautogui_pause,
                minimum_duration=minimum_duration,
            )

    @property
    def speed(self) -> float:
        """Base mouse speed in pixels per second."""

        return self._speed_px_s

    @property
    def profile(self) -> MouseProfile:
        return self._profile

    @property
    def path_provider(self) -> MousePathProvider | None:
        return self._path_provider

    def _setup_controller(
        self, *, fail_safe: bool, pause: float, minimum_duration: float
    ) -> None:
        if hasattr(self._controller, "MINIMUM_DURATION"):
            self._controller.MINIMUM_DURATION = max(0.0, float(minimum_duration))
        if hasattr(self._controller, "PAUSE"):
            self._controller.PAUSE = max(0.0, float(pause))
        if hasattr(self._controller, "FAILSAFE"):
            self._controller.FAILSAFE = bool(fail_safe)

    @staticmethod
    def _resolve_profile(profile: str | MouseProfile) -> MouseProfile:
        if isinstance(profile, MouseProfile):
            return profile
        if not isinstance(profile, str):
            raise TypeError("profile must be a profile name or MouseProfile")
        try:
            return MOUSE_PROFILES[profile]
        except KeyError as exc:
            available = ", ".join(sorted(MOUSE_PROFILES))
            raise ValueError(
                f"unknown mouse profile {profile!r}. Available profiles: {available}"
            ) from exc

    @staticmethod
    def _validate_number(name: str, value: Any) -> float:
        return _finite_number(name, value)

    @classmethod
    def _point(cls, x: Any, y: Any) -> np.ndarray:
        return np.array(
            [cls._validate_number("x", x), cls._validate_number("y", y)],
            dtype=np.float64,
        )

    @staticmethod
    def _distance(start: np.ndarray, destination: np.ndarray) -> float:
        return float(np.linalg.norm(destination - start))

    @staticmethod
    def _validate_button(button: str) -> str:
        if not isinstance(button, str):
            raise TypeError(f"button must be a string, got {type(button).__name__}")
        if button not in _VALID_BUTTONS:
            raise ValueError(
                "Invalid button: {}. Button must be one of {}.".format(
                    button, ", ".join(sorted(_VALID_BUTTONS))
                )
            )
        return button

    @staticmethod
    def _validate_times(name: str, value: int) -> int:
        if not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value < 1:
            raise ValueError(f"{name} must be at least 1")
        return value

    def _pause_between(self, pause_range: tuple[float, float]) -> None:
        start, end = pause_range
        self._sleep(self._rng.uniform(start, end))

    def _speed_for_segment(self, distance: float) -> float:
        profile = self._profile
        speed = self._speed_px_s
        if distance >= profile.fast_segment_distance_px:
            speed *= profile.fast_segment_speed_multiplier
        if profile.speed_variation:
            speed *= self._rng.uniform(
                1.0 - profile.speed_variation,
                1.0 + profile.speed_variation,
            )
        return max(speed, 1e-6)

    def _point_count_for_distance(self, distance: float) -> int:
        profile = self._profile
        return int(
            np.clip(
                distance / profile.target_segment_length_px,
                profile.min_path_points,
                profile.max_path_points,
            )
        )

    def _speed_profile(self, point_count: int) -> np.ndarray:
        profile = self._profile
        t = np.linspace(0.0, 1.0, point_count, dtype=np.float64)
        speed_factor = 1.15 - 0.45 * np.sin(np.pi * t)
        if profile.speed_factor_variation:
            noise = np.array(
                [
                    self._rng.uniform(
                        -profile.speed_factor_variation,
                        profile.speed_factor_variation,
                    )
                    for _ in range(point_count)
                ],
                dtype=np.float64,
            )
            speed_factor += noise
        return np.clip(speed_factor, profile.min_speed_factor, profile.max_speed_factor)

    def _clip_points_to_screen(self, points: np.ndarray) -> np.ndarray:
        width, height = self.screen_size()
        clipped = points.copy()
        clipped[:, 0] = np.clip(clipped[:, 0], 0, width - 1)
        clipped[:, 1] = np.clip(clipped[:, 1], 0, height - 1)
        return clipped

    @staticmethod
    def _segment_reaches_point(
        start: np.ndarray, end: np.ndarray, target: np.ndarray, radius: float
    ) -> bool:
        segment = end - start
        length_sq = float(np.dot(segment, segment))
        if length_sq <= 1e-9:
            closest = start
        else:
            t = float(np.dot(target - start, segment) / length_sq)
            t = min(max(t, 0.0), 1.0)
            closest = start + t * segment
        return float(np.linalg.norm(closest - target)) <= radius

    def _trim_or_force_destination(
        self,
        path_points: np.ndarray,
        destination: np.ndarray,
        *,
        force_destination: bool,
        trim_after_target_reached: bool,
        target_radius_px: float,
    ) -> np.ndarray:
        if target_radius_px <= 0:
            raise ValueError("target_radius_px must be positive")

        if trim_after_target_reached:
            for index in range(1, len(path_points)):
                previous_xy = path_points[index - 1, :2]
                current_xy = path_points[index, :2]
                if self._segment_reaches_point(
                    previous_xy, current_xy, destination, target_radius_px
                ):
                    trimmed = path_points[:index].copy()
                    destination_row = np.array(
                        [[destination[0], destination[1], path_points[index, 2]]],
                        dtype=np.float64,
                    )
                    if (
                        len(trimmed) == 0
                        or self._distance(trimmed[-1, :2], destination) > 1e-9
                    ):
                        trimmed = np.concatenate([trimmed, destination_row], axis=0)
                    else:
                        trimmed[-1, :2] = destination
                    return trimmed

        if force_destination:
            if self._distance(path_points[-1, :2], destination) <= target_radius_px:
                path_points = path_points.copy()
                path_points[-1, :2] = destination
                return path_points
            destination_row = np.array(
                [[destination[0], destination[1], path_points[-1, 2]]],
                dtype=np.float64,
            )
            return np.concatenate([path_points, destination_row], axis=0)

        return path_points

    def _generate_procedural_path(
        self, start: np.ndarray, destination: np.ndarray
    ) -> np.ndarray:
        """Generate a fallback path without adding jitter/curvature by default."""

        distance = self._distance(start, destination)
        if distance <= 0:
            return np.array([[start[0], start[1], 1.0]], dtype=np.float64)

        profile = self._profile
        point_count = self._point_count_for_distance(distance)
        t = np.linspace(0.0, 1.0, point_count, dtype=np.float64)
        direction = destination - start
        unit = direction / distance
        perpendicular = np.array([-unit[1], unit[0]], dtype=np.float64)

        curve_px = min(distance * profile.curve_strength, profile.max_curve_px)
        if curve_px > 0:
            control_1 = (
                start
                + direction * self._rng.uniform(0.25, 0.40)
                + perpendicular * self._rng.uniform(-curve_px, curve_px)
            )
            control_2 = (
                start
                + direction * self._rng.uniform(0.60, 0.80)
                + perpendicular * self._rng.uniform(-curve_px, curve_px)
            )
            one_minus_t = 1.0 - t
            points = (
                (one_minus_t**3)[:, None] * start
                + (3 * one_minus_t**2 * t)[:, None] * control_1
                + (3 * one_minus_t * t**2)[:, None] * control_2
                + (t**3)[:, None] * destination
            )
        else:
            points = start + t[:, None] * direction

        if profile.jitter_px > 0:
            jitter = np.array(
                [self._rng.gauss(0.0, profile.jitter_px) for _ in range(point_count)],
                dtype=np.float64,
            )
            points += (np.sin(np.pi * t) * jitter)[:, None] * perpendicular

        points[0] = start
        points[-1] = destination
        points = self._clip_points_to_screen(points)
        return np.column_stack([points, self._speed_profile(point_count)])

    def _densify_path_points(self, path_points: np.ndarray) -> np.ndarray:
        max_distance = self._profile.max_segment_distance_px
        if max_distance <= 0 or len(path_points) < 2:
            return path_points

        dense_rows = [path_points[0]]
        for previous, current in zip(path_points[:-1], path_points[1:]):
            segment_distance = self._distance(previous[:2], current[:2])
            pieces = max(1, int(math.ceil(segment_distance / max_distance)))
            for piece_index in range(1, pieces + 1):
                alpha = piece_index / pieces
                dense_rows.append(previous + (current - previous) * alpha)
        return np.asarray(dense_rows, dtype=np.float64)

    def _normalize_path_points(
        self,
        path_points: np.ndarray,
        *,
        start: np.ndarray,
        destination: np.ndarray | None = None,
        force_destination: bool | None = None,
        trim_after_target_reached: bool | None = None,
        target_radius_px: float | None = None,
    ) -> np.ndarray:
        path_points = np.asarray(path_points, dtype=np.float64)
        if path_points.ndim != 2 or path_points.shape[1] not in {2, 3}:
            raise ValueError("path_points must have shape (N, 2) or (N, 3)")
        if len(path_points) == 0:
            raise ValueError("path_points cannot be empty")

        if path_points.shape[1] == 2:
            path_points = np.column_stack(
                [path_points, self._speed_profile(len(path_points))]
            )
        else:
            path_points = path_points.copy()
            path_points[:, 2] = np.clip(
                path_points[:, 2],
                self._profile.min_speed_factor,
                self._profile.max_speed_factor,
            )

        if not np.all(np.isfinite(path_points)):
            raise ValueError("path_points must contain only finite values")

        rows = [path_points]
        if self._distance(path_points[0, :2], start) > 1.0:
            rows.insert(0, np.array([[start[0], start[1], path_points[0, 2]]]))
        normalized = np.concatenate(rows, axis=0)
        if destination is not None:
            normalized = self._trim_or_force_destination(
                normalized,
                destination,
                force_destination=(
                    self._profile.force_destination
                    if force_destination is None
                    else force_destination
                ),
                trim_after_target_reached=(
                    self._profile.trim_after_target_reached
                    if trim_after_target_reached is None
                    else trim_after_target_reached
                ),
                target_radius_px=(
                    self._profile.target_radius_px
                    if target_radius_px is None
                    else target_radius_px
                ),
            )

        normalized[:, :2] = self._clip_points_to_screen(normalized[:, :2])
        return self._densify_path_points(normalized)

    def _path_to(self, start: np.ndarray, destination: np.ndarray) -> np.ndarray:
        if self._path_provider is None:
            return self._generate_procedural_path(start, destination)
        path = self._path_provider(start.copy(), destination.copy())
        return self._normalize_path_points(
            path,
            start=start,
            destination=destination,
        )

    def _prepare_data_for_move(self, path_points: np.ndarray) -> list[MouseMoveStep]:
        path_points = np.asarray(path_points, dtype=np.float64)
        if path_points.ndim != 2 or path_points.shape[1] != 3:
            raise ValueError("path_points must have shape (N, 3)")
        if len(path_points) < 2:
            return []

        x = path_points[:, 0]
        y = path_points[:, 1]
        speed_factor = path_points[:, 2]
        segment_lengths = np.hypot(np.diff(x), np.diff(y))

        steps: list[MouseMoveStep] = []
        accumulated_distance = 0.0
        last_kept_xy = path_points[0, :2]
        for index in range(1, len(path_points)):
            accumulated_distance += float(segment_lengths[index - 1])
            current_xy = path_points[index, :2]
            distance_from_last_kept = self._distance(last_kept_xy, current_xy)
            is_final_point = index == len(path_points) - 1
            should_keep = (
                is_final_point
                or self._profile.min_segment_distance_px <= 0
                or distance_from_last_kept >= self._profile.min_segment_distance_px
            )
            if not should_keep:
                continue

            if accumulated_distance <= 0:
                continue

            speed = self._speed_for_segment(accumulated_distance)
            time_to_move = round(
                (accumulated_distance / speed) * float(speed_factor[index]), 4
            )
            move_duration = (
                0.0
                if time_to_move < self._profile.short_move_duration_threshold
                else time_to_move
            )
            sleep_duration = max(0.0, round(time_to_move - move_duration, 6))
            steps.append(
                MouseMoveStep(
                    x=int(round(x[index])),
                    y=int(round(y[index])),
                    move_duration=move_duration,
                    sleep_duration=sleep_duration,
                )
            )
            accumulated_distance = 0.0
            last_kept_xy = current_xy

        return steps

    def set_speed(self, speed: float | int) -> None:
        """Set the base mouse cursor speed in pixels per second."""

        speed = self._validate_number("speed", speed)
        if speed <= 0:
            raise ValueError("speed must be greater than 0")
        self._speed_px_s = speed

    def set_profile(
        self, profile: str | MouseProfile, *, apply_speed: bool = True
    ) -> None:
        """Apply one of: default, precise, fast, natural, messy."""

        self._profile = self._resolve_profile(profile)
        if apply_speed:
            self._speed_px_s = self._profile.speed_px_s

    def set_path_provider(self, path_provider: MousePathProvider | None) -> None:
        """Set an optional path provider, for example an RL policy wrapper."""

        self._path_provider = path_provider

    def position(self) -> tuple[int, int]:
        x, y = self._controller.position()
        return int(x), int(y)

    def screen_size(self) -> tuple[int, int]:
        width, height = self._controller.size()
        return int(width), int(height)

    @staticmethod
    def _coerce_bounds(
        bounds: MouseBounds | tuple[float, float, float, float],
    ) -> MouseBounds:
        if isinstance(bounds, MouseBounds):
            return bounds
        try:
            left, top, right, bottom = bounds
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "bounds must be MouseBounds or a (left, top, right, bottom) tuple"
            ) from exc
        return MouseBounds(left, top, right, bottom)

    def random_point_in_bounds(
        self,
        bounds: MouseBounds | tuple[float, float, float, float],
        *,
        padding: int | float = 0,
        clamp_to_screen: bool = True,
    ) -> tuple[int, int]:
        """Choose a random point inside a clickable rectangle."""

        area = self._coerce_bounds(bounds).padded(padding)
        if clamp_to_screen:
            width, height = self.screen_size()
            area = area.clamped_to_screen(width, height)
        return area.sample(self._rng)

    def random_point_in_rect(
        self,
        x: int | float,
        y: int | float,
        width: int | float,
        height: int | float,
        *,
        padding: int | float = 0,
        clamp_to_screen: bool = True,
    ) -> tuple[int, int]:
        """Choose a random point inside an ``x, y, width, height`` rectangle."""

        return self.random_point_in_bounds(
            MouseBounds.from_xywh(x, y, width, height),
            padding=padding,
            clamp_to_screen=clamp_to_screen,
        )

    def generate_path(
        self,
        destX: int | float,
        destY: int | float,
        *,
        start: tuple[int | float, int | float] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Preview the path that would be used for ``move`` without moving."""

        start_point = (
            np.asarray(start, dtype=np.float64)
            if start is not None
            else np.asarray(self.position(), dtype=np.float64)
        )
        if start_point.shape != (2,) or not np.all(np.isfinite(start_point)):
            raise ValueError("start must contain two finite coordinates")
        destination = self._point(destX, destY)
        return self._path_to(start_point, destination)

    def move_path(
        self,
        path_points: np.ndarray,
        *,
        destination: tuple[int | float, int | float] | np.ndarray | None = None,
        force_destination: bool | None = None,
        trim_after_target_reached: bool | None = None,
        target_radius_px: float | None = None,
    ) -> None:
        """Execute a prepared path with shape ``(N, 2)`` or ``(N, 3)``.

        When ``destination`` is provided, the path is trimmed once it reaches the
        target radius and/or forced to end exactly at that destination, depending on
        the active profile and optional flags.
        """

        path_points = np.asarray(path_points, dtype=np.float64)
        if path_points.ndim != 2 or path_points.shape[1] not in {2, 3}:
            raise ValueError("path_points must have shape (N, 2) or (N, 3)")
        if len(path_points) == 0:
            return

        current = np.asarray(self.position(), dtype=np.float64)
        target = (
            self._point(destination[0], destination[1])
            if destination is not None
            else path_points[-1, :2]
        )
        path_points = self._normalize_path_points(
            path_points,
            start=current,
            destination=target,
            force_destination=force_destination,
            trim_after_target_reached=trim_after_target_reached,
            target_radius_px=target_radius_px,
        )
        for step in self._prepare_data_for_move(path_points):
            self._controller.moveTo(step.x, step.y, duration=step.move_duration)
            self._sleep(step.sleep_duration)

    def move(self, destX: int | float, destY: int | float) -> None:
        """Move the mouse cursor to the specified coordinates."""

        destination = self._point(destX, destY)
        start = np.asarray(self.position(), dtype=np.float64)
        if self._distance(start, destination) <= 0:
            return
        self.move_path(self._path_to(start, destination), destination=destination)

    def move_to(self, destX: int | float, destY: int | float) -> None:
        """Alias for :meth:`move`."""

        self.move(destX, destY)

    def move_relative(self, dx: int | float, dy: int | float) -> None:
        dx = self._validate_number("dx", dx)
        dy = self._validate_number("dy", dy)
        x, y = self.position()
        self.move(x + dx, y + dy)

    def click(
        self,
        button: Any = "left",
        *,
        clicks: int = 1,
        interval: float | None = None,
    ) -> None:
        """Simulate one or more mouse clicks at the current position."""

        button = self._validate_button(button)
        clicks = self._validate_times("clicks", clicks)
        interval = 0.0 if interval is None else max(0.0, float(interval))
        self._pause_between(self._profile.click_pause_range)
        self._controller.click(button=button, clicks=clicks, interval=interval)

    def double_click(self, button: str = "left", interval: float = 0.08) -> None:
        self.click(button=button, clicks=2, interval=interval)

    def right_click(self) -> None:
        self.click(button="right")

    def click_at(
        self,
        destX: int | float,
        destY: int | float,
        button: str = "left",
        *,
        clicks: int = 1,
        interval: float | None = None,
    ) -> None:
        self.move(destX, destY)
        self.click(button=button, clicks=clicks, interval=interval)

    def move_to_random_in_bounds(
        self,
        bounds: MouseBounds | tuple[float, float, float, float],
        *,
        padding: int | float = 0,
        clamp_to_screen: bool = True,
    ) -> tuple[int, int]:
        """Move to a random coordinate inside bounds and return that coordinate."""

        x, y = self.random_point_in_bounds(
            bounds,
            padding=padding,
            clamp_to_screen=clamp_to_screen,
        )
        self.move(x, y)
        return x, y

    def move_to_random_in_rect(
        self,
        x: int | float,
        y: int | float,
        width: int | float,
        height: int | float,
        *,
        padding: int | float = 0,
        clamp_to_screen: bool = True,
    ) -> tuple[int, int]:
        """Move to a random coordinate inside an ``x, y, width, height`` rectangle."""

        return self.move_to_random_in_bounds(
            MouseBounds.from_xywh(x, y, width, height),
            padding=padding,
            clamp_to_screen=clamp_to_screen,
        )

    def click_in_bounds(
        self,
        bounds: MouseBounds | tuple[float, float, float, float],
        button: str = "left",
        *,
        clicks: int = 1,
        interval: float | None = None,
        padding: int | float = 0,
        clamp_to_screen: bool = True,
    ) -> tuple[int, int]:
        """Click a random coordinate inside bounds and return that coordinate."""

        x, y = self.move_to_random_in_bounds(
            bounds,
            padding=padding,
            clamp_to_screen=clamp_to_screen,
        )
        self.click(button=button, clicks=clicks, interval=interval)
        return x, y

    def click_in_rect(
        self,
        x: int | float,
        y: int | float,
        width: int | float,
        height: int | float,
        button: str = "left",
        *,
        clicks: int = 1,
        interval: float | None = None,
        padding: int | float = 0,
        clamp_to_screen: bool = True,
    ) -> tuple[int, int]:
        """Click a random coordinate inside an ``x, y, width, height`` rectangle."""

        return self.click_in_bounds(
            MouseBounds.from_xywh(x, y, width, height),
            button=button,
            clicks=clicks,
            interval=interval,
            padding=padding,
            clamp_to_screen=clamp_to_screen,
        )

    def click_random_in(
        self,
        bounds: MouseBounds | tuple[float, float, float, float],
        button: str = "left",
        **kwargs: Any,
    ) -> tuple[int, int]:
        """Alias for :meth:`click_in_bounds`."""

        return self.click_in_bounds(bounds, button=button, **kwargs)

    def mouse_down(self, button: str = "left") -> None:
        self._controller.mouseDown(button=self._validate_button(button))

    def mouse_up(self, button: str = "left") -> None:
        self._controller.mouseUp(button=self._validate_button(button))

    def drag_to(
        self,
        destX: int | float,
        destY: int | float,
        button: str = "left",
        *,
        human_like: bool = True,
    ) -> None:
        """Drag the mouse to a destination."""

        button = self._validate_button(button)
        destination = self._point(destX, destY)
        if human_like:
            self.mouse_down(button)
            try:
                self.move(destination[0], destination[1])
            finally:
                self.mouse_up(button)
            return

        start = np.asarray(self.position(), dtype=np.float64)
        distance = self._distance(start, destination)
        duration = (
            distance / self._speed_px_s if distance > 0 else self._rng.uniform(0.5, 1.0)
        )
        self._pause_between(self._profile.click_pause_range)
        self._controller.dragTo(
            int(round(destination[0])),
            int(round(destination[1])),
            duration=duration,
            button=button,
        )

    def drag_relative(
        self,
        dx: int | float,
        dy: int | float,
        button: str = "left",
        *,
        human_like: bool = True,
    ) -> None:
        dx = self._validate_number("dx", dx)
        dy = self._validate_number("dy", dy)
        x, y = self.position()
        self.drag_to(x + dx, y + dy, button=button, human_like=human_like)

    def scroll(
        self,
        clicks: int,
        *,
        x: int | float | None = None,
        y: int | float | None = None,
    ) -> None:
        if not isinstance(clicks, int):
            raise TypeError("clicks must be an integer")
        if x is not None or y is not None:
            if x is None or y is None:
                raise ValueError("x and y must be provided together")
            self.move(x, y)
        self._controller.scroll(clicks)

    def move_to_and_click(self, destX, destY, button: Any = "left") -> None:
        """Move the mouse cursor to the specified coordinates and click."""

        self.move(destX, destY)
        self._pause_between(self._profile.post_move_click_pause_range)
        self.click(button)

    def move_to_and_double_click(
        self, destX: int | float, destY: int | float, button: str = "left"
    ) -> None:
        self.move(destX, destY)
        self._pause_between(self._profile.post_move_click_pause_range)
        self.double_click(button=button)

    def with_profile(self, profile: str | MouseProfile) -> "Mouse":
        """Return a shallow copy using a different movement profile."""

        new_profile = self._resolve_profile(profile)
        clone = Mouse(
            speed=new_profile.speed_px_s,
            profile=new_profile,
            controller=self._controller,
            rng=self._rng,
            sleep=self._sleep,
            path_provider=self._path_provider,
            configure_pyautogui=False,
        )
        return clone

    def configure_profile(self, **overrides: Any) -> None:
        """Update the active profile with dataclass field overrides."""

        self._profile = replace(self._profile, **overrides)
