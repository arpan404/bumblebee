import random
import time
from typing import Any

import numpy as np
import pyautogui

_VALID_BUTTONS = {"left", "middle", "right", "primary", "secondary"}
_MIN_PATH_POINTS = 18
_MAX_PATH_POINTS = 120
_TARGET_SEGMENT_LENGTH_PX = 12.0
_MIN_SEGMENT_DISTANCE_PX = 5.0
_FAST_SEGMENT_DISTANCE_PX = 40.0
_SHORT_MOVE_DURATION_THRESHOLD = 0.1


class Mouse:
    def __init__(self):
        self.__setup_pyautogui()
        self.__SPEED = 2000.0

    def __setup_pyautogui(self) -> None:
        pyautogui.MINIMUM_DURATION = 0
        pyautogui.PAUSE = 0.001
        pyautogui.FAILSAFE = False

    def __assert_valid_button(self, button: str) -> None:
        assert isinstance(button, str), "Invalid button type:{}".format(type(button))
        assert (
            button in _VALID_BUTTONS
        ), "Invalid button:{}. Button must be one of {}.".format(
            button, ", ".join(sorted(_VALID_BUTTONS))
        )

    @staticmethod
    def __calculate_distance(start: np.ndarray, destination: np.ndarray) -> float:
        start = np.asarray(start, dtype=np.float64)
        destination = np.asarray(destination, dtype=np.float64)
        return float(np.linalg.norm(destination - start))

    def __speed_for_segment(self, distance: float) -> float:
        if distance >= _FAST_SEGMENT_DISTANCE_PX:
            return self.__SPEED * random.uniform(1.0, 1.1) * 1.2
        return self.__SPEED * random.uniform(0.95, 1.05)

    def __generate_path(self, start: np.ndarray, destination: np.ndarray) -> np.ndarray:
        """Generate a smooth stochastic cursor path without a bundled model."""

        start = np.asarray(start, dtype=np.float32)
        destination = np.asarray(destination, dtype=np.float32)
        distance = self.__calculate_distance(start, destination)
        if distance <= 0:
            return np.array([[start[0], start[1], 1.0]], dtype=np.float32)

        point_count = int(
            np.clip(
                distance / _TARGET_SEGMENT_LENGTH_PX,
                _MIN_PATH_POINTS,
                _MAX_PATH_POINTS,
            )
        )
        t = np.linspace(0.0, 1.0, point_count, dtype=np.float32)
        direction = destination - start
        unit = direction / distance
        perpendicular = np.array([-unit[1], unit[0]], dtype=np.float32)

        curve_strength = min(distance * 0.18, 160.0)
        control_1 = (
            start
            + direction * random.uniform(0.20, 0.42)
            + perpendicular * random.uniform(-curve_strength, curve_strength)
        )
        control_2 = (
            start
            + direction * random.uniform(0.58, 0.85)
            + perpendicular * random.uniform(-curve_strength, curve_strength)
        )

        one_minus_t = 1.0 - t
        points = (
            (one_minus_t**3)[:, None] * start
            + (3 * one_minus_t**2 * t)[:, None] * control_1
            + (3 * one_minus_t * t**2)[:, None] * control_2
            + (t**3)[:, None] * destination
        )

        rng = np.random.default_rng()
        jitter = rng.normal(0.0, min(distance * 0.015, 8.0), size=point_count)
        points += (np.sin(np.pi * t) * jitter)[:, None] * perpendicular
        points[0] = start
        points[-1] = destination

        screen_width, screen_height = pyautogui.size()
        points[:, 0] = np.clip(points[:, 0], 0, screen_width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, screen_height - 1)

        speed_factor = 1.15 - 0.45 * np.sin(np.pi * t)
        speed_factor += rng.normal(0.0, 0.03, size=point_count)
        speed_factor = np.clip(speed_factor, 0.65, 1.25)

        return np.column_stack([points, speed_factor]).astype(np.float32)

    def __prepare_data_for_move(self, path_points: np.ndarray) -> np.ndarray:
        """Convert path points into ``x, y, move_duration, sleep_duration`` rows."""

        path_points = np.asarray(path_points, dtype=np.float64)
        if path_points.ndim != 2 or path_points.shape[1] != 3:
            raise ValueError("path_points must have shape (N, 3)")
        if len(path_points) < 2:
            return np.empty((0, 4), dtype=np.float64)

        x = path_points[:, 0]
        y = path_points[:, 1]
        speed_factor = path_points[:, 2]
        adjacent_distances = np.insert(np.hypot(np.diff(x), np.diff(y)), 0, 0.0)

        rows: list[tuple[int, int, float, float]] = []
        accumulated_distance = 0.0
        for index in range(1, len(path_points)):
            distance = float(adjacent_distances[index])
            is_final_point = index == len(path_points) - 1
            if distance < _MIN_SEGMENT_DISTANCE_PX and not is_final_point:
                accumulated_distance += distance
                continue

            movement_distance = distance + accumulated_distance
            if movement_distance <= 0:
                continue

            speed = self.__speed_for_segment(movement_distance)
            time_to_move = round(
                (movement_distance / speed) * float(speed_factor[index]), 4
            )
            move_duration = (
                0.0 if time_to_move < _SHORT_MOVE_DURATION_THRESHOLD else time_to_move
            )
            sleep_duration = max(0.0, round(time_to_move - move_duration, 6))
            rows.append((int(x[index]), int(y[index]), move_duration, sleep_duration))
            accumulated_distance = 0.0

        if not rows:
            return np.empty((0, 4), dtype=np.float64)
        return np.asarray(rows, dtype=np.float64)

    def set_speed(self, speed: float | int) -> None:
        """Set the base mouse cursor speed in pixels per second."""

        assert isinstance(speed, (float, int)), "Speed must be either float or int"
        assert speed > 0, "Speed must be greater than 0"
        self.__SPEED = float(speed)

    def click(self, button: Any = "left") -> None:
        """Simulate a mouse click at the current position."""

        self.__assert_valid_button(button)
        time.sleep(random.uniform(0.05, 0.1))
        pyautogui.click(button=button)

    def move(self, destX, destY) -> None:
        """Move the mouse cursor to the specified coordinates."""

        currentX, currentY = pyautogui.position()
        start = np.array([currentX, currentY], dtype=np.float32)
        destination = np.array([destX, destY], dtype=np.float32)
        if np.array_equal(start, destination):
            return

        path_points = self.__generate_path(start, destination)
        path_data = self.__prepare_data_for_move(path_points)

        for x, y, move_duration, sleep_duration in path_data:
            pyautogui.moveTo(int(x), int(y), duration=float(move_duration))
            time.sleep(float(sleep_duration))

    def drag_to(self, destX, destY, button: str = "left") -> None:
        """Drag the mouse to a destination using distance-based duration."""

        self.__assert_valid_button(button)
        current_coordinates = np.array(pyautogui.position())
        dest_coordinates = np.array([destX, destY])
        distance = self.__calculate_distance(current_coordinates, dest_coordinates)

        time.sleep(random.uniform(0.05, 0.1))
        duration = (
            distance / self.__SPEED
            if not np.array_equal(current_coordinates, dest_coordinates)
            else random.uniform(0.5, 1)
        )
        pyautogui.dragTo(destX, destY, duration=duration, button=button)

    def move_to_and_click(self, destX, destY, button: Any = "left") -> None:
        """Move the mouse cursor to the specified coordinates and click."""

        self.move(destX, destY)
        time.sleep(random.uniform(0.1, 1.4))
        self.click(button)
