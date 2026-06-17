from __future__ import annotations

import platform
import random
import string
import time
from dataclasses import dataclass
from typing import Callable, Iterable

from pynput.keyboard import Controller, Key

DEFAULT_CPM = 400.0
AVERAGE_CHARS_PER_WORD = 5.0
MIN_TYPING_SPEED_PERCENT = 1.0

KEYBOARD_LAYOUT = [
    "`1234567890-=",
    " qwertyuiop[]\\",
    " asdfghjkl;'",
    " zxcvbnm,./",
    " ",
]
SHIFT_CHAR_MAP = {
    "a": "A",
    "b": "B",
    "c": "C",
    "d": "D",
    "e": "E",
    "f": "F",
    "g": "G",
    "h": "H",
    "i": "I",
    "j": "J",
    "k": "K",
    "l": "L",
    "m": "M",
    "n": "N",
    "o": "O",
    "p": "P",
    "q": "Q",
    "r": "R",
    "s": "S",
    "t": "T",
    "u": "U",
    "v": "V",
    "w": "W",
    "x": "X",
    "y": "Y",
    "z": "Z",
    "1": "!",
    "2": "@",
    "3": "#",
    "4": "$",
    "5": "%",
    "6": "^",
    "7": "&",
    "8": "*",
    "9": "(",
    "0": ")",
    "-": "_",
    "=": "+",
    "[": "{",
    "]": "}",
    "\\": "|",
    ";": ":",
    "'": '"',
    ",": "<",
    ".": ">",
    "/": "?",
}
SHIFTED_CHARS = set(SHIFT_CHAR_MAP.values())
SHIFT_BASE_MAP = {shifted: base for base, shifted in SHIFT_CHAR_MAP.items()}
ESCAPE_CHARACTER_MAP = {
    "\b": Key.backspace,
    "\n": Key.enter,
    "\r": Key.enter,
    "\t": Key.tab,
    " ": Key.space,
}
KEY_ALIASES = {
    "alt": Key.alt,
    "alt_l": Key.alt_l,
    "alt_left": Key.alt_l,
    "alt_r": Key.alt_r,
    "alt_right": Key.alt_r,
    "backspace": Key.backspace,
    "caps_lock": Key.caps_lock,
    "cmd": Key.cmd,
    "command": Key.cmd,
    "ctrl": Key.ctrl,
    "control": Key.ctrl,
    "ctrl_l": Key.ctrl_l,
    "ctrl_left": Key.ctrl_l,
    "ctrl_r": Key.ctrl_r,
    "ctrl_right": Key.ctrl_r,
    "del": Key.delete,
    "delete": Key.delete,
    "down": Key.down,
    "end": Key.end,
    "enter": Key.enter,
    "esc": Key.esc,
    "escape": Key.esc,
    "home": Key.home,
    "left": Key.left,
    "option": Key.alt,
    "page_down": Key.page_down,
    "pagedown": Key.page_down,
    "page_up": Key.page_up,
    "pageup": Key.page_up,
    "return": Key.enter,
    "right": Key.right,
    "shift": Key.shift,
    "shift_l": Key.shift_l,
    "shift_left": Key.shift_l,
    "shift_r": Key.shift_r,
    "shift_right": Key.shift_r,
    "space": Key.space,
    "tab": Key.tab,
    "up": Key.up,
}
TYPO_MAP = {
    "q": ["w", "a", "1", "2"],
    "w": ["q", "e", "s", "3", "2"],
    "e": ["w", "r", "d", "4", "3"],
    "r": ["e", "t", "f", "5", "4"],
    "t": ["r", "y", "g", "6", "5"],
    "y": ["t", "u", "h", "7", "6"],
    "u": ["y", "i", "j", "8", "7"],
    "i": ["u", "o", "k", "9", "8"],
    "o": ["i", "p", "l", "0", "9"],
    "p": ["o", "[", ";", "-", "0"],
    "[": ["p", "]"],
    "]": ["[", "\\"],
    "\\": ["]"],
    "a": ["q", "w", "s", "z"],
    "s": ["q", "w", "e", "a", "d", "x", "z"],
    "d": ["w", "e", "r", "s", "f", "c", "x"],
    "f": ["e", "r", "t", "d", "g", "v", "c"],
    "g": ["r", "t", "y", "f", "h", "b", "v"],
    "h": ["t", "y", "u", "g", "j", "n", "b"],
    "j": ["y", "u", "i", "h", "k", "m", "n"],
    "k": ["u", "i", "o", "j", "l", ",", "m"],
    "l": ["i", "o", "p", "k", ";", ".", ","],
    ";": ["o", "p", "[", "l", "'", "/", "."],
    "'": ["[", ";", "/"],
    "z": ["a", "s", "x"],
    "x": ["s", "d", "c", "z"],
    "c": ["d", "f", "v", "x"],
    "v": ["f", "g", "b", "c"],
    "b": ["g", "h", "n", "v"],
    "n": ["h", "j", "m", "b"],
    "m": ["j", "k", ",", "n"],
    ",": ["k", "l", ".", "m"],
    ".": ["l", ";", "/", ","],
    "/": [";", "'", "."],
    "1": ["`", "2", "q"],
    "2": ["1", "3", "q", "w"],
    "3": ["2", "4", "w", "e"],
    "4": ["3", "5", "e", "r"],
    "5": ["4", "6", "r", "t"],
    "6": ["5", "7", "t", "y"],
    "7": ["6", "8", "y", "u"],
    "8": ["7", "9", "u", "i"],
    "9": ["8", "0", "i", "o"],
    "0": ["9", "-", "o", "p"],
    "-": ["0", "=", "p", "["],
    "=": ["-", "["],
    "`": ["1", "q"],
    "~": ["!", "@"],
    "!": ["@", "1"],
    "@": ["#", "2"],
    "#": ["$", "3"],
    "$": ["%", "4"],
    "%": ["^", "5"],
    "^": ["&", "6"],
    "&": ["*", "7"],
    "*": ["(", "8"],
    "(": [")", "9"],
    ")": ["_", "0"],
    "_": ["+", "-"],
    "+": ["=", "_"],
    "<": [","],
    ">": ["."],
    "?": ["/"],
    ":": [";"],
    '"': ["'"],
    "{": ["["],
    "}": ["]"],
    "|": ["\\"],
    " ": ["c", "v", "b", "n", "m"],
}


@dataclass(frozen=True)
class KeyboardProfile:
    typing_speed: float = 100
    consistency: float = 95
    typo_rate: float = 5
    correct_typos: bool = True
    pause_at_punctuation: bool = True


KEYBOARD_PROFILES = {
    "default": KeyboardProfile(),
    "fast": KeyboardProfile(typing_speed=160, consistency=88, typo_rate=2),
    "careful": KeyboardProfile(typing_speed=75, consistency=98, typo_rate=0.5),
    "messy": KeyboardProfile(typing_speed=105, consistency=65, typo_rate=12),
    "developer": KeyboardProfile(typing_speed=120, consistency=94, typo_rate=1),
}

__all__ = ["Keyboard", "KeyboardProfile", "KEYBOARD_PROFILES"]


class Keyboard:
    def __init__(
        self,
        typing_speed: int | float = 100,
        consistency: int | float = 95,
        typo_rate: int | float = 5,
        *,
        controller: Controller | None = None,
        rng: random.Random | None = None,
        sleep: Callable[[float], None] = time.sleep,
        correct_typos: bool = True,
        pause_at_punctuation: bool = True,
        primary_modifier: str | Key | None = None,
    ):
        self._keyboard_controller = controller or Controller()
        self._rng = rng or random.Random()
        self._sleep = sleep
        self._last_typed_char: str | None = None
        self._key_map = self._build_key_map()
        self._correct_typos = bool(correct_typos)
        self._pause_at_punctuation = bool(pause_at_punctuation)
        self._primary_modifier = (
            self._resolve_key(primary_modifier)
            if primary_modifier is not None
            else self._default_primary_modifier()
        )

        self._typing_speed = 1.0
        self._consistency = 0.95
        self._typo_rate = 0.05
        self.set_speed(typing_speed)
        self.set_consistency(consistency)
        self.set_typo_rate(typo_rate)

    @property
    def typing_speed(self) -> float:
        """Typing speed as a percentage of Bumblebee's default speed."""

        return self._typing_speed * 100

    @property
    def consistency(self) -> float:
        """Typing consistency as a percentage. Higher means less timing variance."""

        return self._consistency * 100

    @property
    def typo_rate(self) -> float:
        """Typo rate as a percentage."""

        return self._typo_rate * 100

    @property
    def correct_typos(self) -> bool:
        return self._correct_typos

    @property
    def pause_at_punctuation(self) -> bool:
        return self._pause_at_punctuation

    @property
    def primary_modifier(self) -> Key | str:
        """Primary shortcut modifier: Command on macOS, Control elsewhere."""

        return self._primary_modifier

    @staticmethod
    def _default_primary_modifier() -> Key:
        return Key.cmd if platform.system() == "Darwin" else Key.ctrl

    @staticmethod
    def _validate_number(name: str, value: int | float) -> float:
        if not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be an integer or float")
        return float(value)

    @staticmethod
    def _validate_percent(name: str, value: int | float) -> float:
        value = Keyboard._validate_number(name, value)
        if not 0 <= value <= 100:
            raise ValueError(f"{name} must be between 0 and 100")
        return value

    @staticmethod
    def _validate_times(times: int) -> int:
        if not isinstance(times, int):
            raise TypeError("times must be an integer")
        if times < 1:
            raise ValueError("times must be at least 1")
        return times

    @staticmethod
    def _build_key_map() -> dict[str, tuple[int, int]]:
        key_positions: dict[str, tuple[int, int]] = {}
        for row, keys in enumerate(KEYBOARD_LAYOUT):
            for col, key in enumerate(keys):
                key_positions[key] = (row, col)

        for base_char, shifted_char in SHIFT_CHAR_MAP.items():
            if base_char in key_positions:
                key_positions[shifted_char] = key_positions[base_char]

        return key_positions

    def _resolve_key(self, key: str | Key) -> str | Key:
        if isinstance(key, Key):
            return key
        if not isinstance(key, str):
            raise TypeError("key must be a string or pynput.keyboard.Key")
        if key in ESCAPE_CHARACTER_MAP:
            return ESCAPE_CHARACTER_MAP[key]
        if len(key) == 1:
            return key

        normalized = key.lower().replace("-", "_").replace(" ", "_")
        if normalized in KEY_ALIASES:
            return KEY_ALIASES[normalized]
        if hasattr(Key, normalized):
            return getattr(Key, normalized)
        raise ValueError(f"unknown key: {key}")

    def _calculate_keys_distance(self, char1: str | None, char2: str | None) -> float:
        if char1 is None or char2 is None:
            return 0.0
        pos1 = self._key_map.get(char1)
        pos2 = self._key_map.get(char2)
        if pos1 is None or pos2 is None:
            return 0.0
        return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5

    @staticmethod
    def _is_shift_needed(char: str) -> bool:
        return char.isupper() or char in SHIFTED_CHARS

    def _get_realistic_typo(self, char: str) -> str | None:
        if char in TYPO_MAP:
            return self._rng.choice(TYPO_MAP[char])
        lower_char = char.lower()
        typo_options = TYPO_MAP.get(lower_char)
        if not typo_options:
            return None
        typo = self._rng.choice(typo_options)
        return typo.upper() if char.isupper() and typo.isalpha() else typo

    def _timing_variation(self) -> float:
        # 100% consistency means no timing jitter. 0% means up to +/-50% jitter.
        jitter = 1.0 - self._consistency
        return self._rng.uniform(1.0 - (0.5 * jitter), 1.0 + (0.5 * jitter))

    def _context_pause(self, char: str, *, pause_at_punctuation: bool) -> float:
        if not pause_at_punctuation:
            return 0.0
        if char in ".?!":
            return self._rng.uniform(0.08, 0.25)
        if char in ",;:":
            return self._rng.uniform(0.03, 0.12)
        if char == "\n":
            return self._rng.uniform(0.08, 0.22)
        if char == " " and self._last_typed_char in ".?!":
            return self._rng.uniform(0.06, 0.18)
        if char == " " and self._last_typed_char in ",;:":
            return self._rng.uniform(0.02, 0.08)
        return 0.0

    def _char_complexity(self, char: str) -> float:
        if char in ESCAPE_CHARACTER_MAP:
            return 1.0
        if self._is_shift_needed(char):
            return 1.25
        if char in string.punctuation:
            return 1.15
        return 1.0

    def _delay_after_char(
        self,
        char: str,
        *,
        speed_multiplier: float,
        pause_at_punctuation: bool,
    ) -> float:
        base_delay = 60.0 / DEFAULT_CPM
        distance = self._calculate_keys_distance(self._last_typed_char, char)
        distance_factor = distance * 0.05
        delay = (
            base_delay
            * self._char_complexity(char)
            * (1.0 + distance_factor)
            * self._timing_variation()
        )
        delay += self._context_pause(char, pause_at_punctuation=pause_at_punctuation)
        return max(0.0, delay / speed_multiplier)

    def _type_char(
        self,
        char: str,
        *,
        speed_multiplier: float | None = None,
        pause_at_punctuation: bool | None = None,
    ) -> None:
        if not isinstance(char, str) or len(char) != 1:
            raise ValueError("char must be a single-character string")

        speed_multiplier = speed_multiplier or self._typing_speed
        pause_at_punctuation = (
            self._pause_at_punctuation
            if pause_at_punctuation is None
            else pause_at_punctuation
        )
        press_duration = self._rng.uniform(0.025, 0.09) / speed_multiplier
        base_char = SHIFT_BASE_MAP.get(char)
        if base_char is not None:
            self._keyboard_controller.press(Key.shift)
            self._sleep(self._rng.uniform(0.01, 0.03) / speed_multiplier)
            self._keyboard_controller.press(base_char)
            self._sleep(press_duration)
            self._keyboard_controller.release(base_char)
            self._keyboard_controller.release(Key.shift)
        else:
            key = ESCAPE_CHARACTER_MAP.get(char, char)
            self._keyboard_controller.press(key)
            self._sleep(press_duration)
            self._keyboard_controller.release(key)

        delay = self._delay_after_char(
            char,
            speed_multiplier=speed_multiplier,
            pause_at_punctuation=pause_at_punctuation,
        )
        self._last_typed_char = char
        self._sleep(delay)

    def set_typo_rate(self, new_typo_rate: int | float) -> None:
        """Set typo rate as a percentage from 0 to 100."""

        self._typo_rate = self._validate_percent("new_typo_rate", new_typo_rate) / 100

    def set_consistency(self, new_consistency: int | float) -> None:
        """Set consistency as a percentage from 0 to 100."""

        self._consistency = (
            self._validate_percent("new_consistency", new_consistency) / 100
        )

    def set_speed(self, new_speed: int | float) -> None:
        """Set typing speed as a percentage of Bumblebee's default speed."""

        speed = self._validate_number("new_speed", new_speed)
        if speed < MIN_TYPING_SPEED_PERCENT:
            raise ValueError(f"new_speed must be at least {MIN_TYPING_SPEED_PERCENT}")
        self._typing_speed = speed / 100

    def set_wpm(self, words_per_minute: int | float) -> None:
        """Set typing speed using words per minute."""

        wpm = self._validate_number("words_per_minute", words_per_minute)
        if wpm <= 0:
            raise ValueError("words_per_minute must be positive")
        self._typing_speed = (wpm * AVERAGE_CHARS_PER_WORD) / DEFAULT_CPM

    def set_correct_typos(self, enabled: bool) -> None:
        self._correct_typos = bool(enabled)

    def set_pause_at_punctuation(self, enabled: bool) -> None:
        self._pause_at_punctuation = bool(enabled)

    def set_primary_modifier(self, key: str | Key) -> None:
        self._primary_modifier = self._resolve_key(key)

    def set_profile(self, name: str) -> None:
        """Apply one of: default, fast, careful, messy, developer."""

        try:
            profile = KEYBOARD_PROFILES[name]
        except KeyError as exc:
            available = ", ".join(sorted(KEYBOARD_PROFILES))
            raise ValueError(
                f"unknown profile {name!r}. Available profiles: {available}"
            ) from exc

        self.set_speed(profile.typing_speed)
        self.set_consistency(profile.consistency)
        self.set_typo_rate(profile.typo_rate)
        self.set_correct_typos(profile.correct_typos)
        self.set_pause_at_punctuation(profile.pause_at_punctuation)

    def press(self, key: str | Key) -> None:
        """Press and hold a key."""

        self._keyboard_controller.press(self._resolve_key(key))

    def release(self, key: str | Key) -> None:
        """Release a key."""

        self._keyboard_controller.release(self._resolve_key(key))

    def tap(
        self,
        key: str | Key,
        *,
        times: int = 1,
        interval: float | None = None,
        hold_seconds: float | None = None,
    ) -> None:
        """Press and release a key one or more times."""

        times = self._validate_times(times)
        resolved_key = self._resolve_key(key)
        interval = 0.03 if interval is None else max(0.0, float(interval))
        hold_seconds = 0.02 if hold_seconds is None else max(0.0, float(hold_seconds))

        for index in range(times):
            self._keyboard_controller.press(resolved_key)
            self._sleep(hold_seconds)
            self._keyboard_controller.release(resolved_key)
            if index < times - 1:
                self._sleep(interval)

    def hotkey(self, *keys: str | Key, hold_seconds: float = 0.03) -> None:
        """Press keys together, then release them in reverse order."""

        if len(keys) == 1 and isinstance(keys[0], (list, tuple)):
            keys = tuple(keys[0])
        if not keys:
            raise ValueError("hotkey requires at least one key")

        pressed: list[str | Key] = []
        try:
            for key in keys:
                resolved_key = self._resolve_key(key)
                self._keyboard_controller.press(resolved_key)
                pressed.append(resolved_key)
            self._sleep(max(0.0, float(hold_seconds)))
        finally:
            for key in reversed(pressed):
                self._keyboard_controller.release(key)

    def enter(self, times: int = 1) -> None:
        self.tap(Key.enter, times=times)

    def tab(self, times: int = 1) -> None:
        self.tap(Key.tab, times=times)

    def escape(self, times: int = 1) -> None:
        self.tap(Key.esc, times=times)

    def backspace(self, times: int = 1) -> None:
        self.tap(Key.backspace, times=times)

    def delete(self, times: int = 1) -> None:
        self.tap(Key.delete, times=times)

    def space(self, times: int = 1) -> None:
        self.tap(Key.space, times=times)

    def arrow(self, direction: str, times: int = 1) -> None:
        direction_key = direction.lower()
        if direction_key not in {"up", "down", "left", "right"}:
            raise ValueError("direction must be one of: up, down, left, right")
        self.tap(direction_key, times=times)

    def up(self, times: int = 1) -> None:
        self.arrow("up", times=times)

    def down(self, times: int = 1) -> None:
        self.arrow("down", times=times)

    def left(self, times: int = 1) -> None:
        self.arrow("left", times=times)

    def right(self, times: int = 1) -> None:
        self.arrow("right", times=times)

    def select_all(self) -> None:
        self.hotkey(self._primary_modifier, "a")

    def copy(self) -> None:
        self.hotkey(self._primary_modifier, "c")

    def cut(self) -> None:
        self.hotkey(self._primary_modifier, "x")

    def paste(self) -> None:
        self.hotkey(self._primary_modifier, "v")

    def undo(self) -> None:
        self.hotkey(self._primary_modifier, "z")

    def redo(self) -> None:
        if platform.system() == "Darwin":
            self.hotkey(self._primary_modifier, Key.shift, "z")
        else:
            self.hotkey(self._primary_modifier, "y")

    def clear(self) -> None:
        self.select_all()
        self.backspace()

    def delete_word(self) -> None:
        modifier = Key.alt if platform.system() == "Darwin" else Key.ctrl
        self.hotkey(modifier, Key.backspace)

    def move_to_line_start(self) -> None:
        if platform.system() == "Darwin":
            self.hotkey(Key.cmd, Key.left)
        else:
            self.tap(Key.home)

    def move_to_line_end(self) -> None:
        if platform.system() == "Darwin":
            self.hotkey(Key.cmd, Key.right)
        else:
            self.tap(Key.end)

    def _speed_multiplier_from_wpm(self, wpm: int | float | None) -> float:
        if wpm is None:
            return self._typing_speed
        wpm = self._validate_number("wpm", wpm)
        if wpm <= 0:
            raise ValueError("wpm must be positive")
        return (wpm * AVERAGE_CHARS_PER_WORD) / DEFAULT_CPM

    def type(
        self,
        text: str,
        *,
        wpm: int | float | None = None,
        typo_rate: int | float | None = None,
        correct_typos: bool | None = None,
        pause_at_punctuation: bool | None = None,
    ) -> None:
        """Type text with human-like rhythm, pauses, and optional corrected typos."""

        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not text:
            return

        speed_multiplier = self._speed_multiplier_from_wpm(wpm)
        typo_probability = (
            self._typo_rate
            if typo_rate is None
            else self._validate_percent("typo_rate", typo_rate) / 100
        )
        should_correct_typos = (
            self._correct_typos if correct_typos is None else bool(correct_typos)
        )
        should_pause_at_punctuation = (
            self._pause_at_punctuation
            if pause_at_punctuation is None
            else bool(pause_at_punctuation)
        )

        for char in text:
            typo_char = None
            if self._rng.random() < typo_probability:
                typo_char = self._get_realistic_typo(char)

            if typo_char and should_correct_typos:
                self._type_char(
                    typo_char,
                    speed_multiplier=speed_multiplier,
                    pause_at_punctuation=should_pause_at_punctuation,
                )
                self._sleep(self._rng.uniform(0.05, 0.15) / speed_multiplier)
                self._type_char(
                    "\b",
                    speed_multiplier=speed_multiplier,
                    pause_at_punctuation=False,
                )
            elif typo_char:
                char = typo_char

            self._type_char(
                char,
                speed_multiplier=speed_multiplier,
                pause_at_punctuation=should_pause_at_punctuation,
            )

    def write(self, text: str, **kwargs) -> None:
        """Alias for :meth:`type`."""

        self.type(text, **kwargs)

    def type_lines(
        self,
        lines: Iterable[str],
        *,
        submit: bool = False,
        **type_kwargs,
    ) -> None:
        """Type multiple lines, inserting Enter between them."""

        lines = list(lines)
        for index, line in enumerate(lines):
            self.type(line, **type_kwargs)
            if index < len(lines) - 1 or submit:
                self.enter()

    def paste_text(
        self,
        text: str,
        *,
        restore_clipboard: bool = True,
        restore_delay: float = 0.05,
    ) -> None:
        """Copy text to the clipboard and paste it with the primary modifier."""

        if not isinstance(text, str):
            raise TypeError("text must be a string")
        try:
            import pyperclip
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "paste_text requires pyperclip. Install Bumblebee with runtime dependencies."
            ) from exc

        previous_clipboard = pyperclip.paste() if restore_clipboard else None
        pyperclip.copy(text)
        self._sleep(0.03)
        self.paste()
        self._sleep(max(0.0, float(restore_delay)))
        if restore_clipboard:
            pyperclip.copy(previous_clipboard)

    def type_or_paste(self, text: str, *, threshold: int = 500, **type_kwargs) -> None:
        """Type short text and paste long text."""

        if not isinstance(threshold, int):
            raise TypeError("threshold must be an integer")
        if threshold < 0:
            raise ValueError("threshold cannot be negative")
        if len(text) >= threshold:
            self.paste_text(text)
        else:
            self.type(text, **type_kwargs)
