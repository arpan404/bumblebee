import random
import string
import time

from pynput.keyboard import Controller, Key


class Keyboard:
    def __init__(
        self,
        typing_speed: int | float = 100,
        consistency: int | float = 95,
        typo_rate: int | float = 5,
    ):
        """
        Initialize the Keyboard class with typing speed and consistency.

        Args:
            typing_speed (int): The speed as percentage of what bumblebee types normally.
            consistency (int): The consistency of typing of bumblebee, should be between 0 and 100.
            typo_rate (int): The percentage of typos that bumblebee will make while typing, should be between 0 and 100.

        Attributes:
            __typing_speed (float): The typing speed as a percentage (0 to 100).
            __consistency (float): The consistency of typing as a percentage (0 to 100).
            __typo_rate (float): The typo rate as a percentage (0 to 100).
            __keyboard_controller (Controller): The keyboard controller for simulating key presses, using pynput.
            __last_typed_char (str): The last character typed for distance calculation.
            __typo (dict): A dictionary mapping characters to their possible typos.
            __key_map (dict): A dictionary mapping characters to their positions on the keyboard.
            __shift_chars_maps (dict): A dictionary mapping characters that need the shift key to be pressed.
            __escape_characters_map (dict): A dictionary mapping escape characters for the keyboard.

        """
        assert isinstance(typing_speed, int) or isinstance(
            typing_speed, float
        ), "typing_speed must be an integer or float"
        assert isinstance(consistency, int) or isinstance(
            consistency, float
        ), "consistency must be an integer or float"
        assert 0 <= consistency <= 100, "consistency must be between 0 and 100"
        assert isinstance(typo_rate, int) or isinstance(
            typo_rate, float
        ), "typo_rate must be an integer or float"
        assert 0 <= typo_rate <= 100, "typo_rate must be between 0 and 100"

        self.__typing_speed = 100
        self.__consistency = 100
        self.__typo_rate = 0
        self.set_speed(typing_speed)
        self.set_consistency(consistency)
        self.set_typo_rate(typo_rate)

        self.__keyboard_controller = Controller()

        self.__last_typed_char = (
            None  # To keep track of the last character typed for distance calculation
        )
        # order of following variables should not be changed
        self.__typo = self.__build_typo()
        self.__shift_chars_maps = self.__build_shift_needed_chars()
        self.__key_map = self.__build_key_map()
        self.__escape_characters_map = self.__build__escape_characters_map()

    def __build__escape_characters_map(self):
        """
        Build the escape characters for the keyboard.

        This method initializes a list of escape characters that are commonly used in typing.
        These characters include backspace, enter, tab, space, etc.
        """
        escape_character_map = {
            "\b": Key.backspace,  # Backspace
            "\n": Key.enter,  # Enter
            "\t": Key.tab,  # Tab
            " ": Key.space,  # Space
            "\r": Key.enter,  # Carriage return (Enter)
            "\f": Key.enter,  # Form feed (Enter)
            "\v": Key.enter,  # Vertical tab (Enter)
            "\0": Key.enter,  # Null character (Enter)
        }
        return escape_character_map

    def __build_typo(self):
        """
        Build the typo map for the keyboard.

        This method initializes the common typo map for the keyboard.
        """
        typo_map = {
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
        return typo_map

    def __build_shift_needed_chars(self):
        """
        Return the map of characters that need the shift key to be pressed.
        This method initializes a dictionary that maps characters to their shifted counterparts.
        This includes letters, numbers, and special characters that are typically accessed with the shift key.
        """
        shift_chars = {
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
        }  # characters that need shift key to be pressed

        return shift_chars

    def __build_key_map(self):
        """
        Build the key map for the keyboard.

        This method creates and returns a dictionary mapping all keys to their positions on a standard QWERTY keyboard layout.
        """

        keyboard_layout = [
            "`1234567890-=",
            " qwertyuiop[]\\",
            " asdfghjkl;'",
            " zxcvbnm,./",
            " ",  # space
        ]  # standard QWERTY keyboard layout
        key_positions = {}

        for row, keys in enumerate(
            keyboard_layout
        ):  # Set position of each key on the keyboard
            for col, key in enumerate(keys):
                key_positions[key] = (row, col)

        for (
            shift_char,
            base_char,
        ) in (
            self.__shift_chars_maps.items()
        ):  # Set the position of the shifted characters
            if base_char in key_positions:
                key_positions[shift_char] = key_positions[base_char]

        return key_positions

    def __calculate_keys_distance(self, char1: str, char2: str):
        """
        Calculate the distance between two keys on the keyboard.

        Args:
            char1 (str): The first character.
            char2 (str): The second character.

        Returns:
            float: The distance between the two keys.
        """
        assert isinstance(char1, str) and isinstance(
            char2, str
        ), "char1 and char2 must be strings"

        if char1 is None or char2 is None:
            return 0

        if char1 not in self.__key_map or char2 not in self.__key_map:
            return 0
        pos1 = self.__key_map.get(char1, (0, 0))
        pos2 = self.__key_map.get(char2, (0, 0))

        # Calculate Euclidean distance between the two keys
        distance = ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5
        return distance

    def __is_shift_needed(self, char):
        """
        Determine if a shift key is needed to type the given character.

        Args:
            char (str): The character to check.

        Returns:
            bool: True if a shift key is needed, False otherwise.
        """
        is_char_in_shift_map = char in self.__shift_chars_maps.values()

        return is_char_in_shift_map

    def __get_realistic_typo(self, char):
        """
        Generate a realistic typo for the given character.

        Args:
            char (str): The character to generate a typo for.

        Returns:
            str: The generated typo.
        """
        assert isinstance(char, str), "char must be a string"

        if char is None:
            return None

        if char in self.__typo.keys():
            # If the character has possible typos, randomly select one of them
            possible_typos = self.__typo[char]
            if possible_typos:
                # Randomly choose a typo from the list
                return random.choice(possible_typos)

    def set_typo_rate(self, new_typo_rate: int | float):
        """
        Change the typo rate for typing.

        Args:
            new_typo_rate (int or float): The new typo rate, should be between 0 and 100.
        """
        assert isinstance(new_typo_rate, int) or isinstance(
            new_typo_rate, float
        ), "new_typo_rate must be an integer or float"
        assert 0 <= new_typo_rate <= 100, "new_typo_rate must be between 0 and 100"
        self.__typo_rate = new_typo_rate / 100

    def set_consistency(self, new_consistency):
        """
        Change the consistency of typing.

        Args:
            new_consistency (int): The new consistency of typing, should be between 0 and 100.
        """
        assert isinstance(new_consistency, int) or isinstance(
            new_consistency, float
        ), "new_consistency must be an integer or float"
        assert 0 <= new_consistency <= 100, "new_consistency must be between 0 and 100"

        self.__consistency = new_consistency / 100

    def set_speed(self, new_speed):
        """
        Change the speed of typing.

        Args:
            new_speed (int): The new speed of typing, should be between 0 and 100.
        """
        assert isinstance(new_speed, int) or isinstance(
            new_speed, float
        ), "new_speed must be an integer or float"
        assert 0 <= new_speed, "new_speed must be between 0 and 100"

        self.__typing_speed = new_speed / 100

    def __type_char(self, char: str, base_delay: float):
        """
        Type a single character with human-like rhythm and pauses.
        This method calculates the delay based on character complexity, distance from the last typed character,
        and other factors to simulate realistic typing behavior.
        Args:
            char (str): The character to type.
            base_delay (float): The base delay for typing a character.
        """
        char_complexity = 1.0
        if char in string.punctuation:
            char_complexity = 1.2
            if self.__is_shift_needed(char):
                char_complexity = 1.3
        elif char.isupper():
            char_complexity = 1.25

        distance_factor = 0
        if self.__last_typed_char:
            distance = self.__calculate_keys_distance(self.__last_typed_char, char)
            distance_factor = distance * 0.05

        consistency_variation = random.uniform(
            1 - self.__consistency, 1 + self.__consistency
        )

        context_pause = 0
        if char in ".?!":
            context_pause = random.uniform(0.02, 0.1)
        elif char in ",":
            context_pause = random.uniform(0.01, 0.05)
        elif char == " " and self.__last_typed_char and self.__last_typed_char in ".!?":
            context_pause = random.uniform(0.05, 0.1)
        elif char == " " and self.__last_typed_char and self.__last_typed_char in ",":
            context_pause = random.uniform(0.01, 0.05)
        elif (
            char == " "
            and self.__last_typed_char
            and self.__last_typed_char not in ".!?,"
        ):
            context_pause = 0

        shift_delay = random.uniform(0.1, 0.3) if self.__is_shift_needed(char) else 0
        escape_character_delay = random.uniform(
            0.05, 0.15
        )  # Delay for escape characters like backspace, enter, etc.

        delay = round(
            (
                (
                    (
                        base_delay
                        * char_complexity
                        * (1 + distance_factor)
                        * consistency_variation
                    )
                )
                + context_pause
            )
            / (
                2 * self.__typing_speed
            ),  # multiply by 2 so that the delay comes average of context_pause and base_delay
            3,
        )
        if char in self.__escape_characters_map:
            self.__keyboard_controller.press(self.__escape_characters_map[char])
            time.sleep(
                escape_character_delay
                / self.__typing_speed  # For changing delay using speed set by user
            )
            self.__keyboard_controller.release(self.__escape_characters_map[char])
        else:
            self.__keyboard_controller.press(char)
            time.sleep(shift_delay / self.__typing_speed)
            self.__keyboard_controller.release(char)

        self.__last_typed_char = (
            char  # Update the last typed character for distance calculation
        )
        time.sleep(delay)  # Pause for the calculated delay

    def type(self, text: str):
        """
        Type the given text with human-like rhythm, pauses, and occasional typos.
        Simulates realistic typing behavior based on character complexity and keyboard layout.

        Args:
            text (str): The text to type.

        Raises:
            Exception: If the caps lock is on.
        """
        assert isinstance(text, str), "text must be a string"
        if not text:
            return

        for char in text:
            base_delay = (
                60 / 400
            )  # assume a base typing speed of 500 characters per minute (CPM)
            if (
                random.random() < self.__typo_rate
            ):  # Introduce a typo based on the typo rate
                typo_char = self.__get_realistic_typo(char)
                if typo_char:
                    self.__type_char(typo_char, base_delay)
                    time.sleep(
                        random.uniform(0.05, 0.15)
                        / self.__typing_speed  # divide the delay by typing speed to simulate variable delay for correction based on set typing_speed
                    )  # Pause after typing a typo to simulate correction
                    self.__type_char(
                        "\b", base_delay
                    )  # Backspace to remove the typo character

            self.__type_char(char, base_delay)  # Type the character normally
