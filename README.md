# 🐝 Bumblebee – Human-Like Mouse & Keyboard Controller

Bumblebee provides realistic, human-like control of the mouse and keyboard. Cursor movement is generated locally with smooth stochastic paths and natural timing, while keyboard input includes configurable delays and typo behavior.

---

## ✨ Features

- **Natural cursor movement:** Smooth curved paths with subtle variation, speed changes, and jitter reduction.
- **Smart keystroke simulation:** Natural delays, punctuation handling, and variability in keystroke timing.
- **RL tooling:** Utilities for cleaning mouse-tracker data, training SAC mouse policies, and visualizing trained policies.

---

## 🚀 How It Works

### 🖱️ Cursor Movement

Runtime mouse movement uses a lightweight procedural path generator. It creates a curved route from the current position to the destination, adds bounded variation, and converts the path into PyAutoGUI movement commands with natural timing.

### ⌨️ Keyboard Control

The keyboard simulation is driven by:

- **Mathematical models:** Considering distances between keys and natural typing rhythms.
- **Timing variations:** Simulated delays for each keystroke to mimic human typing quirks.
- **Punctuation & special character handling:** Adjusted behavior for realistic typing patterns.

### 🏗️ Under the Hood

- **Runtime control:** Uses [PyAutoGUI](https://pyautogui.readthedocs.io/) for mouse events and [pynput](https://pynput.readthedocs.io/) for keyboard events.
- **RL workspace:** Code under `src/bumblebee/rl` is organized into `core`, `data`, `envs`, and `training` modules. Training dependencies are kept in the `train` dependency group.

---

## ⚙️ How to Use

1. Install Bumblebee

```bash
uv add the-bumblebee
```

For a local checkout, install with:

```bash
uv sync
```

2. Import the core modules and use them as follows:

#### Mouse Control Examples

```python
from bumblebee import Mouse

mouse = Mouse()
mouse.set_speed(1000)
mouse.move(100, 200)
mouse.drag_to(233, 244)
mouse.click(button="left")
mouse.move_to_and_click(destX=150, destY=250, button="left")

# Additional controls
mouse.set_profile("precise")
mouse.move_relative(25, -10)
mouse.click_at(300, 400, clicks=2)
mouse.right_click()
mouse.scroll(-3)

# Preview or execute a custom/RL-generated path. Bumblebee does not add
# extra jitter/rounding to provided paths.
path = mouse.generate_path(500, 600)
mouse.move_path(path)
```

#### Keyboard Control Example

```python
from bumblebee import Keyboard

keyboard = Keyboard()
keyboard.set_speed(new_speed=400)
keyboard.set_typo_rate(3)
keyboard.set_consistency(99)
keyboard.type("Bumblebee is great.")

# Additional controls
keyboard.hotkey("cmd", "a")      # or "ctrl" on Windows/Linux
keyboard.copy()
keyboard.paste()
keyboard.backspace(times=3)
keyboard.arrow("left", times=2)
keyboard.set_profile("careful")
keyboard.type("Careful typing with fewer typos.", wpm=70, typo_rate=1)
keyboard.type_or_paste("Long text can be pasted through the clipboard.")
```

---

## 🛠️ Development

This repository uses [uv](https://docs.astral.sh/uv/) for environment and dependency management.

```bash
uv sync --group dev
uv run python -c "import bumblebee; print(bumblebee.__file__)"
uv run black src scripts
uv run isort src scripts
```

RL training tools require the train group:

```bash
uv sync --group train
uv run --group train python scripts/prepare_mouse_data.py
uv run --group train python scripts/train_mouse_sac.py
```

Runtime package code lives in `src/bumblebee`. Local datasets, checkpoints, and training artifacts are intentionally kept out of git.

---

### Interested in contributing?

Please review [CONTRIBUTING.md](https://github.com/socioy/bumblebee/blob/master/CONTRIBUTING.md) for guidelines.

🐝 **Making automation feel more human, one movement at a time.**
