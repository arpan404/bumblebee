# 🐝 Bumblebee – Human-Like Mouse & Keyboard Controller

Bumblebee is a Python package for realistic mouse and keyboard automation. Runtime mouse movement is local and lightweight, keyboard typing includes human-like timing controls, and the repository also includes an RL workspace for training and visualizing mouse policies.

---

## ✨ Features

### Mouse

- Human-like cursor movement with configurable speed and timing.
- Safe default path generation without extra jitter/roundness, so complete RL-generated paths can be executed directly.
- Optional movement profiles: `default`, `precise`, `fast`, `natural`, and `messy`.
- Custom path provider support for RL policies or external trajectory generators.
- Low-level controls: mouse down/up, relative movement, scrolling, drag, click-at, double click, right click.
- Test-friendly design with injectable controller, RNG, and sleep function.

### Keyboard

- Human-like typing rhythm with configurable speed, consistency, typo rate, and punctuation pauses.
- Keyboard profiles: `default`, `fast`, `careful`, `messy`, and `developer`.
- Low-level controls: press, release, tap, and hotkey.
- Editing helpers: copy, paste, cut, undo, redo, select all, clear, delete word, arrows, line start/end.
- Clipboard-based long text input via `type_or_paste()`.
- Test-friendly design with injectable controller, RNG, and sleep function.

### RL tooling

- Clean raw mouse tracker JSON into imitation datasets.
- Train SAC policies with Stable-Baselines3.
- Visualize trained policies with an interactive Tkinter visualizer.
- RL code is isolated from runtime dependencies through the `train` dependency group.

---

## 🏗️ Architecture

```text
src/bumblebee/
├── mouse.py                 # Runtime mouse API and path execution
├── keyboard.py              # Runtime keyboard API and typing model
└── rl/
    ├── core/
    │   ├── geometry.py      # Polyline resampling, local frames, curvature
    │   ├── reward.py        # Terminal imitation reward
    │   └── types.py         # MouseTrace and trajectory signatures
    ├── data/
    │   └── dataset.py       # Tracker loading, cleaning, dataset building
    ├── envs/
    │   ├── mouse.py         # Lightweight mouse imitation environment
    │   └── gymnasium.py     # Gymnasium wrapper for SB3
    └── training/
        └── callbacks.py     # Rich training progress callback

scripts/
├── prepare_mouse_data.py
├── train_mouse_sac.py
├── run_mouse_sac_long.sh
└── visualize_mouse_policy.py
```

Runtime imports are intentionally small: mouse/keyboard control does not require training dependencies. RL dependencies live in the `train` dependency group.

---

## 📦 Installation

Install from PyPI with uv:

```bash
uv add the-bumblebee
```

For local development:

```bash
uv sync
```

For RL training tools:

```bash
uv sync --group train
```

---

## 🚀 Quick Start

### Mouse control

```python
from bumblebee import Mouse

mouse = Mouse(fail_safe=True)
mouse.set_speed(1000)

mouse.move(100, 200)
mouse.move_relative(25, -10)
mouse.click(button="left")
mouse.click_at(300, 400, clicks=2)
mouse.right_click()
mouse.scroll(-3)
mouse.drag_to(500, 500)
mouse.move_to_and_click(150, 250)
```

Use profiles for different movement styles:

```python
mouse.set_profile("precise")
mouse.move(800, 500)

fast_mouse = mouse.with_profile("fast")
fast_mouse.move(1200, 700)
```

Preview or execute a path:

```python
path = mouse.generate_path(500, 600)
mouse.move_path(path)
```

Use an RL/custom path provider. A provider receives `start` and `destination` arrays and returns either `N x 2` points or `N x 3` points where column 3 is a speed factor.

```python
import numpy as np
from bumblebee import Mouse


def policy_path(start: np.ndarray, destination: np.ndarray) -> np.ndarray:
    # Replace this with an RL policy rollout.
    return np.array([start, (start + destination) / 2, destination])


mouse = Mouse(path_provider=policy_path)
mouse.move(700, 450)
```

Bumblebee does not add extra jitter or curve decoration to provided paths.

### Keyboard control

```python
from bumblebee import Keyboard

keyboard = Keyboard()
keyboard.set_speed(180)
keyboard.set_consistency(96)
keyboard.set_typo_rate(2)

keyboard.type("Bumblebee is great.")
keyboard.type("Careful typing.", wpm=70, typo_rate=1)
```

Keyboard shortcuts and editing helpers:

```python
keyboard.hotkey("cmd", "a")      # Use "ctrl" on Windows/Linux
keyboard.copy()
keyboard.paste()
keyboard.undo()
keyboard.redo()
keyboard.backspace(times=3)
keyboard.arrow("left", times=2)
keyboard.clear()
```

Profiles and long text:

```python
keyboard.set_profile("careful")
keyboard.type_lines(["First line", "Second line"])
keyboard.type_or_paste("Long text can be pasted through the clipboard.")
```

---

## 🧠 RL Workflow

RL training artifacts are kept out of git. By default, scripts read/write under `artifacts/`.

### 1. Prepare demonstration data

Raw tracker files should be named `mouse_positions_*.json` and contain `x`, `y`, and `timestamp` entries.

```bash
uv run --group train python scripts/prepare_mouse_data.py \
  --source "~/Documents/Mouse Tracker" \
  --output artifacts/mouse_demonstrations.npz \
  --num-points 64
```

### 2. Train SAC policy

```bash
uv run --group train python scripts/train_mouse_sac.py \
  --dataset artifacts/mouse_demonstrations.npz \
  --output-dir artifacts/rl/sac_mouse \
  --preset m4-max \
  --timesteps 250000
```

Useful training options:

- `--auto-resume` resumes from the newest checkpoint.
- `--num-envs` controls parallel environments.
- `--net-arch 512 512` changes SAC MLP size for fresh runs.
- `--reward-terms` includes per-step reward term dictionaries in env info.

For long-running local training:

```bash
./scripts/run_mouse_sac_long.sh
```

### 3. Visualize trained policy

```bash
uv run --group train python scripts/visualize_mouse_policy.py \
  --model artifacts/rl/sac_mouse/best_model.zip
```

---

## 📚 API Reference

### Mouse

| Method | Purpose |
| --- | --- |
| `move(x, y)` / `move_to(x, y)` | Move to absolute coordinates. |
| `move_relative(dx, dy)` | Move relative to the current position. |
| `generate_path(x, y)` | Preview the path without moving. |
| `move_path(path)` | Execute a custom `N x 2` or `N x 3` path. |
| `click(...)`, `double_click()`, `right_click()` | Click helpers. |
| `click_at(x, y, ...)` | Move then click. |
| `mouse_down()`, `mouse_up()` | Low-level button controls. |
| `drag_to(...)`, `drag_relative(...)` | Drag controls. |
| `scroll(clicks)` | Scroll up/down. |
| `set_speed(px_per_second)` | Set base movement speed. |
| `set_profile(name)` | Apply a movement profile. |
| `set_path_provider(provider)` | Attach or replace an RL/custom path provider. |

### Keyboard

| Method | Purpose |
| --- | --- |
| `type(text, ...)` / `write(text, ...)` | Type text with human-like timing. |
| `type_lines(lines)` | Type multiple lines. |
| `type_or_paste(text)` | Type short text or paste long text. |
| `press(key)`, `release(key)`, `tap(key)` | Low-level key controls. |
| `hotkey(*keys)` | Press a shortcut combination. |
| `copy()`, `paste()`, `cut()` | Clipboard shortcuts. |
| `undo()`, `redo()`, `clear()` | Editing helpers. |
| `backspace(times)`, `delete(times)` | Deletion helpers. |
| `arrow(direction, times)` | Arrow-key movement. |
| `set_speed(percent)`, `set_wpm(wpm)` | Speed controls. |
| `set_consistency(percent)` | Timing variance control. |
| `set_typo_rate(percent)` | Typo frequency control. |
| `set_profile(name)` | Apply a typing profile. |

---

## 🛠️ Development

```bash
uv sync --group dev
uv run python -c "import bumblebee; print(bumblebee.__file__)"
uv run black src scripts
uv run isort src scripts
uv run python -m compileall scripts src
```

Runtime package code lives in `src/bumblebee`. Local datasets, checkpoints, replay buffers, TensorBoard logs, and other training artifacts should stay outside git.

---

## ⚠️ Safety Notes

- `Mouse(fail_safe=True)` keeps PyAutoGUI's corner fail-safe enabled. This is the default.
- Use automation responsibly and only on systems or applications you are allowed to control.
- Clipboard helpers temporarily replace the clipboard and restore it by default.

---

### Interested in contributing?

Please review [CONTRIBUTING.md](https://github.com/socioy/bumblebee/blob/master/CONTRIBUTING.md) for guidelines.

🐝 **Making automation feel more human, one movement at a time.**
