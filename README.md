# Bumblebee

[![PyPI](https://img.shields.io/pypi/v/the-bumblebee.svg)](https://pypi.org/project/the-bumblebee/)
[![Python](https://img.shields.io/pypi/pyversions/the-bumblebee.svg)](https://pypi.org/project/the-bumblebee/)
[![License](https://img.shields.io/pypi/l/the-bumblebee.svg)](https://github.com/arpan404/bumblebee/blob/master/LICENSE.md)

Human-like mouse and keyboard automation for Python.

Bumblebee gives you a small runtime API for moving the mouse, clicking, dragging, scrolling, typing, shortcuts, clipboard actions, and optional RL-generated cursor paths. The base install stays lightweight; Torch and Stable-Baselines3 are only installed when you ask for RL support.

## Install

```bash
pip install the-bumblebee
```

or with uv:

```bash
uv add the-bumblebee
```

To use the packaged RL mouse policy:

```bash
pip install "the-bumblebee[rl]"
# or
uv add "the-bumblebee[rl]"
```

## Quick start

### Mouse

```python
from bumblebee import Mouse, MouseBounds

mouse = Mouse(fail_safe=True)
mouse.set_profile("natural")
mouse.set_speed(1200)

mouse.move(500, 300)
mouse.click()
mouse.drag_to(700, 450)
mouse.scroll(-3)

# Click a random safe point inside a UI rectangle.
mouse.click_in_bounds(MouseBounds(100, 200, 260, 240), padding=6)
```

### Keyboard

```python
from bumblebee import Keyboard

keyboard = Keyboard()
keyboard.set_profile("careful")
keyboard.type("Hello from Bumblebee.", wpm=70, typo_rate=1)
keyboard.hotkey("cmd", "a")  # use "ctrl" on Windows/Linux
```

### Packaged RL mouse policy

```python
from bumblebee import Mouse
from bumblebee.rl.policy import SB3MousePolicyPathProvider

provider = SB3MousePolicyPathProvider.from_packaged(deterministic=False)
mouse = Mouse(path_provider=provider, fail_safe=True)
mouse.move(700, 450)
```

## What is included

- Mouse movement, clicking, dragging, scrolling, and bounded clicks
- Keyboard typing, hotkeys, copy/paste, undo/redo, and multi-line helpers
- Configurable movement and typing profiles
- Custom mouse path providers
- Optional packaged SAC mouse policy for RL-generated paths
- Local tooling for preparing data, training, and visualizing mouse policies

## Documentation

| Topic | Link |
| --- | --- |
| Installation | [docs/installation.md](https://github.com/arpan404/bumblebee/blob/master/docs/installation.md) |
| Mouse API | [docs/mouse.md](https://github.com/arpan404/bumblebee/blob/master/docs/mouse.md) |
| Keyboard API | [docs/keyboard.md](https://github.com/arpan404/bumblebee/blob/master/docs/keyboard.md) |
| Examples | [docs/examples.md](https://github.com/arpan404/bumblebee/blob/master/docs/examples.md) |
| RL workflow | [docs/rl-workflow.md](https://github.com/arpan404/bumblebee/blob/master/docs/rl-workflow.md) |
| Architecture | [docs/architecture.md](https://github.com/arpan404/bumblebee/blob/master/docs/architecture.md) |
| Model and release assets | [docs/models-and-releases.md](https://github.com/arpan404/bumblebee/blob/master/docs/models-and-releases.md) |
| Publishing | [PUBLISHING.md](https://github.com/arpan404/bumblebee/blob/master/PUBLISHING.md) |
| Security | [SECURITY.md](https://github.com/arpan404/bumblebee/blob/master/SECURITY.md) |

## Development

```bash
git clone https://github.com/arpan404/bumblebee.git
cd bumblebee
uv sync --group dev

uv run black --check .
uv run isort --check-only .
uv run python -m compileall scripts src
uv build
uv run twine check dist/*
```

For RL training tools:

```bash
uv sync --group train
```

## Model and dataset policy

The wheel includes the default model:

```text
bumblebee/models/sac_mouse_v2.zip
```

Datasets, checkpoints, replay buffers, TensorBoard logs, and other large training artifacts are not included in the wheel. Release artifacts live on [GitHub Releases](https://github.com/arpan404/bumblebee/releases).

## Safety

- `Mouse(fail_safe=True)` is the default.
- Move the cursor to a screen corner to trigger PyAutoGUI's fail-safe.
- Only automate systems and applications you are allowed to control.
- Be careful with real clicks, typing, and clipboard operations.

On macOS, you may need to grant Accessibility permission to your terminal, Python, or IDE.

## Contributing

Contributions are welcome. Start with [CONTRIBUTING.md](https://github.com/arpan404/bumblebee/blob/master/CONTRIBUTING.md).
