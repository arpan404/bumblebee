# 🐝 Bumblebee

[![PyPI](https://img.shields.io/pypi/v/the-bumblebee.svg)](https://pypi.org/project/the-bumblebee/)
[![Python](https://img.shields.io/pypi/pyversions/the-bumblebee.svg)](https://pypi.org/project/the-bumblebee/)

Human-like mouse and keyboard automation for Python, with an optional packaged SAC mouse policy for RL-generated cursor paths.

Bumblebee v2 focuses on:

- a lightweight runtime API for mouse and keyboard automation
- configurable movement/typing profiles
- custom and RL mouse path providers
- a packaged default SAC mouse model
- local tooling for preparing data, training, and visualizing mouse policies

> Detailed usage lives in [`docs/`](docs/index.md). This README is only the quick start.

---

## Installation

Runtime only:

```bash
uv add the-bumblebee
```

Runtime plus packaged RL model loading:

```bash
uv add "the-bumblebee[rl]"
```

Local development:

```bash
git clone https://github.com/arpan404/bumblebee.git
cd bumblebee
uv sync --group dev
```

Local RL training tools:

```bash
uv sync --group train
```

---

## Quick examples

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

More: [Mouse API](docs/mouse.md)

### Keyboard

```python
from bumblebee import Keyboard

keyboard = Keyboard()
keyboard.set_profile("careful")
keyboard.type("Hello from Bumblebee.", wpm=70, typo_rate=1)
keyboard.hotkey("cmd", "a")  # use "ctrl" on Windows/Linux
```

More: [Keyboard API](docs/keyboard.md)

### Packaged RL mouse model

Install with `the-bumblebee[rl]`, then:

```python
from bumblebee import Mouse
from bumblebee.rl.policy import SB3MousePolicyPathProvider

provider = SB3MousePolicyPathProvider.from_packaged(deterministic=False)
mouse = Mouse(path_provider=provider)
mouse.move(700, 450)
```

More: [Packaged model and release assets](docs/models-and-releases.md)

---

## Documentation

| Topic | Link |
| --- | --- |
| Installation | [`docs/installation.md`](docs/installation.md) |
| Mouse API | [`docs/mouse.md`](docs/mouse.md) |
| Keyboard API | [`docs/keyboard.md`](docs/keyboard.md) |
| Examples | [`docs/examples.md`](docs/examples.md) |
| RL workflow | [`docs/rl-workflow.md`](docs/rl-workflow.md) |
| Architecture | [`docs/architecture.md`](docs/architecture.md) |
| Model/release assets | [`docs/models-and-releases.md`](docs/models-and-releases.md) |
| Publishing | [`PUBLISHING.md`](PUBLISHING.md) |
| Security | [`SECURITY.md`](SECURITY.md) |

---

## Model and dataset policy

The wheel includes the default model:

```text
bumblebee/models/sac_mouse_v2.zip
```

The dataset is **not** included in the wheel. Large datasets/checkpoints/logs should be distributed through GitHub Releases or another artifact host.

---

## Development checks

```bash
uv run black --check .
uv run isort --check-only .
uv run python -m compileall scripts src
uv build
uv run twine check dist/*
```

CI runs these checks and verifies that the packaged model is present in the wheel.

---

## Safety

- `Mouse(fail_safe=True)` is the default.
- Move the cursor to a screen corner to trigger PyAutoGUI's fail-safe.
- Only automate systems and applications you are allowed to control.
- Be careful with real clicks, typing, and clipboard operations.

---

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md).
