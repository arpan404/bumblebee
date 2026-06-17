# Installation

## Runtime-only install

Use this when you only need mouse and keyboard automation, procedural mouse paths, custom path providers, or precomputed paths.

```bash
uv add the-bumblebee
```

Runtime dependencies:

- `numpy`
- `pyautogui`
- `pynput`
- `pyperclip`

This install does not include Torch, Stable-Baselines3, or Gymnasium.

## Install with packaged RL model support

Use this when you want to load the packaged SAC mouse model with `SB3MousePolicyPathProvider`.

```bash
uv add "the-bumblebee[rl]"
```

The wheel includes the model file:

```text
bumblebee/models/sac_mouse_v2.zip
```

The `rl` extra installs the libraries needed to load and run that model.

## Local development install

```bash
git clone https://github.com/arpan404/bumblebee.git
cd bumblebee
uv sync
```

## Local RL training install

```bash
uv sync --group train
```

Use this for:

- preparing mouse demonstration datasets
- training SAC policies
- visualizing trained policies
- loading the packaged model from a checkout

## Sanity checks

Runtime import:

```bash
uv run python - <<'PY'
from bumblebee import Mouse, Keyboard
print(Mouse, Keyboard)
PY
```

Packaged model hash check:

```bash
uv run python - <<'PY'
from bumblebee.models import verify_packaged_mouse_model
print(verify_packaged_mouse_model())
PY
```

Packaged RL policy load check:

```bash
uv run --group train python - <<'PY'
from bumblebee.rl.policy import SB3MousePolicyPathProvider
provider = SB3MousePolicyPathProvider.from_packaged(deterministic=True)
print(provider.device)
PY
```

## macOS permissions

PyAutoGUI and pynput may require Accessibility permissions on macOS.

Open:

```text
System Settings -> Privacy & Security -> Accessibility
```

Then allow your terminal, Python, or IDE to control the computer.
