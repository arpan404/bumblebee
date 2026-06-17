# Bumblebee Documentation

Bumblebee is split into two layers:

1. **Runtime automation** for mouse and keyboard control.
2. **RL tooling** for training and using mouse trajectory policies.

The runtime package is intentionally usable without Torch or Stable-Baselines3. The packaged SAC mouse model is included in the wheel, but loading it requires the optional `rl` extra.

## Documents

- [Installation](installation.md)
- [Mouse API](mouse.md)
- [Keyboard API](keyboard.md)
- [Packaged model and release assets](models-and-releases.md)
- [RL workflow](rl-workflow.md)
- [Examples](examples.md)

## Important files

- `README.md` — project overview and quick start.
- [`architecture.md`](architecture.md) — detailed RL architecture, reward design, and data schema.
- [`../PUBLISHING.md`](../PUBLISHING.md) — packaging, release, and model update process.
- [`../SECURITY.md`](../SECURITY.md) — supported versions and vulnerability reporting.

## Quick install

Runtime only:

```bash
uv add the-bumblebee
```

Runtime plus packaged RL model support:

```bash
uv add "the-bumblebee[rl]"
```

Local checkout with training tools:

```bash
uv sync --group train
```
