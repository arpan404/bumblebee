# sac_mouse_v2

Packaged default mouse policy for Bumblebee v2.0.0.

## Artifact

- File: `sac_mouse_v2.zip`
- Algorithm: Stable-Baselines3 SAC
- Policy: `MlpPolicy`
- Source run: `artifacts/rl/sac_mouse_512/best_model.zip`
- SHA256: `51900ffb4cfd724592235b8e20dfca0858f5b386543ff975650c1cc5d598bd05`

## Environment

- Observation size: 10
- Action size: 2
- Virtual screen: 4096 x 2304
- Max steps: 96
- dt: 1 / 120 seconds
- Max velocity: 6500 px/s
- Target radius: 8 px

## Training data

The model was trained on the cleaned demonstration dataset distributed separately as a release asset named `dataset.npz`.

Dataset SHA256:

```text
a2679aa55d7687d55a9b08531d33286e8fc548d9d9f1628a6fdda77a9e8f4e38  dataset.npz
```

Raw tracker JSON files are not bundled in the package or release.

## Usage

The model is bundled in the Python package, but loading it requires RL dependencies:

```bash
uv add "the-bumblebee[rl]"
```

Local checkout:

```bash
uv sync --group train
```

Example:

```python
from bumblebee import Mouse
from bumblebee.rl.policy import SB3MousePolicyPathProvider

provider = SB3MousePolicyPathProvider.from_packaged(deterministic=False)
mouse = Mouse(path_provider=provider)
mouse.move(700, 450)
```

## Limitations

- Trained on a 4096 x 2304 virtual screen configuration.
- Runtime clamps and corrects generated paths so the final cursor position reaches the requested destination.
- Quality depends on the training data distribution and the target application context.
