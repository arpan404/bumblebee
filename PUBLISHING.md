# Publishing Bumblebee

This document defines how Bumblebee v2 is packaged and released.

## What goes into the Python package

The wheel contains:

- runtime source code under `src/bumblebee`
- RL source code under `src/bumblebee/rl`
- the default packaged mouse policy:
  - `src/bumblebee/models/sac_mouse_v2.zip`
- model metadata:
  - `src/bumblebee/models/manifest.json`
  - `src/bumblebee/models/MODEL_CARD.md`
- documentation included in the source distribution:
  - `README.md`
  - `ARCHITECTURE.md`
  - `LICENSE.md`

The wheel does **not** contain:

- raw tracker JSON files
- cleaned training datasets such as `dataset.npz`
- checkpoints
- replay buffers
- TensorBoard logs
- run logs

Datasets and large training artifacts belong on GitHub Releases or another artifact host, not in git and not in the wheel.

## Dependency policy

Base install:

```bash
uv add the-bumblebee
```

The base package supports runtime mouse/keyboard APIs and can execute procedural/custom paths. It does not install Torch or Stable-Baselines3.

RL/model loading install:

```bash
uv add "the-bumblebee[rl]"
```

Local training install:

```bash
uv sync --group train
```

`[project.optional-dependencies].rl` is for published package users. `[dependency-groups].train` is for contributors working from the repository.

## Updating the packaged model

Use the best validated model for the release, not an arbitrary latest checkpoint.

Current packaged model location:

```text
src/bumblebee/models/sac_mouse_v2.zip
```

Recommended source artifact:

```text
artifacts/rl/sac_mouse_512/best_model.zip
```

Update steps:

```bash
uv run python scripts/package_mouse_model.py \
  --model artifacts/rl/sac_mouse_512/best_model.zip \
  --source-run artifacts/rl/sac_mouse_512/best_model.zip
```

This copies the model to `src/bumblebee/models/sac_mouse_v2.zip` and updates `src/bumblebee/models/manifest.json`.

Then update:

- `src/bumblebee/models/MODEL_CARD.md`
- release notes checksums, if the model is also attached to GitHub Releases

Verify the packaged model metadata:

```bash
uv run python - <<'PY'
from bumblebee.models import load_model_manifest, verify_packaged_mouse_model
print(load_model_manifest()["default_mouse_model"])
print(verify_packaged_mouse_model())
PY
```

## Using the packaged model

```python
from bumblebee import Mouse
from bumblebee.rl.policy import SB3MousePolicyPathProvider

provider = SB3MousePolicyPathProvider.from_packaged(deterministic=False)
mouse = Mouse(path_provider=provider)
mouse.move(700, 450)
```

This requires:

```bash
uv add "the-bumblebee[rl]"
```

or, in a checkout:

```bash
uv sync --group train
```

## Release assets

For v2.0.0, attach these release assets:

| Asset | Description |
| --- | --- |
| `model.zip` | Same model as `src/bumblebee/models/sac_mouse_v2.zip`, renamed for release asset clarity. |
| `dataset.npz` | Cleaned demonstration dataset used by the RL workflow. |

The release asset model is redundant with the wheel on purpose. It lets users inspect or use the model without installing the Python package.

The dataset remains a release asset because it is too large for the wheel.

## Build checks

Run before creating a release:

```bash
uv run black src scripts
uv run isort src scripts
uv run python -m compileall scripts src
uv run python - <<'PY'
from bumblebee.models import verify_packaged_mouse_model
assert verify_packaged_mouse_model()
print("packaged model ok")
PY
```

Run RL import/model-load smoke check:

```bash
uv run --group train python - <<'PY'
from bumblebee.rl.policy import SB3MousePolicyPathProvider
provider = SB3MousePolicyPathProvider.from_packaged(deterministic=True)
print(provider.device)
PY
```

Build package:

```bash
rm -rf dist
uv build
uv run twine check dist/*
```

Confirm the wheel contains the model:

```bash
python - <<'PY'
from pathlib import Path
from zipfile import ZipFile
wheel = next(Path("dist").glob("the_bumblebee-*.whl"))
with ZipFile(wheel) as zf:
    names = set(zf.namelist())
    assert "bumblebee/models/sac_mouse_v2.zip" in names
    assert "bumblebee/models/manifest.json" in names
    assert "bumblebee/models/MODEL_CARD.md" in names
print(f"model assets found in {wheel}")
PY
```

## Publishing to PyPI

After checks pass:

```bash
uv run twine upload dist/*
```

For TestPyPI, use:

```bash
uv run twine upload --repository testpypi dist/*
```

## GitHub release flow

1. Make sure `pyproject.toml` version matches the release tag.
2. Make sure the packaged model hash in `manifest.json` is correct.
3. Merge the release PR.
4. Tag the release commit:

   ```bash
   git tag v2.0.0
   git push origin v2.0.0
   ```

5. Create or update the GitHub release.
6. Attach release assets:

   ```bash
   cp src/bumblebee/models/sac_mouse_v2.zip /tmp/model.zip
   cp artifacts/mouse_demonstrations.npz /tmp/dataset.npz
   gh release upload v2.0.0 /tmp/model.zip /tmp/dataset.npz --clobber
   ```

7. Publish the draft release after reviewing notes and assets.

## Versioning notes

- Patch releases may update code without changing the packaged model.
- If the packaged model changes, update `manifest.json`, `MODEL_CARD.md`, and release notes.
- If model architecture or observation/action contracts change, bump at least the minor version and document migration steps.
