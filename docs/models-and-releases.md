# Models and Release Assets

Bumblebee v2 packages a default SAC mouse model in the Python wheel and distributes the training dataset separately as a GitHub release asset.

## Packaged model

Packaged location:

```text
src/bumblebee/models/sac_mouse_v2.zip
```

Installed package resource:

```text
bumblebee/models/sac_mouse_v2.zip
```

Metadata:

```text
src/bumblebee/models/manifest.json
src/bumblebee/models/MODEL_CARD.md
```

The package exposes helpers:

```python
from bumblebee.models import (
    export_packaged_mouse_model,
    load_model_manifest,
    packaged_mouse_model_file,
    packaged_mouse_model_path,
    verify_packaged_mouse_model,
)
```

## Verify packaged model

```python
from bumblebee.models import load_model_manifest, verify_packaged_mouse_model

manifest = load_model_manifest()
print(manifest["default_mouse_model"])
print(verify_packaged_mouse_model())
```

## Export packaged model

Use this when another tool needs a real file path outside the package.

```python
from bumblebee.models import export_packaged_mouse_model

path = export_packaged_mouse_model("./model.zip")
print(path)
```

## Use packaged model at runtime

Requires RL dependencies:

```bash
uv add "the-bumblebee[rl]"
```

Local checkout:

```bash
uv sync --group train
```

Then:

```python
from bumblebee import Mouse
from bumblebee.rl.policy import SB3MousePolicyPathProvider

provider = SB3MousePolicyPathProvider.from_packaged(deterministic=False)
mouse = Mouse(path_provider=provider)
mouse.move(700, 450)
```

## Release assets

GitHub Releases should include:

| Asset | Included in wheel? | Purpose |
| --- | --- | --- |
| `model.zip` | Yes, same bytes as `sac_mouse_v2.zip` | Manual inspection/use outside the package. |
| `dataset.npz` | No | Cleaned demonstration dataset for training/retraining. |

Why the model is both packaged and attached to the release:

- Packaged model makes normal app usage simple.
- Release asset lets users download the model without installing the package.
- Checksums make the relationship explicit.

Why the dataset is not packaged:

- It is large.
- It is only needed for training/retraining.
- Runtime users do not need it.

## Update packaged model

Use the helper script:

```bash
uv run python scripts/package_mouse_model.py \
  --model artifacts/rl/sac_mouse_512/best_model.zip \
  --source-run artifacts/rl/sac_mouse_512/best_model.zip
```

This updates:

```text
src/bumblebee/models/sac_mouse_v2.zip
src/bumblebee/models/manifest.json
```

Then manually review/update:

```text
src/bumblebee/models/MODEL_CARD.md
PUBLISHING.md
release notes
```

## Build verification

```bash
uv run python - <<'PY'
from bumblebee.models import verify_packaged_mouse_model
assert verify_packaged_mouse_model()
print("packaged model ok")
PY
```

Check wheel contents:

```bash
rm -rf dist
uv build
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

See also: [`../PUBLISHING.md`](../PUBLISHING.md).
