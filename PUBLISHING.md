# Publishing

Bumblebee publishes to PyPI from GitHub Releases using PyPI trusted publishing.

## Release checklist

```bash
uv run black src scripts
uv run isort src scripts
uv run python -m compileall scripts src
rm -rf dist
uv build
uv run twine check dist/*
```

Optional model check:

```bash
uv run python - <<'PY'
from bumblebee.models import verify_packaged_mouse_model
assert verify_packaged_mouse_model()
print("packaged model ok")
PY
```

## Create a release

1. Bump `version` in `pyproject.toml`.
2. Commit and push.
3. Tag the same commit:

   ```bash
   git tag v2.0.1
   git push origin v2.0.1
   ```

4. Publish a GitHub Release for that tag.

The release workflow builds the wheel/sdist, uploads `model.zip` and `SHA256SUMS.txt`, then publishes to PyPI.

## PyPI setup

Configure PyPI trusted publishing once:

```text
Owner: arpan404
Repository: bumblebee
Workflow: release.yml
Environment: pypi
```

No PyPI token is needed.

## Large assets

The wheel includes:

```text
bumblebee/models/sac_mouse_v2.zip
```

Datasets, checkpoints, replay buffers, and logs stay out of git and out of the wheel. Attach them to GitHub Releases only when needed.
