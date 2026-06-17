from __future__ import annotations

import hashlib
import json
import shutil
from contextlib import contextmanager
from importlib.resources import as_file, files
from pathlib import Path
from typing import Iterator

PACKAGE = __name__
MANIFEST_FILE = "manifest.json"
DEFAULT_MOUSE_MODEL = "sac_mouse_v2.zip"

__all__ = [
    "DEFAULT_MOUSE_MODEL",
    "export_packaged_mouse_model",
    "load_model_manifest",
    "packaged_mouse_model_file",
    "packaged_mouse_model_path",
    "verify_packaged_mouse_model",
]


def load_model_manifest() -> dict:
    """Load packaged model metadata."""

    manifest = files(PACKAGE).joinpath(MANIFEST_FILE)
    return json.loads(manifest.read_text())


@contextmanager
def packaged_mouse_model_file() -> Iterator[Path]:
    """Yield a filesystem path for the packaged default mouse model.

    `importlib.resources` keeps this safe even if the package is imported from a
    non-standard loader. Consumers that need a persistent path should use
    :func:`export_packaged_mouse_model` instead.
    """

    resource = files(PACKAGE).joinpath(DEFAULT_MOUSE_MODEL)
    with as_file(resource) as model_path:
        yield model_path


def packaged_mouse_model_path() -> Path:
    """Return the packaged model path for normal wheel/editable installs."""

    return Path(str(files(PACKAGE).joinpath(DEFAULT_MOUSE_MODEL)))


def export_packaged_mouse_model(destination: str | Path) -> Path:
    """Copy the packaged model to a persistent destination and return its path."""

    destination = Path(destination).expanduser()
    if destination.is_dir() or destination.suffix == "":
        destination = destination / DEFAULT_MOUSE_MODEL
    destination.parent.mkdir(parents=True, exist_ok=True)
    with packaged_mouse_model_file() as model_path:
        shutil.copy2(model_path, destination)
    return destination


def verify_packaged_mouse_model() -> bool:
    """Return True when the packaged model SHA256 matches the manifest."""

    manifest = load_model_manifest()
    expected = manifest["models"][0]["sha256"]
    digest = hashlib.sha256()
    with packaged_mouse_model_file() as model_path:
        with model_path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest() == expected
