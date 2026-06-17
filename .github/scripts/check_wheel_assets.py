from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

REQUIRED_WHEEL_FILES = {
    "bumblebee/models/sac_mouse_v2.zip",
    "bumblebee/models/manifest.json",
    "bumblebee/models/MODEL_CARD.md",
}


def main() -> None:
    wheels = sorted(Path("dist").glob("the_bumblebee-*.whl"))
    if not wheels:
        raise SystemExit("No the_bumblebee wheel found in dist/")

    wheel = wheels[-1]
    with ZipFile(wheel) as archive:
        names = set(archive.namelist())

    missing = sorted(REQUIRED_WHEEL_FILES - names)
    if missing:
        raise SystemExit(
            f"Wheel {wheel} is missing required packaged assets: {', '.join(missing)}"
        )

    print(f"Wheel asset check passed for {wheel}")


if __name__ == "__main__":
    main()
