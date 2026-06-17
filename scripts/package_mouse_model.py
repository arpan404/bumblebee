#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

MODEL_DIR = Path("src/bumblebee/models")
DEFAULT_MODEL_NAME = "sac_mouse_v2.zip"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy a trained mouse policy into the package model directory."
    )
    parser.add_argument(
        "--model",
        default="artifacts/rl/sac_mouse_512/best_model.zip",
        help="Source SB3 .zip model to package.",
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_MODEL_NAME,
        help="Packaged model filename under src/bumblebee/models.",
    )
    parser.add_argument(
        "--dataset",
        default="artifacts/mouse_demonstrations.npz",
        help="Optional dataset used for training. Its SHA256 is recorded when present.",
    )
    parser.add_argument("--source-run", default=None)
    parser.add_argument("--screen-width", type=int, default=4096)
    parser.add_argument("--screen-height", type=int, default=2304)
    parser.add_argument("--max-steps", type=int, default=96)
    parser.add_argument("--dt", type=float, default=1 / 120)
    parser.add_argument("--max-velocity-px-s", type=float, default=6500.0)
    parser.add_argument("--target-radius-px", type=float, default=8.0)
    parser.add_argument("--net-arch", type=int, nargs="*", default=[512, 512])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.model).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"model not found: {source}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    destination = MODEL_DIR / args.output_name
    shutil.copy2(source, destination)

    dataset_path = Path(args.dataset).expanduser()
    dataset_sha = sha256(dataset_path) if dataset_path.exists() else None
    model_sha = sha256(destination)

    manifest = {
        "default_mouse_model": args.output_name,
        "models": [
            {
                "name": Path(args.output_name).stem,
                "file": args.output_name,
                "algorithm": "SAC",
                "policy": "MlpPolicy",
                "source_run": args.source_run or str(source),
                "sha256": model_sha,
                "size_bytes": destination.stat().st_size,
                "screen": {
                    "width": args.screen_width,
                    "height": args.screen_height,
                },
                "env": {
                    "max_steps": args.max_steps,
                    "dt": args.dt,
                    "max_velocity_px_s": args.max_velocity_px_s,
                    "target_radius_px": args.target_radius_px,
                    "observation_size": 10,
                    "action_size": 2,
                },
                "training": {
                    "preset": "m4-max",
                    "net_arch": args.net_arch,
                    "dataset_sha256": dataset_sha,
                },
            }
        ],
    }
    (MODEL_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"packaged model: {destination}")
    print(f"sha256: {model_sha}")
    if dataset_sha:
        print(f"dataset sha256: {dataset_sha}")


if __name__ == "__main__":
    main()
