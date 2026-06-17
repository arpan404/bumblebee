#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from bumblebee.rl import CleaningConfig, build_demonstration_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean raw Mouse Tracker JSON files into an imitation dataset."
    )
    parser.add_argument(
        "--source",
        default="~/Documents/Mouse Tracker",
        help="Directory containing mouse_positions_*.json files.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/mouse_demonstrations.npz",
        help="Output .npz dataset path. Kept out of git by default.",
    )
    parser.add_argument("--num-points", type=int, default=64)
    parser.add_argument("--max-traces", type=int, default=None)
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=0.75,
        help="Pause gap used to split one raw stream into separate trajectories. Use 0.5-1.0 for the current tracker data.",
    )
    parser.add_argument("--min-points", type=int, default=8)
    parser.add_argument("--min-distance-px", type=float, default=20.0)
    parser.add_argument("--min-duration-seconds", type=float, default=0.05)
    parser.add_argument("--max-duration-seconds", type=float, default=8.0)
    parser.add_argument("--min-avg-speed-px-s", type=float, default=50.0)
    parser.add_argument("--max-avg-speed-px-s", type=float, default=8000.0)
    parser.add_argument("--max-instant-speed-px-s", type=float, default=20_000.0)
    parser.add_argument("--max-jump-px", type=float, default=500.0)
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of CPU workers. Defaults to all available cores.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = CleaningConfig(
        min_points=args.min_points,
        min_distance_px=args.min_distance_px,
        min_duration_seconds=args.min_duration_seconds,
        pause_seconds=args.pause_seconds,
        max_jump_px=args.max_jump_px,
        max_duration_seconds=args.max_duration_seconds,
        min_avg_speed_px_s=args.min_avg_speed_px_s,
        max_avg_speed_px_s=args.max_avg_speed_px_s,
        max_instant_speed_px_s=args.max_instant_speed_px_s,
    )
    dataset = build_demonstration_dataset(
        Path(args.source).expanduser(),
        num_points=args.num_points,
        config=config,
        max_traces=args.max_traces,
        workers=args.workers,
    )
    dataset.save(args.output)
    print(
        f"saved {len(dataset.signatures)} cleaned trajectories to {args.output} "
        f"with {args.num_points} points each"
    )


if __name__ == "__main__":
    main()
