# RL Workflow

The RL workflow trains mouse trajectory policies from cleaned mouse movement demonstrations.

For deeper implementation details, see [`architecture.md`](architecture.md).

## Install training dependencies

```bash
uv sync --group train
```

## 1. Prepare mouse data

Raw files should be named:

```text
mouse_positions_*.json
```

Each item should contain:

```json
{
  "x": 100.0,
  "y": 200.0,
  "timestamp": 1710000000.123
}
```

Run:

```bash
uv run --group train python scripts/prepare_mouse_data.py \
  --source "~/Documents/Mouse Tracker" \
  --output artifacts/mouse_demonstrations.npz \
  --num-points 64
```

Useful options:

```bash
--workers 8
--max-traces 100000
--pause-seconds 0.75
--min-distance-px 20
--max-instant-speed-px-s 20000
```

Output schema:

| Key | Shape | Meaning |
| --- | --- | --- |
| `signatures` | `(N, num_points, 2)` | Normalized local-frame paths. |
| `speed_profiles` | `(N, num_points)` | Normalized speed profiles. |
| `durations` | `(N,)` | Original movement durations. |

## 2. Train SAC policy

```bash
uv run --group train python scripts/train_mouse_sac.py \
  --dataset artifacts/mouse_demonstrations.npz \
  --output-dir artifacts/rl/sac_mouse \
  --preset m4-max \
  --timesteps 250000
```

Common options:

```bash
--auto-resume
--num-envs 16
--vec-env subproc
--net-arch 512 512
--checkpoint-freq 100000
--reward-terms
```

Long run helper:

```bash
./scripts/run_mouse_sac_long.sh
```

## 3. Visualize policy

Visualize the packaged model:

```bash
uv run --group train python scripts/visualize_mouse_policy.py
```

Visualize an explicit model:

```bash
uv run --group train python scripts/visualize_mouse_policy.py \
  --model artifacts/rl/sac_mouse/best_model.zip
```

The visualizer lets you:

- click a start point
- click a destination
- overlay multiple stochastic rollouts
- switch deterministic/non-deterministic rollout mode
- load another model file

## 4. Package a model

After validating a training run, package the chosen model:

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

Then update `MODEL_CARD.md` and release notes if needed.

## 5. Runtime use

```python
from bumblebee import Mouse
from bumblebee.rl.policy import SB3MousePolicyPathProvider

provider = SB3MousePolicyPathProvider.from_packaged(deterministic=False)
mouse = Mouse(path_provider=provider)
mouse.move(700, 450)
```

## Environment summary

The training environment is `MouseImitationEnv`.

Observation shape: `10`

```text
position xy
 destination xy
 delta xy
 distance
 previous velocity xy
 progress
```

Action shape: `2`

```text
vx_action, vy_action
```

Actions are clipped to `[-1, 1]`, clipped to the unit circle, scaled by max velocity, and integrated with `dt`.

## Reward summary

Reward combines:

- step penalty
- progress reward
- backward movement penalty
- small-move/stall penalty
- acceleration penalty
- jitter/direction-change penalty
- local-loop penalty
- success reward
- terminal imitation bonus
- early finish bonus
- timeout penalty

The imitation bonus only applies after reaching the target. It compares:

- path shape
- speed profile
- turn/curvature profile
- travel efficiency

See [`architecture.md`](architecture.md) for formulas.

## Artifacts

Training outputs usually live under:

```text
artifacts/rl/sac_mouse/
├── best_model.zip
├── final_model.zip
├── latest_model.zip
├── latest_replay_buffer.pkl
├── checkpoints/
└── logs/
```

Do not commit these artifacts. Release selected models/datasets through GitHub Releases.
