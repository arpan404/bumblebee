# Bumblebee Architecture

This document describes the current Bumblebee runtime architecture and, in detail, the RL mouse-imitation subsystem: package layout, data flow, data types, environment contract, reward system, training workflow, and runtime integration points.

---

## 1. High-level overview

Bumblebee has two intentionally separate layers:

1. **Runtime control layer**
   - `src/bumblebee/mouse.py`
   - `src/bumblebee/keyboard.py`
   - Lightweight runtime dependencies only.
   - Provides human-like mouse/keyboard automation APIs.
   - Can execute custom/RL-generated mouse paths, but does not require RL dependencies.

2. **RL training/tooling layer**
   - `src/bumblebee/rl/**`
   - `scripts/package_mouse_model.py`
   - `scripts/prepare_mouse_data.py`
   - `scripts/train_mouse_sac.py`
   - `scripts/visualize_mouse_policy.py`
   - Used to clean mouse-tracker data, train SAC policies, score trajectories, and visualize policies.
   - Requires the `train` dependency group.

The Python wheel bundles the default v2 SAC mouse model at `bumblebee.models/sac_mouse_v2.zip`. Runtime mouse/keyboard imports remain lightweight; loading the packaged model requires the optional `rl` extra or the local `train` dependency group. External models can still be connected through `Mouse(path_provider=...)`, where the provider returns a complete path.

---

## 2. Package layout

```text
src/bumblebee/
├── __init__.py
├── mouse.py
├── keyboard.py
├── models/
│   ├── __init__.py
│   ├── manifest.json
│   ├── MODEL_CARD.md
│   └── sac_mouse_v2.zip
└── rl/
    ├── __init__.py
    ├── core/
    │   ├── __init__.py
    │   ├── geometry.py
    │   ├── reward.py
    │   └── types.py
    ├── data/
    │   ├── __init__.py
    │   └── dataset.py
    ├── envs/
    │   ├── __init__.py
    │   ├── mouse.py
    │   └── gymnasium.py
    ├── policy.py
    └── training/
        ├── __init__.py
        └── callbacks.py
```

### Runtime modules

| Module | Responsibility |
| --- | --- |
| `bumblebee.mouse` | Runtime mouse API, movement profiles, path execution, custom/RL path provider hook. |
| `bumblebee.keyboard` | Runtime keyboard API, typing profiles, hotkeys, editing helpers, clipboard helpers. |
| `bumblebee.models` | Packaged default mouse model, model manifest, and helper functions for exporting/verifying the model. |

### RL modules

| Module | Responsibility |
| --- | --- |
| `bumblebee.rl.core.geometry` | Arc-length resampling, local-frame conversion, curvature. |
| `bumblebee.rl.core.types` | `MouseTrace` and normalized trajectory signatures. |
| `bumblebee.rl.core.reward` | Terminal imitation scoring used after successful target reaches. |
| `bumblebee.rl.data.dataset` | Tracker file loading, cleaning/segmentation, `.npz` dataset creation/loading. |
| `bumblebee.rl.envs.mouse` | Lightweight non-Gym mouse imitation environment and shaped reward. |
| `bumblebee.rl.envs.gymnasium` | Gymnasium wrapper for Stable-Baselines3. |
| `bumblebee.rl.training.callbacks` | Rich terminal training progress callback. |
| `bumblebee.rl.policy` | Optional Stable-Baselines3 path provider for loading the packaged/default SAC model. |

---

## 3. Dependencies

### Runtime dependencies

Declared in `[project].dependencies`:

- `numpy`
- `pyautogui`
- `pynput`
- `pyperclip`

### Optional RL/training dependencies

Published package users install these with:

```bash
uv add "the-bumblebee[rl]"
```

Contributors working from a checkout install the equivalent local group with:

```bash
uv sync --group train
```

Declared in `[project.optional-dependencies].rl` and `[dependency-groups].train`:

- `gymnasium`
- `stable-baselines3`
- `torch`
- `rich`
- `tensorboard`
- `tqdm`

Install training tools with:

```bash
uv sync --group train
```

---

## 4. RL data pipeline

### 4.1 Raw input format

The data preparation script expects files named:

```text
mouse_positions_*.json
```

Each file should contain entries with:

```json
{
  "x": 123.0,
  "y": 456.0,
  "timestamp": 1710000000.123
}
```

`load_tracker_file()` loads a file into an `N x 3` NumPy array:

```text
[x, y, timestamp]
```

The array is sorted by timestamp.

### 4.2 Cleaning and segmentation

`clean_and_segment()` converts a raw cursor stream into a list of `MouseTrace` objects.

Cuts happen on:

- non-increasing timestamps
- long gaps / pauses
- impossible jumps
- excessive instantaneous speed
- sustained stationary samples

Filtering happens through `CleaningConfig`:

| Field | Default | Meaning |
| --- | ---: | --- |
| `min_points` | `8` | Minimum samples in a valid trace. |
| `min_distance_px` | `20.0` | Minimum start-to-end displacement. |
| `min_duration_seconds` | `0.05` | Minimum trace duration. |
| `pause_seconds` | `0.75` | Gap/stationary duration used to split traces. |
| `max_jump_px` | `500.0` | Maximum allowed single-sample jump. |
| `stationary_epsilon_px` | `0.5` | Movement below this is considered stationary. |
| `max_duration_seconds` | `8.0` | Maximum trace duration. |
| `min_avg_speed_px_s` | `50.0` | Minimum average movement speed. |
| `max_avg_speed_px_s` | `8000.0` | Maximum average movement speed. |
| `max_instant_speed_px_s` | `20000.0` | Maximum single-step speed. |

### 4.3 `MouseTrace`

Defined in `bumblebee.rl.core.types`.

```python
@dataclass(frozen=True)
class MouseTrace:
    points: np.ndarray  # shape: (N, 3), columns: x, y, timestamp
```

Important properties/methods:

| API | Meaning |
| --- | --- |
| `start` | First `(x, y)` point. |
| `destination` | Last `(x, y)` point. |
| `duration` | Last timestamp minus first timestamp. |
| `displacement` | `destination - start`. |
| `distance` | Euclidean displacement length. |
| `velocities()` | Per-sample velocity vectors in px/s. |
| `normalized_signature(num_points)` | Returns normalized path shape and normalized speed profile. |

### 4.4 Normalized signatures

A cleaned trace is converted into:

1. **Path signature** with shape `(num_points, 2)`
2. **Speed profile** with shape `(num_points,)`

The path is converted to a local coordinate frame:

- local `x`: progress along the start→destination vector, normalized by distance
- local `y`: lateral deviation from the direct line, normalized by distance

This allows a real human movement style to be reused for any sampled virtual start/destination pair.

### 4.5 Dataset schema

`MouseDemonstrationDataset.save()` writes a compressed `.npz` with:

| Key | Shape | Meaning |
| --- | --- | --- |
| `signatures` | `(N, num_points, 2)` | Normalized local-frame path signatures. |
| `speed_profiles` | `(N, num_points)` | Normalized speed profiles. |
| `durations` | `(N,)` | Original movement durations in seconds. |

`MouseDemonstrationDataset.get(index, start, destination)` transforms a stored local signature back into screen coordinates with `from_local_frame()`.

---

## 5. Geometry primitives

Defined in `bumblebee.rl.core.geometry`.

| Function | Purpose |
| --- | --- |
| `resample_polyline(points, num_points)` | Resample a 2D polyline by arc length. |
| `to_local_frame(points, start, destination)` | Convert screen coordinates into normalized local coordinates. |
| `from_local_frame(local, start, destination)` | Convert normalized local coordinates back to screen coordinates. |
| `curvature(points)` | Approximate unsigned turn angle at each point. |

The local-frame representation is the bridge between raw human demonstrations and arbitrary virtual-screen training tasks.

---

## 6. RL environment

The core environment is `MouseImitationEnv` in `bumblebee.rl.envs.mouse`.

It is intentionally lightweight and NumPy-based. The Gymnasium adapter is a thin wrapper in `bumblebee.rl.envs.gymnasium`.

### 6.1 `VirtualScreen`

```python
@dataclass(frozen=True)
class VirtualScreen:
    width: int = 4096
    height: int = 2304
```

The virtual screen defines coordinate bounds and diagonal normalization.

### 6.2 `MouseEnvConfig`

Important fields:

| Field | Default | Meaning |
| --- | ---: | --- |
| `max_steps` | `96` | Episode step cap. |
| `dt` | `1 / 120` | Integration timestep in seconds. |
| `max_velocity_px_s` | `6500.0` | Maximum cursor speed. |
| `min_start_dest_distance_px` | `20.0` | Minimum task distance. |
| `max_start_dest_distance_px` | `None` | Optional max task distance. |
| `target_radius_px` | `8.0` | Target reach/crossing radius. |
| `task_reachability_margin` | `0.85` | Keeps sampled tasks physically reachable. |
| `record_reward_terms` | `True` | Include reward breakdown in `info`. |

`max_reachable_distance_px` is computed as:

```text
max_velocity_px_s * dt * max_steps * task_reachability_margin
```

and then clamped by the screen diagonal and optional configured maximum.

### 6.3 Reset behavior

`reset()`:

1. samples a reachable start/destination pair on the virtual screen
2. samples a demonstration index from `MouseDemonstrationDataset`
3. clears velocity, rollout, travel, and reward state
4. returns the initial observation

### 6.4 Observation contract

Observation shape is `10`:

| Index | Value | Range-ish |
| ---: | --- | --- |
| `0` | `position_x / screen.width` | `[0, 1]` |
| `1` | `position_y / screen.height` | `[0, 1]` |
| `2` | `destination_x / screen.width` | `[0, 1]` |
| `3` | `destination_y / screen.height` | `[0, 1]` |
| `4` | `(destination_x - position_x) / screen.width` | `[-1, 1]` |
| `5` | `(destination_y - position_y) / screen.height` | `[-1, 1]` |
| `6` | `distance_to_target / screen.diagonal` | `[0, 1]` |
| `7` | `previous_velocity_x / max_velocity_px_s` | `[-1, 1]` |
| `8` | `previous_velocity_y / max_velocity_px_s` | `[-1, 1]` |
| `9` | `step_count / max_steps` | `[0, 1]` |

The Gymnasium wrapper exposes matching Box bounds.

### 6.5 Action contract

Action shape is `2`:

```text
[vx_action, vy_action]
```

Rules:

1. Values are clipped to `[-1, 1]`.
2. The vector is clipped to the unit circle.
3. The vector is scaled by `max_velocity_px_s`.
4. Position is integrated with `dt` and clipped to the virtual screen.

This keeps `max_velocity_px_s` as a real physical speed limit.

### 6.6 Termination

An episode ends when:

- the movement segment reaches/crosses the target radius, or
- `max_steps` is reached.

`info` contains:

```python
{
    "success": bool,
    "reached": bool,
    "truncated": bool,
    "distance_to_target": float,
    "reward_terms": dict[str, float]  # when enabled
}
```

The Gymnasium wrapper maps `reached` to `terminated` and `truncated` to `truncated`.

---

## 7. Reward system

There are two reward layers:

1. **Dense environment reward** in `MouseImitationEnv._step_reward()`
2. **Terminal imitation bonus** in `ImitationReward`

The design goal is: **reach the target first, then reward human-like path quality**.

### 7.1 Dense reward terms

At each step:

```text
progress = (previous_distance - current_distance) / start_distance
```

The total reward is the sum of:

| Term | Formula / behavior | Purpose |
| --- | --- | --- |
| `step` | `step_penalty` | Small cost for taking steps. |
| `progress` | `progress_reward_scale * progress` | Rewards moving closer to the target. |
| `backward` | `-backward_penalty_scale * abs(progress)` when progress is negative | Penalizes moving away. |
| `small_move` | `small_move_penalty` when far from target and speed is too low | Discourages stalling. |
| `acceleration` | `-acceleration_penalty_scale * ||v_t - v_{t-1}||` | Encourages smooth acceleration. |
| `jitter` | `-jitter_penalty_scale * direction_change_penalty()` | Penalizes sharp direction reversals. |
| `local_loop` | penalty when local travel is inefficient | Discourages loops/circles. |
| `success` | `success_reward` on target reach | Main terminal success reward. |
| `imitation` | `imitation_bonus * imitation_score` on success | Human-like terminal bonus. |
| `early_finish` | `max(0, 1 - step_count / max_steps)` on success | Rewards finishing early. |
| `terminal` | `failure_penalty` on truncation | Penalizes timeout. |

Default key scales from `MouseEnvConfig`:

| Field | Default |
| --- | ---: |
| `success_reward` | `12.0` |
| `imitation_bonus` | `2.0` |
| `failure_penalty` | `-3.0` |
| `step_penalty` | `-0.01` |
| `progress_reward_scale` | `0.35` |
| `backward_penalty_scale` | `3.0` |
| `acceleration_penalty_scale` | `0.00002` |
| `jitter_penalty_scale` | `0.10` |
| `small_move_penalty` | `-0.04` |
| `local_loop_penalty` | `-0.12` |

### 7.2 Direction-change penalty

The environment compares the previous velocity direction and current velocity direction.

- If either velocity is near zero, penalty is `0`.
- Otherwise, it computes cosine similarity.
- Direction changes with cosine below `0.25` are scaled into `[0, 1]`.

This catches sudden reversals and jitter without punishing gentle turns.

### 7.3 Local-loop penalty

The environment looks at a trailing window of points:

```text
efficiency = net_displacement / traveled_distance
```

If:

- traveled distance exceeds `local_loop_min_travel_px`, and
- efficiency is below `local_loop_efficiency_threshold`

then a scaled loop penalty is applied.

This discourages policies from circling or oscillating while collecting dense reward.

### 7.4 Terminal imitation reward

Defined in `bumblebee.rl.core.reward.ImitationReward`.

It is only evaluated after the target is reached.

Default weights:

| Component | Weight |
| --- | ---: |
| Path shape | `0.40` |
| Speed profile | `0.25` |
| Turn/curvature | `0.20` |
| Efficiency | `0.15` |

#### Path score

1. Resample rollout to demonstration length.
2. Convert rollout and demo to local frames.
3. Compute mean pointwise L2 error.
4. Convert to score:

```text
path_score = exp(-4.0 * path_error)
```

#### Speed score

1. Interpolate rollout speeds to demonstration length.
2. Normalize rollout speed profile by max speed.
3. Compare to demo speed profile by mean absolute error.
4. Convert to score:

```text
speed_score = exp(-3.0 * speed_error)
```

#### Turn score

1. Compute curvature for rollout and demo in local frame.
2. Compare by mean absolute error.
3. Convert to score:

```text
turn_score = exp(-2.0 * turn_error)
```

#### Efficiency score

```text
efficiency_score = clip(direct_distance / traveled_distance, 0, 1)
```

#### Final imitation score

```text
imitation_score =
    0.40 * path_score +
    0.25 * speed_score +
    0.20 * turn_score +
    0.15 * efficiency_score
```

Then environment reward adds:

```text
imitation_bonus * imitation_score
```

---

## 8. Training workflow

### 8.1 Prepare data

```bash
uv run --group train python scripts/prepare_mouse_data.py \
  --source "~/Documents/Mouse Tracker" \
  --output artifacts/mouse_demonstrations.npz \
  --num-points 64
```

Output:

```text
artifacts/mouse_demonstrations.npz
```

### 8.2 Train policy

```bash
uv run --group train python scripts/train_mouse_sac.py \
  --dataset artifacts/mouse_demonstrations.npz \
  --output-dir artifacts/rl/sac_mouse \
  --preset m4-max \
  --timesteps 250000
```

The script uses:

- Stable-Baselines3 `SAC`
- `MlpPolicy`
- `Monitor`
- `DummyVecEnv` or `SubprocVecEnv`
- TensorBoard logs
- checkpoint saving
- optional replay-buffer saving/resuming
- `RichTrainingCallback` for live terminal metrics

Runtime presets:

| Preset | Intent |
| --- | --- |
| `manual` | Conservative defaults, mostly explicit CLI control. |
| `cpu-fast` | CPU-optimized local run with dummy vector env. |
| `cpu-parallel` | CPU run with subprocess vector env. |
| `m4-max` | Local Apple Silicon CPU-oriented preset. |
| `mps-heavy` | MPS/GPU utilization experiment. |

### 8.3 Training artifacts

Typical output directory:

```text
artifacts/rl/sac_mouse/
├── best_model.zip
├── final_model.zip
├── latest_model.zip
├── latest_replay_buffer.pkl
├── checkpoints/
└── logs/
```

Artifacts are intentionally not committed.

### 8.4 Visualize policy

The visualizer defaults to the packaged model. You can also pass any SB3 `.zip` model.

```bash
uv run --group train python scripts/visualize_mouse_policy.py

uv run --group train python scripts/visualize_mouse_policy.py \
  --model artifacts/rl/sac_mouse/best_model.zip
```

The visualizer lets you click start/destination points and overlay multiple stochastic rollouts.

---

## 9. Runtime integration with RL paths

`Mouse` can execute the packaged SAC policy directly:

```python
from bumblebee import Mouse
from bumblebee.rl.policy import SB3MousePolicyPathProvider

provider = SB3MousePolicyPathProvider.from_packaged(deterministic=False)
mouse = Mouse(path_provider=provider)
mouse.move(700, 450)
```

This requires `the-bumblebee[rl]` or `uv sync --group train`.

`Mouse` can also execute any complete path with:

```python
mouse.move_path(path)
```

or use a provider:

```python
from bumblebee import Mouse


def path_provider(start, destination):
    # Return N x 2 points or N x 3 points.
    # If N x 3, column 3 is a speed factor.
    return policy_rollout(start, destination)


mouse = Mouse(path_provider=path_provider)
mouse.move(700, 450)
```

Path rules:

- accepted shapes: `(N, 2)` or `(N, 3)`
- coordinates must be finite
- `N x 2` paths get a generated speed profile
- `N x 3` paths keep their speed factors, clipped to profile bounds
- if the path does not start at the current cursor position, Bumblebee prepends the current position
- if the path reaches/crosses the requested target radius early, Bumblebee trims the rest of the path and snaps the final point to the requested destination
- if the path ends within the target radius, Bumblebee snaps the final point exactly to the requested destination
- if the path misses the requested destination, Bumblebee appends a final correction segment
- provided paths are not decorated with extra jitter or curvature

This is the intended bridge between a trained RL policy and the production mouse API.

### 9.1 Random bounded clicks

Runtime mouse code also supports choosing a random point inside a rectangle before clicking. This is useful when an AI detector returns a bounding box for a button or target region and any point inside that region is acceptable.

```python
from bumblebee import Mouse, MouseBounds

mouse = Mouse()
mouse.click_in_bounds(MouseBounds(100, 200, 260, 240), padding=6)
mouse.click_in_rect(100, 200, 160, 40, padding=6)  # x, y, width, height
```

Bounds APIs:

- `MouseBounds(left, top, right, bottom)` stores absolute rectangle edges.
- `MouseBounds.from_xywh(x, y, width, height)` builds bounds from an `x/y/width/height` rectangle.
- `random_point_in_bounds(...)` returns the selected random point without clicking.
- `move_to_random_in_bounds(...)` moves to the selected point and returns it.
- `click_in_bounds(...)` and `click_in_rect(...)` move to and click a selected point.

Options:

- `padding` shrinks the clickable rectangle so clicks avoid edges.
- `clamp_to_screen=True` intersects the rectangle with the current screen.
- if the rectangle does not intersect the screen, a `ValueError` is raised.

---

## 10. Public exports

`bumblebee.rl` exports commonly used RL types:

- `CleaningConfig`
- `Demonstration`
- `MouseDemonstrationDataset`
- `MouseEnvConfig`
- `MouseImitationEnv`
- `MouseTrace`
- `VirtualScreen`
- `ImitationReward`
- `build_demonstration_dataset`
- `clean_and_segment`
- `iter_tracker_files`
- `load_tracker_file`

Runtime package exports come from `bumblebee.__init__`:

- `Mouse`
- `Keyboard`
- related profile/data classes exposed by their modules, including `MouseBounds`, `MouseProfile`, and `KeyboardProfile`

`bumblebee.models` exports packaged-model helpers:

- `packaged_mouse_model_file()`
- `packaged_mouse_model_path()`
- `export_packaged_mouse_model(destination)`
- `load_model_manifest()`
- `verify_packaged_mouse_model()`

`bumblebee.rl.policy` exports `SB3MousePolicyPathProvider` for loading the packaged model or another SB3 SAC model as a runtime mouse path provider.

---

## 11. Important design decisions

1. **Training dependencies are isolated**
   - Runtime automation does not require SB3, Gymnasium, Torch, or Rich.

2. **Reach target before imitation matters**
   - Imitation scoring is terminal-only to avoid policies earning high reward for pretty paths that never arrive.

3. **Local-frame demonstrations**
   - Human path styles are normalized and reused across arbitrary start/destination tasks.

4. **Packaged default model, external datasets**
   - The wheel includes the default v2 SAC mouse model so users can run the RL path provider without downloading a separate model asset.
   - Datasets, checkpoints, replay buffers, and logs stay under `artifacts/` locally and are distributed separately as release assets when needed.

5. **Runtime accepts complete paths**
   - RL-generated trajectories should be used directly through `Mouse(path_provider=...)` or `move_path()`.

---

## 12. Common validation checks

Recommended checks before opening a PR:

```bash
uv run black src scripts
uv run isort src scripts
uv run python -m compileall scripts src
uv run --group train python -c "from bumblebee.rl.envs.gymnasium import GymMouseImitationEnv; print(GymMouseImitationEnv.__name__)"
```

Optional smoke test with an existing dataset:

```bash
uv run --group train python - <<'PY'
from pathlib import Path
import numpy as np
from bumblebee.rl.envs.gymnasium import GymMouseImitationEnv

path = Path("artifacts/mouse_demonstrations.npz")
if path.exists():
    env = GymMouseImitationEnv(path, seed=123)
    obs, _ = env.reset(seed=123)
    obs, reward, terminated, truncated, info = env.step(
        np.array([0.1, 0.2], dtype=np.float32)
    )
    print(obs.shape, reward, terminated, truncated)
else:
    print("dataset missing; skipped")
PY
```
