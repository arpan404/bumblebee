#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATASET="${DATASET:-artifacts/mouse_demonstrations.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/rl/sac_mouse_512}"
TIMESTEPS="${TIMESTEPS:-100000000}"
NUM_ENVS="${NUM_ENVS:-16}"
CHECKPOINT_FREQ="${CHECKPOINT_FREQ:-100000}"
LOG_INTERVAL_SECONDS="${LOG_INTERVAL_SECONDS:-60}"
NET_ARCH="${NET_ARCH:-512 512}"
read -r -a NET_ARCH_ARGS <<< "$NET_ARCH"

LOG_DIR="$OUTPUT_DIR/run_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"

echo "Logging to $LOG_FILE"
echo "Target timesteps: $TIMESTEPS"
echo "Output dir: $OUTPUT_DIR"
echo "Env workers: $NUM_ENVS"
echo "Network architecture: $NET_ARCH"
echo "Checkpoint frequency: $CHECKPOINT_FREQ"

uv run --group train python scripts/train_mouse_sac.py \
  --dataset "$DATASET" \
  --output-dir "$OUTPUT_DIR" \
  --timesteps "$TIMESTEPS" \
  --timesteps-mode target \
  --preset m4-max \
  --num-envs "$NUM_ENVS" \
  --net-arch "${NET_ARCH_ARGS[@]}" \
  --auto-resume \
  --checkpoint-freq "$CHECKPOINT_FREQ" \
  --log-interval-seconds "$LOG_INTERVAL_SECONDS" \
  "$@" 2>&1 | tee "$LOG_FILE"
