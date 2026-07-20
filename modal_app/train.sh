#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

DATASET_REPO="${DATASET_REPO:-aether5896/arcaine_dataset}"
MODEL_REPO="${MODEL_REPO:-aether5896/arcaine-agentic-lora-v8}"
MAX_STEPS="${MAX_STEPS:-1000}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs}"
SLEEP_SECONDS="${SLEEP_SECONDS:-30}"
MAX_RESTARTS="${MAX_RESTARTS:-0}" # 0 = forever

mkdir -p "$LOG_DIR"

restart_count=0

while true; do
  run_id="$(date -u +%Y%m%dT%H%M%SZ)"
  log_file="$LOG_DIR/modal_train_${run_id}.log"

  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Starting Modal training run #$((restart_count + 1))"
  echo "Dataset repo: $DATASET_REPO"
  echo "Model repo:   $MODEL_REPO"
  echo "Log file:     $log_file"

  "$PYTHON_BIN" -m modal run "$PROJECT_ROOT/modal_app/train.py" \
    --dataset-repo "$DATASET_REPO" \
    --model-repo "$MODEL_REPO" \
    --max-steps "$MAX_STEPS" \
    2>&1 | tee "$log_file"

  exit_code=${PIPESTATUS[0]}
  restart_count=$((restart_count + 1))

  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Modal run exited with code $exit_code"

  if grep -q '"event": "train_end"' "$log_file" || grep -q "Done\." "$log_file"; then
    echo "Training appears complete. Not restarting."
    exit 0
  fi

  if [[ "$MAX_RESTARTS" != "0" && "$restart_count" -ge "$MAX_RESTARTS" ]]; then
    echo "Reached MAX_RESTARTS=$MAX_RESTARTS. Exiting."
    exit "$exit_code"
  fi

  echo "Training did not complete. Restarting in ${SLEEP_SECONDS}s..."
  sleep "$SLEEP_SECONDS"
done
