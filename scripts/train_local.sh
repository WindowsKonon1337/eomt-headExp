#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${CONFIG_PATH:-configs/dinov2/coco/panoptic/eomt_base_640.yaml}"
DATA_PATH="${DATA_PATH:-./datasets}"
TRAINER_DEVICES="${TRAINER_DEVICES:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
WANDB_MODE="${WANDB_MODE:-offline}"

WANDB_MODE="${WANDB_MODE}" uv run python main.py fit \
  -c "${CONFIG_PATH}" \
  --data.path "${DATA_PATH}" \
  --trainer.devices "${TRAINER_DEVICES}" \
  --data.batch_size "${BATCH_SIZE}" \
  "$@"
