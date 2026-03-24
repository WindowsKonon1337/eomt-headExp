#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${CONFIG_PATH:-configs/dinov2/coco/panoptic/eomt_base_640.yaml}"
DATA_PATH="${DATA_PATH:-./datasets}"
TRAINER_DEVICES="${TRAINER_DEVICES:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
WANDB_MODE="${WANDB_MODE:-offline}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts}"
WANDB_DIR="${WANDB_DIR:-${OUTPUT_DIR}/wandb}"

WANDB_MODE="${WANDB_MODE}" OUTPUT_DIR="${OUTPUT_DIR}" WANDB_DIR="${WANDB_DIR}" uv run python main.py fit \
  -c "${CONFIG_PATH}" \
  --data.path "${DATA_PATH}" \
  --trainer.devices "${TRAINER_DEVICES}" \
  --data.batch_size "${BATCH_SIZE}" \
  "$@"
