#!/usr/bin/env bash
# Runs convert_dex_zarr_to_lerobot.py with multi-process workers to speed up
# per-episode parquet + MP4 writing (bottleneck is cv2 video encoding).
#
# Usage:
#   bash script/run_convert_our_dex_lerobot_parallel.sh
#
# Configurable via env vars (defaults below):
#   SOURCE_ROOT   - zarr root containing train/ (and optionally eval/)
#   OUTPUT_ROOT   - output LeRobot dataset directory
#   TELEOP_ROOT   - Teleop-franka-test repo (used only by validate step)
#   TASK_TEXT     - natural language task description
#   SPLITS        - comma-separated zarr splits (default: "train")
#   FPS, HEIGHT, WIDTH, ACTION_TYPE - see convert_dex_zarr_to_lerobot.py
#   WORKERS       - number of parallel worker processes (default: 32)
#   OVERWRITE=1   - overwrite existing OUTPUT_ROOT
#   PRESERVE_LATENTS=1 - keep latents/ when overwriting

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SOURCE_ROOT="${SOURCE_ROOT:-/share-2/code/fanqilin/peiqi/download/our_data_samples/pour_egg_0409_first1_episode_zarr}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/share-2/code/fanqilin/peiqi/download/our_dex_lerobot}"
TELEOP_ROOT="${TELEOP_ROOT:-/share-2/code/fanqilin/peiqi/Teleop-franka-test}"

TASK_TEXT="${TASK_TEXT:-pour egg}"
SPLITS="${SPLITS:-train}"
FPS="${FPS:-30}"
HEIGHT="${HEIGHT:-256}"
WIDTH="${WIDTH:-256}"
ACTION_TYPE="${ACTION_TYPE:-relative}"
WORKERS="${WORKERS:-32}"
OVERWRITE="${OVERWRITE:-1}"
PRESERVE_LATENTS="${PRESERVE_LATENTS:-1}"
SOURCE_BGR="${SOURCE_BGR:-0}"

convert_args=(
  --input "${SOURCE_ROOT}"
  --output "${OUTPUT_ROOT}"
  --splits "${SPLITS}"
  --task-text "${TASK_TEXT}"
  --fps "${FPS}"
  --resize "${HEIGHT}" "${WIDTH}"
  --action-type "${ACTION_TYPE}"
  --workers "${WORKERS}"
)

if [[ "${OVERWRITE}" == "1" ]]; then
  convert_args+=(--overwrite)
fi

if [[ "${PRESERVE_LATENTS}" == "1" ]]; then
  convert_args+=(--preserve-latents)
fi

if [[ "${SOURCE_BGR}" == "1" ]]; then
  convert_args+=(--source-bgr)
fi

cd "${REPO_ROOT}"

echo "[1/3] Convert zarr -> LeRobot (workers=${WORKERS})"
conda run -n dp python tools/convert_dex_zarr_to_lerobot.py "${convert_args[@]}"

echo "[2/3] Check converted LeRobot structure"
conda run -n dp python tools/check_dex_lerobot_dataset.py --dataset "${OUTPUT_ROOT}"

if [[ "${SPLITS}" == "train" ]]; then
  echo "[3/3] Validate action/state parity against Teleop code"
  conda run -n dp python tools/validate_dex_conversion_against_teleop.py \
    --source-zarr-split "${SOURCE_ROOT}/train" \
    --converted-dataset "${OUTPUT_ROOT}" \
    --teleop-root "${TELEOP_ROOT}" \
    --action-type "${ACTION_TYPE}"
else
  echo "[3/3] Skipped Teleop parity check because SPLITS=${SPLITS}; run validate_dex_conversion_against_teleop.py per split if needed."
fi

echo "Done: ${OUTPUT_ROOT}"
