#!/usr/bin/env bash
# End-to-end parallel pipeline: zarr -> LeRobot -> VAE latents -> validate.
#
#   * Step 1 is parallelized with --workers N processes (CPU-bound video encoding)
#   * Step 2 is parallelized with N GPUs (one process per GPU)
#
# Usage:
#   SOURCE_ROOT=... OUTPUT_ROOT=... bash script/run_prepare_our_dex_full_dataset_parallel.sh
#
# Configurable via env vars (defaults shown):
#   SOURCE_ROOT   - zarr root
#   OUTPUT_ROOT   - output LeRobot dataset root
#   MODEL_ROOT    - lingbot-va base model
#   TASK_TEXT     - natural language task text
#   SPLITS        - comma-separated zarr splits ("train" or "train,eval")
#   FPS=30  TARGET_FPS=15  HEIGHT=256  WIDTH=256  ACTION_TYPE=relative
#   WORKERS=32          - step 1 CPU worker count
#   NUM_GPUS            - step 2 GPU count (default: auto-detect)
#   GPU_IDS             - explicit GPU id list, e.g. "0 2 5" (overrides NUM_GPUS)
#   RUN_CONVERT=1
#   CONVERT_OVERWRITE=1
#   OVERWRITE_LATENTS=1
#   PRESERVE_LATENTS=1  - keep latents/ when overwriting output during step 1
#   RUN_LOADER_SMOKE=0  - run dataset-loader smoke test (requires datasets/pyarrow in lingbot-va)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SOURCE_ROOT="${SOURCE_ROOT:-/share-2/code/fanqilin/peiqi/download/our_data_samples/pour_egg_0409_first1_episode_zarr}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/share-2/code/fanqilin/peiqi/download/our_dex_lerobot}"
MODEL_ROOT="${MODEL_ROOT:-/share-2/code/fanqilin/peiqi/download/lingbot-va-base}"

TASK_TEXT="${TASK_TEXT:-pour egg}"
SPLITS="${SPLITS:-train}"
FPS="${FPS:-30}"
TARGET_FPS="${TARGET_FPS:-15}"
HEIGHT="${HEIGHT:-256}"
WIDTH="${WIDTH:-256}"
ACTION_TYPE="${ACTION_TYPE:-relative}"
SOURCE_BGR="${SOURCE_BGR:-0}"

WORKERS="${WORKERS:-32}"
RUN_CONVERT="${RUN_CONVERT:-1}"
CONVERT_OVERWRITE="${CONVERT_OVERWRITE:-1}"
PRESERVE_LATENTS="${PRESERVE_LATENTS:-1}"
OVERWRITE_LATENTS="${OVERWRITE_LATENTS:-1}"
RUN_LOADER_SMOKE="${RUN_LOADER_SMOKE:-0}"
LATENT_DTYPE="${LATENT_DTYPE:-auto}"

cd "${REPO_ROOT}"

if [[ "${RUN_CONVERT}" == "1" ]]; then
  echo "[1/4] Convert zarr -> LeRobot (CPU parallel, WORKERS=${WORKERS})"
  OVERWRITE="${CONVERT_OVERWRITE}" \
  PRESERVE_LATENTS="${PRESERVE_LATENTS}" \
  WORKERS="${WORKERS}" \
  SOURCE_ROOT="${SOURCE_ROOT}" \
  OUTPUT_ROOT="${OUTPUT_ROOT}" \
  TASK_TEXT="${TASK_TEXT}" \
  SPLITS="${SPLITS}" \
  FPS="${FPS}" \
  HEIGHT="${HEIGHT}" \
  WIDTH="${WIDTH}" \
  ACTION_TYPE="${ACTION_TYPE}" \
  SOURCE_BGR="${SOURCE_BGR}" \
    bash script/run_convert_our_dex_lerobot_parallel.sh
else
  echo "[1/4] Skip conversion because RUN_CONVERT=${RUN_CONVERT}"
fi

echo ""
echo "[2/4] Extract Wan2.2 VAE latents (multi-GPU)"
DATASET="${OUTPUT_ROOT}" \
MODEL_ROOT="${MODEL_ROOT}" \
TARGET_FPS="${TARGET_FPS}" \
ORI_FPS="${FPS}" \
HEIGHT="${HEIGHT}" \
WIDTH="${WIDTH}" \
LATENT_DTYPE="${LATENT_DTYPE}" \
OVERWRITE="${OVERWRITE_LATENTS}" \
  bash script/run_extract_latents_multi_gpu.sh

echo ""
echo "[3/4] Check converted dataset including latents"
conda run -n lingbot-va python tools/check_dex_lerobot_dataset.py \
  --dataset "${OUTPUT_ROOT}" \
  --check-latents

if [[ "${RUN_LOADER_SMOKE}" == "1" ]]; then
  echo ""
  echo "[4/4] Run LingBot-VA dataset loader smoke test"
  LINGBOT_VA_DEX_DATASET_PATH="${OUTPUT_ROOT}" \
  LINGBOT_VA_MODEL_PATH="${MODEL_ROOT}" \
    conda run -n lingbot-va python - <<'PY'
import importlib.util

missing = [
    name for name in ("datasets", "pyarrow")
    if importlib.util.find_spec(name) is None
]
if missing:
    raise SystemExit(
        "Missing packages in lingbot-va env for loader smoke test: "
        + ", ".join(missing)
    )

from wan_va.configs import VA_CONFIGS
from wan_va.dataset import MultiLatentLeRobotDataset

dataset = MultiLatentLeRobotDataset(config=VA_CONFIGS["dex_train"])
if len(dataset) == 0:
    raise SystemExit("Dataset loader found zero trainable latent segments.")

item = dataset[0]
print("dataset_len", len(dataset))
for key in ("latents", "actions", "actions_mask", "text_emb"):
    value = item[key]
    print(key, tuple(value.shape), value.dtype)
if item["actions"].shape[0] != 58:
    raise SystemExit(f"actions first dim must be 58, got {item['actions'].shape}")
PY
else
  echo ""
  echo "[4/4] Skip loader smoke test (RUN_LOADER_SMOKE=${RUN_LOADER_SMOKE})"
fi

echo ""
echo "Done: ${OUTPUT_ROOT}"
