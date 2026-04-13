#!/usr/bin/env bash
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

RUN_CONVERT="${RUN_CONVERT:-1}"
CONVERT_OVERWRITE="${CONVERT_OVERWRITE:-1}"
PRESERVE_LATENTS="${PRESERVE_LATENTS:-1}"
OVERWRITE_LATENTS="${OVERWRITE_LATENTS:-1}"
RUN_LOADER_SMOKE="${RUN_LOADER_SMOKE:-0}"
LATENT_DEVICE="${LATENT_DEVICE:-}"
LATENT_DTYPE="${LATENT_DTYPE:-auto}"

cd "${REPO_ROOT}"

if [[ "${RUN_CONVERT}" == "1" ]]; then
  echo "[1/4] Convert zarr -> LeRobot"
  OVERWRITE="${CONVERT_OVERWRITE}" \
  PRESERVE_LATENTS="${PRESERVE_LATENTS}" \
  SOURCE_ROOT="${SOURCE_ROOT}" \
  OUTPUT_ROOT="${OUTPUT_ROOT}" \
  TASK_TEXT="${TASK_TEXT}" \
  SPLITS="${SPLITS}" \
  FPS="${FPS}" \
  HEIGHT="${HEIGHT}" \
  WIDTH="${WIDTH}" \
  ACTION_TYPE="${ACTION_TYPE}" \
  SOURCE_BGR="${SOURCE_BGR}" \
    bash script/run_convert_our_dex_lerobot.sh
else
  echo "[1/4] Skip conversion because RUN_CONVERT=${RUN_CONVERT}"
fi

latent_args=(
  --dataset "${OUTPUT_ROOT}"
  --model-root "${MODEL_ROOT}"
  --target-fps "${TARGET_FPS}"
  --ori-fps "${FPS}"
  --height "${HEIGHT}"
  --width "${WIDTH}"
  --dtype "${LATENT_DTYPE}"
)

if [[ -n "${LATENT_DEVICE}" ]]; then
  latent_args+=(--device "${LATENT_DEVICE}")
fi

if [[ "${OVERWRITE_LATENTS}" == "1" ]]; then
  latent_args+=(--overwrite)
fi

echo "[2/4] Extract Wan2.2 VAE latents"
conda run -n lingbot-va python tools/extract_dex_video_latents.py "${latent_args[@]}"

echo "[3/4] Check converted dataset including latents"
conda run -n dp python tools/check_dex_lerobot_dataset.py \
  --dataset "${OUTPUT_ROOT}" \
  --check-latents

if [[ "${RUN_LOADER_SMOKE}" == "1" ]]; then
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
        + "\nInstall them first, for example: conda run -n lingbot-va pip install datasets pyarrow"
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
  echo "[4/4] Skip loader smoke test because RUN_LOADER_SMOKE=${RUN_LOADER_SMOKE}"
  echo "      Enable it after installing datasets/pyarrow in lingbot-va: RUN_LOADER_SMOKE=1"
fi

echo "Done: ${OUTPUT_ROOT}"
