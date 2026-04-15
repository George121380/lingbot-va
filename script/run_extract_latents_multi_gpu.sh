#!/usr/bin/env bash
# Multi-GPU parallel latent extraction for a converted dex LeRobot dataset.
#
# The single-GPU extract_dex_video_latents.py iterates episodes sequentially.
# This wrapper launches one process per GPU, each handling a disjoint episode
# range (--episodes 0-46 on cuda:0, 47-93 on cuda:1, ...). No changes to the
# Python script are needed — --device and --episodes are existing CLI flags.
#
# Usage:
#   bash script/run_extract_latents_multi_gpu.sh
#
# Configurable via env vars:
#   DATASET          - converted LeRobot dataset root (must contain meta/episodes.jsonl)
#   MODEL_ROOT       - lingbot-va base model directory
#   NUM_GPUS         - number of GPUs to use (default: auto-detect via nvidia-smi)
#   GPU_IDS          - override list, space-separated, e.g. "0 2 5" (optional)
#   TARGET_FPS=15  ORI_FPS=30  HEIGHT=256  WIDTH=256
#   LATENT_DTYPE     - auto|bfloat16|float16|float32 (default: auto)
#   OVERWRITE=1      - pass --overwrite to each worker
#   LOG_DIR          - directory to write per-GPU logs (default: $DATASET/logs)
#   CONDA_ENV        - conda env with torch+diffusers (default: lingbot-va)
#
# Exit code is non-zero if any worker fails.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATASET="${DATASET:-/share-2/code/fanqilin/peiqi/dataset/converted_ours_dataset}"
MODEL_ROOT="${MODEL_ROOT:-/share-2/code/fanqilin/peiqi/download/lingbot-va-base}"
TARGET_FPS="${TARGET_FPS:-15}"
ORI_FPS="${ORI_FPS:-30}"
HEIGHT="${HEIGHT:-256}"
WIDTH="${WIDTH:-256}"
LATENT_DTYPE="${LATENT_DTYPE:-auto}"
OVERWRITE="${OVERWRITE:-1}"
LOG_DIR="${LOG_DIR:-${DATASET}/logs}"
CONDA_ENV="${CONDA_ENV:-lingbot-va}"

# --- Determine GPU list ---
if [[ -n "${GPU_IDS:-}" ]]; then
  # User explicitly passed GPU IDs (e.g. GPU_IDS="0 2 5")
  read -r -a GPU_ARRAY <<< "${GPU_IDS}"
else
  NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)}"
  if [[ "${NUM_GPUS}" -le 0 ]]; then
    echo "ERROR: no GPU detected. Set NUM_GPUS or GPU_IDS explicitly." >&2
    exit 1
  fi
  GPU_ARRAY=()
  for ((i=0; i<NUM_GPUS; i++)); do
    GPU_ARRAY+=("${i}")
  done
fi
NUM_GPUS="${#GPU_ARRAY[@]}"

# --- Count total episodes from meta/episodes.jsonl ---
EPISODES_JSONL="${DATASET}/meta/episodes.jsonl"
if [[ ! -f "${EPISODES_JSONL}" ]]; then
  echo "ERROR: ${EPISODES_JSONL} not found. Run conversion first." >&2
  exit 1
fi
TOTAL_EP=$(wc -l < "${EPISODES_JSONL}")
if [[ "${TOTAL_EP}" -le 0 ]]; then
  echo "ERROR: empty episodes.jsonl" >&2
  exit 1
fi

# --- Split episodes evenly across GPUs ---
PER_GPU=$(( (TOTAL_EP + NUM_GPUS - 1) / NUM_GPUS ))

mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"

echo "Dataset: ${DATASET}"
echo "Episodes: ${TOTAL_EP}, GPUs: ${NUM_GPUS} (${GPU_ARRAY[*]}), per-GPU: ${PER_GPU}"
echo "Logs: ${LOG_DIR}/step_latents_gpu*.log"
echo ""

PIDS=()
EP_RANGES=()
for (( idx=0; idx<NUM_GPUS; idx++ )); do
  gpu="${GPU_ARRAY[$idx]}"
  start=$(( idx * PER_GPU ))
  end=$(( (idx + 1) * PER_GPU - 1 ))
  if [[ "${end}" -ge "${TOTAL_EP}" ]]; then
    end=$(( TOTAL_EP - 1 ))
  fi
  if [[ "${start}" -gt "${end}" ]]; then
    # Not enough episodes for this GPU (e.g. NUM_GPUS > TOTAL_EP)
    continue
  fi
  episode_range="${start}-${end}"
  EP_RANGES+=("${episode_range}")
  log_file="${LOG_DIR}/step_latents_gpu${gpu}.log"

  extract_args=(
    --dataset "${DATASET}"
    --model-root "${MODEL_ROOT}"
    --target-fps "${TARGET_FPS}"
    --ori-fps "${ORI_FPS}"
    --height "${HEIGHT}"
    --width "${WIDTH}"
    --device "cuda:${gpu}"
    --episodes "${episode_range}"
    --dtype "${LATENT_DTYPE}"
  )
  if [[ "${OVERWRITE}" == "1" ]]; then
    extract_args+=(--overwrite)
  fi

  echo "[GPU ${gpu}] episodes ${episode_range} -> ${log_file}"
  conda run -n "${CONDA_ENV}" python tools/extract_dex_video_latents.py \
    "${extract_args[@]}" \
    > "${log_file}" 2>&1 &
  PIDS+=("$!")
done

echo ""
echo "Launched ${#PIDS[@]} worker(s). Waiting for completion ..."

FAIL_COUNT=0
for (( idx=0; idx<${#PIDS[@]}; idx++ )); do
  pid="${PIDS[$idx]}"
  gpu="${GPU_ARRAY[$idx]}"
  if wait "${pid}"; then
    echo "  GPU ${gpu} (pid ${pid}): OK"
  else
    echo "  GPU ${gpu} (pid ${pid}): FAILED (see ${LOG_DIR}/step_latents_gpu${gpu}.log)"
    FAIL_COUNT=$(( FAIL_COUNT + 1 ))
  fi
done

if [[ "${FAIL_COUNT}" -gt 0 ]]; then
  echo "ERROR: ${FAIL_COUNT} worker(s) failed." >&2
  exit 1
fi

echo ""
echo "All latent extraction workers completed successfully."
echo "Output: ${DATASET}/latents/"
