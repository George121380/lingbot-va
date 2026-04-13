#!/usr/bin/bash

set -euo pipefail
set -x

target_conda_env=${LINGBOT_VA_CONDA_ENV:-"lingbot-va"}
if [ "${LINGBOT_VA_SKIP_CONDA_RUN:-"0"}" != "1" ] && command -v conda >/dev/null 2>&1; then
    current_conda_env=${CONDA_DEFAULT_ENV:-""}
    if [ "${current_conda_env}" != "${target_conda_env}" ]; then
        export LINGBOT_VA_SKIP_CONDA_RUN=1
        exec conda run -n "${target_conda_env}" bash "$0" "$@"
    fi
fi

umask 007
 
NGPU=${NGPU:-"8"}
MASTER_PORT=${MASTER_PORT:-"29501"}
PORT=${PORT:-"1106"}
LOG_RANK=${LOG_RANK:-"0"}
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}
CONFIG_NAME=${CONFIG_NAME:-"robotwin_train"} # robotwin_train, libero_train

overrides=("$@")

if [ -n "${WANDB_API_KEY:-}" ]; then
    export WANDB_API_KEY
else
    unset WANDB_API_KEY || true
fi
if [ -n "${WANDB_BASE_URL:-}" ]; then
    export WANDB_BASE_URL
else
    unset WANDB_BASE_URL || true
fi
if [ -n "${WANDB_TEAM_NAME:-}" ]; then
    export WANDB_TEAM_NAME
else
    unset WANDB_TEAM_NAME || true
fi
export WANDB_PROJECT=${WANDB_PROJECT:-lingbot-va-posttrain}

## node setting
num_gpu=${NGPU}
master_port=${MASTER_PORT}
log_rank=${LOG_RANK}
torchft_lighthouse=${TORCHFT_LIGHTHOUSE}
config_name=${CONFIG_NAME}

## cmd setting
export TOKENIZERS_PARALLELISM=false
local_ranks_filter_args=()
if python -m torch.distributed.run --help 2>/dev/null | grep -q -- '--local-ranks-filter'; then
    local_ranks_filter_args+=("--local-ranks-filter=${log_rank}")
fi
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" TORCHFT_LIGHTHOUSE=${torchft_lighthouse} \
python -m torch.distributed.run \
    --nproc_per_node=${num_gpu} \
    "${local_ranks_filter_args[@]}" \
    --master_port ${master_port} \
    --tee 3 \
    -m wan_va.train --config-name ${config_name} "${overrides[@]}"
