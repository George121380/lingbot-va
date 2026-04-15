#!/bin/bash
# Full training: batch_size=8, 8 GPUs, effective batch = 8×8×1 = 64
set -e
cd /share-2/code/fanqilin/peiqi/lingbot-va

export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export WANDB_MODE=online
export WANDB_API_KEY=$(conda run -n lingbot-va python -c "import wandb; print(wandb.api.api_key)" 2>/dev/null)
export LINGBOT_VA_MODEL_PATH=/share-2/code/fanqilin/peiqi/download/lingbot-va-base
export LINGBOT_VA_DATASET_PATH=/share-2/code/fanqilin/peiqi/dataset/robotwin-clean-and-aug-lerobot
export LINGBOT_VA_ALLOW_INCOMPLETE_DATASETS=1
export LINGBOT_VA_DATASET_INIT_REPORT_PATH=/share-2/code/fanqilin/peiqi/lingbot-va/train_out/robotwin_train_bs8/dataset_init_report.json
export LINGBOT_VA_BATCH_SIZE=8
export LINGBOT_VA_GRAD_ACCUM=1
export LINGBOT_VA_NUM_STEPS=50000
export LINGBOT_VA_SAVE_INTERVAL=1000
export NGPU=8
export CONFIG_NAME=robotwin_train

bash script/run_va_posttrain.sh \
  --save-root /share-2/code/fanqilin/peiqi/lingbot-va/train_out/robotwin_train_bs8
