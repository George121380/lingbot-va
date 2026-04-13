# LingBot-VA 本地训练说明

本文件记录当前这台机器上可直接使用的训练命令。

仓库路径:

```bash
/share-2/code/fanqilin/peiqi/lingbot-va
```

数据路径:

```bash
/share-2/code/fanqilin/peiqi/download/robotwin-clean-and-aug-lerobot
```

预训练模型路径:

```bash
/share-2/code/fanqilin/peiqi/download/lingbot-va-posttrain-robotwin
```

说明:

- `script/run_va_posttrain.sh` 现在会自动进入 `lingbot-va` conda 环境，不需要手动 `conda activate lingbot-va`。
- 当前训练配置默认是严格模式。只要数据里有坏 shard、缺失 latent、缺失 parquet，训练会直接报错，并生成报告文件。
- 当前训练用模型的 `transformer/config.json` 已经设置为 `attn_mode=flex`，适合训练。
- 如果训练完要做推理或评测，需要把 `attn_mode` 改回 `torch` 或 `flashattn`。

## 1. 先检查完整数据是否可用

完整数据下载完后，先运行:

```bash
cd /share-2/code/fanqilin/peiqi/lingbot-va

conda run -n lingbot-va python tools/check_robotwin_lerobot_dataset.py \
  --dataset /share-2/code/fanqilin/peiqi/download/robotwin-clean-and-aug-lerobot \
  --report /share-2/code/fanqilin/peiqi/lingbot-va/train_out/robotwin_dataset_check_full.json
```

如果数据没有问题，脚本会输出:

```bash
Validation passed for ...
```

如果有问题，脚本会直接列出具体缺失文件，并把完整报告写到:

```bash
/share-2/code/fanqilin/peiqi/lingbot-va/train_out/robotwin_dataset_check_full.json
```

## 2. 完整数据上的正式 8 卡训练

确认上一步检查通过后，运行:

```bash
cd /share-2/code/fanqilin/peiqi/lingbot-va

WANDB_MODE=offline \
NGPU=8 \
CONFIG_NAME=robotwin_train \
LINGBOT_VA_DATASET_INIT_REPORT_PATH=/share-2/code/fanqilin/peiqi/lingbot-va/train_out/robotwin_train/dataset_init_report.json \
bash script/run_va_posttrain.sh \
  --save-root /share-2/code/fanqilin/peiqi/lingbot-va/train_out/robotwin_train
```

说明:

- 这条命令默认读取:
  - 数据: `/share-2/code/fanqilin/peiqi/download/robotwin-clean-and-aug-lerobot`
  - 预训练模型: `/share-2/code/fanqilin/peiqi/download/lingbot-va-posttrain-robotwin`
- 如果初始化阶段发现坏 shard，会直接失败，并把问题写到:

```bash
/share-2/code/fanqilin/peiqi/lingbot-va/train_out/robotwin_train/dataset_init_report.json
```

## 3. 如果你要接在线 W&B

把下面几个环境变量换成你自己的真实值后再运行正式训练:

```bash
export WANDB_MODE=online
export WANDB_API_KEY=...
export WANDB_TEAM_NAME=...
export WANDB_PROJECT=lingbot-va-posttrain
# 如果你用自建 wandb server，再额外设置:
export WANDB_BASE_URL=...
```

然后执行第 2 节的训练命令即可。

## 4. 残缺数据上的 smoke test

如果数据还没下完，只想先验证训练链路是否能启动，可以运行:

```bash
cd /share-2/code/fanqilin/peiqi/lingbot-va

WANDB_MODE=offline \
LINGBOT_VA_ALLOW_INCOMPLETE_DATASETS=1 \
LINGBOT_VA_NUM_STEPS=1 \
LINGBOT_VA_SAVE_INTERVAL=1 \
LINGBOT_VA_LOAD_WORKER=0 \
LINGBOT_VA_DATASET_INIT_WORKER=1 \
NGPU=1 \
CONFIG_NAME=robotwin_train \
bash script/run_va_posttrain.sh \
  --save-root /share-2/code/fanqilin/peiqi/lingbot-va/train_out/smoke_robotwin_train
```

这条命令只用于 smoke test，不用于正式复现结果。

## 5. 8 卡 smoke test

如果你只想快速验证 8 卡分布式训练能否启动，可以用单任务子集跑 1 step:

```bash
cd /share-2/code/fanqilin/peiqi/lingbot-va

WANDB_MODE=offline \
LINGBOT_VA_DATASET_PATH=/share-2/code/fanqilin/peiqi/download/robotwin-clean-and-aug-lerobot/lerobot_robotwin_eef_aug_500/adjust_bottle-aloha-agilex_randomized_500-1000 \
LINGBOT_VA_EMPTY_EMB_PATH=/share-2/code/fanqilin/peiqi/download/robotwin-clean-and-aug-lerobot/empty_emb.pt \
LINGBOT_VA_NUM_STEPS=1 \
LINGBOT_VA_SAVE_INTERVAL=1000 \
LINGBOT_VA_LOAD_WORKER=0 \
LINGBOT_VA_DATASET_INIT_WORKER=1 \
NGPU=8 \
CONFIG_NAME=robotwin_train \
bash script/run_va_posttrain.sh \
  --save-root /share-2/code/fanqilin/peiqi/lingbot-va/train_out/smoke_robotwin_train_8gpu_single_task
```

## 6. 训练后做推理/评测前

训练结束后，如果你要切回推理或 RoboTwin 评测，先把下面文件里的:

```bash
/share-2/code/fanqilin/peiqi/download/lingbot-va-posttrain-robotwin/transformer/config.json
```

从:

```json
"attn_mode": "flex"
```

改回:

```json
"attn_mode": "torch"
```

或者:

```json
"attn_mode": "flashattn"
```
