# LingBot-VA 本地训练说明

本文件记录当前服务器（B300 GPU）上的训练方法。

## 基本信息

仓库路径:

```bash
/share-2/code/fanqilin/peiqi/lingbot-va
```

数据路径:

```bash
/share-2/code/fanqilin/peiqi/dataset/robotwin-clean-and-aug-lerobot
```

预训练模型路径:

```bash
/share-2/code/fanqilin/peiqi/download/lingbot-va-posttrain-robotwin
```

## 硬件环境

- GPU: 8x NVIDIA B300 SXM6 AC (275GB VRAM, sm_103 架构)
- PyTorch: 2.11.0+cu128
- NCCL: 2.28.9+cuda12.9
- Conda 环境: `lingbot-va`

## 重要注意事项

- `script/run_va_posttrain.sh` 会自动进入 `lingbot-va` conda 环境，不需要手动 activate。
- 脚本内已包含 `NCCL_IB_DISABLE=1`，解决 B300 GPU 上 NCCL IB 传输的兼容问题。
- 训练前确保 `transformer/config.json` 中 `attn_mode=flex`；推理/评测前改回 `torch` 或 `flashattn`。
- `batch_size` 现在支持 > 1（通过自定义 `collate_variable_length` 对变长帧序列做 padding + attention mask）。增大 batch_size 可提高 GPU 显存利用率，但 padding 带来的额外计算量取决于同一 batch 内样本帧数差异。(peiqi: 测试中)
- 通过 `gradient_accumulation_steps` 模拟更大的有效 batch size（参考 GitHub issue #27）。有效 batch = GPU 数 × batch_size × gradient_accumulation_steps。

## 训练参数一览

| 参数 | 默认值 | 环境变量 | 说明 |
|------|--------|---------|------|
| batch_size | 1 | `LINGBOT_VA_BATCH_SIZE` | 支持 >= 1，增大可提高显存利用率 |
| gradient_accumulation_steps | 1 | `LINGBOT_VA_GRAD_ACCUM` | 有效 batch = GPU 数 × bs × accum |
| num_steps | 50000 | `LINGBOT_VA_NUM_STEPS` | 总训练步数 |
| learning_rate | 1e-5 | `LINGBOT_VA_LR` | 学习率 |
| warmup_steps | 10 | `LINGBOT_VA_WARMUP_STEPS` | 预热步数 |
| weight_decay | 0.1 | - | 硬编码 |
| save_interval | 1000 | `LINGBOT_VA_SAVE_INTERVAL` | 每 N 步保存 checkpoint |
| load_worker | 16 | `LINGBOT_VA_LOAD_WORKER` | DataLoader 工作进程数 |
| num_init_worker | 8 | `LINGBOT_VA_DATASET_INIT_WORKER` | 数据集初始化并行度 |
| allow_incomplete | False | `LINGBOT_VA_ALLOW_INCOMPLETE_DATASETS=1` | 跳过坏 shard |
| timing_log_interval | 50 | `LINGBOT_VA_TIMING_LOG_INTERVAL` | 每 N 步输出计时统计 |
| enable_timing | True | `LINGBOT_VA_ENABLE_TIMING=0` 关闭 | 子步骤计时开关 |

## 1. 检查数据完整性

```bash
cd /share-2/code/fanqilin/peiqi/lingbot-va

conda run -n lingbot-va python tools/check_robotwin_lerobot_dataset.py \
  --dataset /share-2/code/fanqilin/peiqi/dataset/robotwin-clean-and-aug-lerobot \
  --report /share-2/code/fanqilin/peiqi/lingbot-va/train_out/robotwin_dataset_check_full.json
```

通过则输出 `Validation passed for ...`；失败则报告写入指定 JSON 文件。

## 2. 正式 8 卡训练（推荐）

```bash
cd /share-2/code/fanqilin/peiqi/lingbot-va

export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800

WANDB_MODE=offline \
LINGBOT_VA_DATASET_PATH=/share-2/code/fanqilin/peiqi/dataset/robotwin-clean-and-aug-lerobot \
LINGBOT_VA_GRAD_ACCUM=8 \
LINGBOT_VA_ALLOW_INCOMPLETE_DATASETS=1 \
LINGBOT_VA_DATASET_INIT_REPORT_PATH=/share-2/code/fanqilin/peiqi/lingbot-va/train_out/robotwin_train/dataset_init_report.json \
NGPU=8 \
CONFIG_NAME=robotwin_train \
bash script/run_va_posttrain.sh \
  --save-root /share-2/code/fanqilin/peiqi/lingbot-va/train_out/robotwin_train
```

说明:

- `LINGBOT_VA_GRAD_ACCUM=8`: 8 GPU × bs=1 × 8 accum = 有效 batch 64，与论文原始 64 GPU 设定一致。也可以用 `LINGBOT_VA_BATCH_SIZE=2 LINGBOT_VA_GRAD_ACCUM=4` 等等价配置。
- `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800`: 首步 CUDA 编译/warmup 可能超过默认 480s 超时。
- `LINGBOT_VA_ALLOW_INCOMPLETE_DATASETS=1`: 当前数据集有少量 shard 问题，需要此选项跳过。
- 如不需要计时统计，可加 `LINGBOT_VA_ENABLE_TIMING=0`。
- Checkpoint 每 1000 步保存到 `--save-root` 下的 `checkpoints/` 目录。

也可以直接运行已经写好的启动脚本:

```bash
bash /share-2/code/fanqilin/peiqi/lingbot-va/train_out/run_training.sh
```

### 在 tmux 中后台运行

```bash
tmux new-session -d -s training -c /share-2/code/fanqilin/peiqi/lingbot-va
tmux send-keys -t training "bash train_out/run_training.sh" Enter
```

查看进度:

```bash
tmux attach -t training
```

## 3. 训练速度参考

在当前 B300 服务器上实测（bs=1, grad_accum=8）:

- 每 1000 optimizer step 约 1h55m
- 50000 步总计约 96 小时（4 天）
- GPU 显存峰值约 42GB / 275GB
- GPU 利用率 90-100%

增大 batch_size 可提高显存利用率和吞吐量，但需注意：同一 batch 内样本帧数差异越大，padding 浪费越多。建议根据实际显存余量逐步调大 batch_size（2、4、...）并观察 OOM 情况。

## 4. 在线 W&B

先登录:

```bash
wandb login
```

然后将训练命令中的 `WANDB_MODE=offline` 改为 `WANDB_MODE=online`。

如果已经完成了离线训练，可以事后同步:

```bash
wandb sync /share-2/code/fanqilin/peiqi/lingbot-va/wandb/offline-run-XXXXXXXX-XXXXXXXX
```

如果使用自建 wandb server:

```bash
export WANDB_BASE_URL=...
```

## 5. Smoke Test

### 5.1 8 卡快速验证（单任务子集）

```bash
cd /share-2/code/fanqilin/peiqi/lingbot-va

WANDB_MODE=offline \
LINGBOT_VA_DATASET_PATH=/share-2/code/fanqilin/peiqi/dataset/robotwin-clean-and-aug-lerobot/lerobot_robotwin_eef_aug_500/adjust_bottle-aloha-agilex_randomized_500-1000 \
LINGBOT_VA_EMPTY_EMB_PATH=/share-2/code/fanqilin/peiqi/dataset/robotwin-clean-and-aug-lerobot/empty_emb.pt \
LINGBOT_VA_ALLOW_INCOMPLETE_DATASETS=1 \
LINGBOT_VA_NUM_STEPS=2 \
LINGBOT_VA_SAVE_INTERVAL=1000 \
LINGBOT_VA_LOAD_WORKER=0 \
LINGBOT_VA_DATASET_INIT_WORKER=1 \
LINGBOT_VA_GRAD_ACCUM=8 \
NGPU=8 \
CONFIG_NAME=robotwin_train \
bash script/run_va_posttrain.sh \
  --save-root /share-2/code/fanqilin/peiqi/lingbot-va/train_out/smoke_8gpu
```

### 5.2 全数据集验证（1 步）

```bash
cd /share-2/code/fanqilin/peiqi/lingbot-va

export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800

WANDB_MODE=offline \
LINGBOT_VA_DATASET_PATH=/share-2/code/fanqilin/peiqi/dataset/robotwin-clean-and-aug-lerobot \
LINGBOT_VA_ALLOW_INCOMPLETE_DATASETS=1 \
LINGBOT_VA_NUM_STEPS=1 \
LINGBOT_VA_SAVE_INTERVAL=1000 \
LINGBOT_VA_LOAD_WORKER=0 \
LINGBOT_VA_DATASET_INIT_WORKER=4 \
LINGBOT_VA_GRAD_ACCUM=8 \
LINGBOT_VA_ENABLE_WANDB=0 \
NGPU=8 \
CONFIG_NAME=robotwin_train \
bash script/run_va_posttrain.sh \
  --save-root /share-2/code/fanqilin/peiqi/lingbot-va/train_out/smoke_full_dataset
```

## 6. 训练后做推理/评测

训练结束后，需要修改以下文件的 `attn_mode`:

**预训练模型:**

```bash
/share-2/code/fanqilin/peiqi/download/lingbot-va-posttrain-robotwin/transformer/config.json
```

**以及每个训练 checkpoint:**

```bash
train_out/robotwin_train/checkpoints/checkpoint_step_*/transformer/config.json
```

将:

```json
"attn_mode": "flex"
```

改为:

```json
"attn_mode": "torch"
```

批量修改所有 checkpoint:

```bash
find train_out/robotwin_train/checkpoints -name config.json -path "*/transformer/*" \
  -exec sed -i 's/"attn_mode": "flex"/"attn_mode": "torch"/g' {} \;
```

另外注意 RoboTwin 评测需要 `sapien==3.0.0b1` 版本（参考 issue #27）。

## 7. 子步骤计时统计

训练代码中已集成计时功能，默认开启。每 50 个 optimizer step 输出一次统计:

```
===== TIMING (step 50) STATS =====
  data_loading        : mean=0.0395s  ...
  data_to_device      : mean=0.0006s  ...
  prepare_input       : mean=0.0122s  ...
  forward             : mean=0.2700s  ...
  loss_and_backward   : mean=0.3200s  ...
  optimizer_step      : mean=0.8500s  ...
  barrier_sync        : mean=0.0138s  ...
  TOTAL_per_microbatch: mean=0.6400s  ...
==================================================
```

关闭计时: 设置 `LINGBOT_VA_ENABLE_TIMING=0`。
调整输出频率: 设置 `LINGBOT_VA_TIMING_LOG_INTERVAL=100`（每 100 步输出一次）。

## 8. 预期训练效果（来自 GitHub issue #27）

| 步数 | 成功率（demo_clean, 50 runs/task） |
|------|----------------------------------|
| 15000 | ~81% |
| 50000 | ~91% |

以上数据来自用户 jiachengliu3 使用 8 GPU + grad_accum=8 的复现结果。

## 9. B300 GPU 特殊说明

- **NCCL_IB_DISABLE=1**: 已写入 `script/run_va_posttrain.sh`，无需手动设置。B300 (sm_103) 上 NCCL IB 传输会导致 `RuntimeError: Invalid argument`。
- **显存利用**: bs=1 时约 42GB/275GB。现已支持更大 batch_size（通过 `collate_variable_length` 自动 padding 变长序列并在 FlexAttn 中 mask 掉 padding tokens），可按需调大以提高利用率。
- **TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC**: 首步 CUDA 编译耗时较长，需设为 1800 以避免 NCCL watchdog 误报。

## 10. batch_size > 1 支持说明

### 背景

原始代码中 `batch_size` 只能为 1，因为不同样本的视频帧数（F 维度）不同，PyTorch DataLoader 默认的 `torch.stack` 无法处理变长张量。该限制对应 GitHub issue #63。

### 实现方式

参考 issue #63 的讨论（pad to max length + attention mask），修改了以下文件：

| 文件 | 修改内容 |
|------|----------|
| `wan_va/dataset/lerobot_latent_dataset.py` | 新增 `collate_variable_length()` 函数：将 batch 内所有样本的 `latents`、`actions`、`actions_mask` pad 到 batch 内最大帧数，生成 `latent_mask`（标记有效/padding 帧）和 `latent_num_frames`/`action_num_frames`（每个样本的原始帧数） |
| `wan_va/train.py` | DataLoader 使用自定义 collate_fn；`_prepare_input_dict` 将 `latent_mask` 传入 `_add_noise` 以零化 padding 帧的 noise/target；`compute_loss` 中 loss 只对有效帧求均值 |
| `wan_va/modules/model.py` | `FlexAttnFunc.init_mask` 接收 per-sample 帧数，将 padding 帧的 `seq_id` 设为 -1（被现有 attention mask 逻辑排除）；`forward_train` 传递帧数元数据 |

### 关键设计

- **Attention masking**: padding 帧的 token 的 `seq_id = -1`，被 `seq_mask` 中的 `seq_ids >= 0` 条件排除，不参与任何 self-attention 和 cross-attention
- **Loss 归一化**: 只对有效帧求均值（而非包含 padding 帧的全部帧），确保 loss scale 不随 padding 量变化
- **向后兼容**: `batch_size=1` 时无 padding 发生，所有代码路径行为与修改前完全一致

### 注意事项

- 增大 batch_size 会增加显存用量（因 padding 带来额外的 token 计算）
- 同一 batch 内样本帧数差异越大，padding 浪费越多
- 建议逐步调大 batch_size 并监控显存使用和训练 loss，确认效果正常
- 增大 batch_size 后应相应减小 `gradient_accumulation_steps`，保持有效 batch size 不变

## 11. 详细调整日志

所有参数选择的完整理由和调试过程记录在:

```bash
/share-2/code/fanqilin/peiqi/lingbot-va/train_out/training_adjustments.log
```


