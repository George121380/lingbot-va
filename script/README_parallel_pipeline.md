# Parallel Dex Dataset Conversion Pipeline

对 `convert_dex_zarr_to_lerobot.py` + `extract_dex_video_latents.py` 加入**多进程 / 多 GPU 并行**的封装脚本。

---

## 文件清单

| 文件 | 作用 | 并行策略 |
|------|------|---------|
| `run_convert_our_dex_lerobot_parallel.sh` | 仅做 zarr → LeRobot 格式转换 + 基础校验 | 32 CPU 进程（per-episode 并行） |
| `run_extract_latents_multi_gpu.sh` | 仅做 VAE latent 提取 | N GPU 并行（每 GPU 一个进程） |
| `run_prepare_our_dex_full_dataset_parallel.sh` | 端到端：转换 → latent → 校验 | 以上两个串起来 |

如果想退回单进程行为（比如调试），给 convert 脚本传 `WORKERS=1` 即可；latent 提取传 `NUM_GPUS=1` 即可。

---

## 快速用法

### 端到端（最常用）

```bash
cd /share-2/code/fanqilin/peiqi/lingbot-va

SOURCE_ROOT=/share-2/code/fanqilin/Teleop-franka-test/data_leo_teleop/pour_egg_0407/processed/all_joint_episode_cache_flat_zarr_resize \
OUTPUT_ROOT=/share-2/code/fanqilin/peiqi/dataset/converted_ours_dataset \
TASK_TEXT="pour egg" \
SPLITS="train,eval" \
WORKERS=32 \
  bash script/run_prepare_our_dex_full_dataset_parallel.sh
```

脚本会自动探测可用 GPU 数并全部用上。如只想用部分 GPU：

```bash
NUM_GPUS=4                        # 用 cuda:0 ~ cuda:3
# 或
GPU_IDS="0 2 5 7"                 # 精确指定哪几块
```

### 只跑转换

```bash
SOURCE_ROOT=... OUTPUT_ROOT=... WORKERS=32 \
  bash script/run_convert_our_dex_lerobot_parallel.sh
```

### 只跑 latent 提取（假设转换已完成）

```bash
DATASET=/share-2/code/fanqilin/peiqi/dataset/converted_ours_dataset \
MODEL_ROOT=/share-2/code/fanqilin/peiqi/download/lingbot-va-base \
NUM_GPUS=8 \
  bash script/run_extract_latents_multi_gpu.sh
```

---

## 并行实现细节

### Step 1: zarr → LeRobot (多进程 CPU)

**代码改动**: `tools/convert_dex_zarr_to_lerobot.py` 里加了 `--workers N` 参数和 `_process_episode()` worker 函数。

流程：
1. 主进程串行扫 zarr，构建全部 episode 的 job 列表（`episode_index`、`global_start`、边界）
2. `ProcessPoolExecutor` 并行处理每个 episode：读 zarr 切片 → 计算 action/state → 写 parquet + 3 路 MP4
3. 主进程按 `episode_index` 顺序串行写 metadata（`episodes.jsonl` 等），保证文件顺序一致

**为什么能并行**: zarr 支持并发只读；每个 episode 的输出路径不同，无写冲突。

### Step 2: VAE latent (多 GPU)

**代码改动**: 无。完全复用 `extract_dex_video_latents.py` 原有的 `--device cuda:N` 和 `--episodes START-END` 两个 CLI 参数。

流程（shell 实现）:
1. 从 `meta/episodes.jsonl` 算出总 episode 数，按 GPU 数均分
2. 对每块 GPU 起一个独立 `python tools/extract_dex_video_latents.py ... --device cuda:N --episodes start-end &`
3. `wait` 等所有进程结束，任一 worker 失败则整体失败

每个 GPU 的 stdout/stderr 独立写到 `$DATASET/logs/step_latents_gpu{N}.log`。

---

## 性能参考（pour_egg_0407 数据集，373 episodes / 718K frames）

| 步骤 | 单进程 | 并行 | 加速比 |
|------|--------|------|--------|
| Step 1 (convert) | ~50 min | ~3.5 min (32 workers) | **~14x** |
| Step 2 (latents) | ~30 min | ~3 min (8 GPUs) | **~8x** |

---

## 环境变量参考

### 通用

| 变量 | 默认 | 说明 |
|------|------|------|
| `SOURCE_ROOT` | （示例路径） | 输入 zarr root |
| `OUTPUT_ROOT` | （示例路径） | 输出 LeRobot dataset 目录 |
| `MODEL_ROOT` | `/share-2/code/fanqilin/peiqi/download/lingbot-va-base` | VAE / text_encoder / tokenizer 所在目录 |
| `TASK_TEXT` | `pour egg` | 任务文本，写入 metadata 与 latent 的 text_emb |
| `SPLITS` | `train` | 逗号分隔，如 `"train,eval"` |
| `FPS` | `30` | 源视频 FPS |
| `HEIGHT` / `WIDTH` | `256` | 输出视频 / VAE 输入分辨率 |
| `ACTION_TYPE` | `relative` | `relative` 或 `absolute` |

### Step 1 专有

| 变量 | 默认 | 说明 |
|------|------|------|
| `WORKERS` | `32` | CPU 并行进程数，太大反而争抢 I/O 带宽 |
| `OVERWRITE` | `1` | 覆盖已有 OUTPUT_ROOT |
| `PRESERVE_LATENTS` | `1` | overwrite 时保留 `latents/` 子目录（避免重复提取） |
| `SOURCE_BGR` | `0` | 源 zarr 图像是否 BGR（默认 RGB） |

### Step 2 专有

| 变量 | 默认 | 说明 |
|------|------|------|
| `NUM_GPUS` | 自动探测 | 使用 GPU 数量 |
| `GPU_IDS` | - | 手动指定 GPU 列表，覆盖 `NUM_GPUS`，如 `"0 2 5 7"` |
| `TARGET_FPS` | `15` | latent 采样帧率（必须整除 ORI_FPS） |
| `ORI_FPS` | `30` | 原始视频 FPS |
| `LATENT_DTYPE` | `auto` | `auto` / `bfloat16` / `float16` / `float32` |
| `OVERWRITE` | `1` | 覆盖已有 latent 文件 |
| `LOG_DIR` | `$DATASET/logs` | 各 GPU 日志目录 |
| `CONDA_ENV` | `lingbot-va` | 含 torch + diffusers 的 conda env |

---

## 日志

所有阶段 stdout/stderr 写到 `$OUTPUT_ROOT/logs/`：

```
logs/
├── step1_convert.log         # 当调用 parallel convert 脚本单独跑时
├── step_latents_gpu0.log     # 每块 GPU 独立日志
├── step_latents_gpu1.log
├── ...
└── step_latents_gpu{N-1}.log
```

注意 `OVERWRITE=1` 会清空 `$OUTPUT_ROOT` 下除 `latents/` 外的全部内容，包括旧 `logs/`。建议每次运行后把需要保留的日志归档到别处。
