# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import os

from easydict import EasyDict

from .va_dex_cfg import _load_norm_stat, va_dex_cfg

va_dex_train_cfg = EasyDict(__name__='Config: VA dex train')
va_dex_train_cfg.update(va_dex_cfg)

va_dex_train_cfg.dataset_path = os.getenv(
    "LINGBOT_VA_DEX_DATASET_PATH",
    va_dex_cfg.dataset_path,
)
va_dex_train_cfg.empty_emb_path = os.getenv(
    "LINGBOT_VA_DEX_EMPTY_EMB_PATH",
    os.path.join(va_dex_train_cfg.dataset_path, 'empty_emb.pt'),
)
va_dex_train_cfg.norm_stat = _load_norm_stat(va_dex_train_cfg.dataset_path)
va_dex_train_cfg.enable_wandb = os.getenv("LINGBOT_VA_ENABLE_WANDB", "1") != "0"
va_dex_train_cfg.load_worker = int(os.getenv("LINGBOT_VA_DEX_LOAD_WORKER", "16"))
va_dex_train_cfg.save_interval = int(os.getenv("LINGBOT_VA_DEX_SAVE_INTERVAL", "1000"))
va_dex_train_cfg.gc_interval = int(os.getenv("LINGBOT_VA_DEX_GC_INTERVAL", "50"))
va_dex_train_cfg.cfg_prob = float(os.getenv("LINGBOT_VA_DEX_CFG_PROB", "0.1"))

# Training parameters. The same config supports post-training and the
# action-layer-randomized pretraining experiment; the checkpoint directory
# chosen by wan22_pretrained_model_name_or_path determines the initialization.
va_dex_train_cfg.learning_rate = float(os.getenv("LINGBOT_VA_DEX_LR", "1e-5"))
va_dex_train_cfg.beta1 = 0.9
va_dex_train_cfg.beta2 = 0.95
va_dex_train_cfg.weight_decay = 0.1
va_dex_train_cfg.warmup_steps = int(os.getenv("LINGBOT_VA_DEX_WARMUP_STEPS", "10"))
va_dex_train_cfg.batch_size = int(os.getenv("LINGBOT_VA_DEX_BATCH_SIZE", "1"))
va_dex_train_cfg.gradient_accumulation_steps = int(
    os.getenv("LINGBOT_VA_DEX_GRAD_ACCUM", "1")
)
va_dex_train_cfg.num_steps = int(os.getenv("LINGBOT_VA_DEX_NUM_STEPS", "50000"))
