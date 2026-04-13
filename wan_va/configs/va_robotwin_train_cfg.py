# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import os

from easydict import EasyDict

from .shared_config import resolve_dataset_path
from .va_robotwin_cfg import va_robotwin_cfg

va_robotwin_train_cfg = EasyDict(__name__='Config: VA robotwin train')
va_robotwin_train_cfg.update(va_robotwin_cfg)

va_robotwin_train_cfg.dataset_path = os.getenv(
    "LINGBOT_VA_ROBOTWIN_DATASET_PATH",
    os.getenv(
        "LINGBOT_VA_DATASET_PATH",
        resolve_dataset_path("robotwin-clean-and-aug-lerobot"),
    ),
)
va_robotwin_train_cfg.empty_emb_path = os.getenv(
    "LINGBOT_VA_ROBOTWIN_EMPTY_EMB_PATH",
    os.getenv(
        "LINGBOT_VA_EMPTY_EMB_PATH",
        os.path.join(va_robotwin_train_cfg.dataset_path, 'empty_emb.pt'),
    ),
)
va_robotwin_train_cfg.enable_wandb = os.getenv("LINGBOT_VA_ENABLE_WANDB",
                                               "1") != "0"
va_robotwin_train_cfg.load_worker = int(
    os.getenv("LINGBOT_VA_ROBOTWIN_LOAD_WORKER",
              os.getenv("LINGBOT_VA_LOAD_WORKER", "16")))
va_robotwin_train_cfg.num_init_worker = int(
    os.getenv("LINGBOT_VA_ROBOTWIN_DATASET_INIT_WORKER",
              os.getenv("LINGBOT_VA_DATASET_INIT_WORKER", "8")))
va_robotwin_train_cfg.allow_incomplete_datasets = os.getenv(
    "LINGBOT_VA_ROBOTWIN_ALLOW_INCOMPLETE_DATASETS",
    os.getenv("LINGBOT_VA_ALLOW_INCOMPLETE_DATASETS", "0"),
) == "1"
va_robotwin_train_cfg.dataset_init_report_path = os.getenv(
    "LINGBOT_VA_ROBOTWIN_DATASET_INIT_REPORT_PATH",
    os.getenv("LINGBOT_VA_DATASET_INIT_REPORT_PATH", ""),
)
va_robotwin_train_cfg.save_interval = int(
    os.getenv("LINGBOT_VA_ROBOTWIN_SAVE_INTERVAL",
              os.getenv("LINGBOT_VA_SAVE_INTERVAL", "1000")))
va_robotwin_train_cfg.gc_interval = int(
    os.getenv("LINGBOT_VA_ROBOTWIN_GC_INTERVAL",
              os.getenv("LINGBOT_VA_GC_INTERVAL", "50")))
va_robotwin_train_cfg.cfg_prob = float(
    os.getenv("LINGBOT_VA_ROBOTWIN_CFG_PROB",
              os.getenv("LINGBOT_VA_CFG_PROB", "0.1")))

# Training parameters
va_robotwin_train_cfg.learning_rate = float(
    os.getenv("LINGBOT_VA_ROBOTWIN_LR",
              os.getenv("LINGBOT_VA_LR", "1e-5")))
va_robotwin_train_cfg.beta1 = 0.9
va_robotwin_train_cfg.beta2 = 0.95
va_robotwin_train_cfg.weight_decay = 0.1
va_robotwin_train_cfg.warmup_steps = int(
    os.getenv("LINGBOT_VA_ROBOTWIN_WARMUP_STEPS",
              os.getenv("LINGBOT_VA_WARMUP_STEPS", "10")))
va_robotwin_train_cfg.batch_size = int(
    os.getenv("LINGBOT_VA_ROBOTWIN_BATCH_SIZE",
              os.getenv("LINGBOT_VA_BATCH_SIZE", "1")))
va_robotwin_train_cfg.gradient_accumulation_steps = int(
    os.getenv("LINGBOT_VA_ROBOTWIN_GRAD_ACCUM",
              os.getenv("LINGBOT_VA_GRAD_ACCUM", "1")))
va_robotwin_train_cfg.num_steps = int(
    os.getenv("LINGBOT_VA_ROBOTWIN_NUM_STEPS",
              os.getenv("LINGBOT_VA_NUM_STEPS", "50000")))
