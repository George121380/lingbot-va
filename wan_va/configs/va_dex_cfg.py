# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import json
import os

from easydict import EasyDict

from .shared_config import resolve_pretrained_model_path, va_shared_cfg


def _load_norm_stat(dataset_path):
    stats_path = os.getenv(
        "LINGBOT_VA_DEX_NORM_STATS_PATH",
        os.path.join(dataset_path, "meta", "dex_action_norm.json"),
    )
    if os.path.isfile(stats_path):
        with open(stats_path, "r") as f:
            stats = json.load(f)
        return {
            "q01": stats["q01"],
            "q99": stats["q99"],
        }

    return {
        "q01": [-1.0] * 58,
        "q99": [1.0] * 58,
    }


va_dex_cfg = EasyDict(__name__='Config: VA dex')
va_dex_cfg.update(va_shared_cfg)

va_dex_cfg.dataset_path = os.getenv(
    "LINGBOT_VA_DEX_DATASET_PATH",
    "/path/to/your/dex_lerobot_dataset",
)

va_dex_cfg.wan22_pretrained_model_name_or_path = resolve_pretrained_model_path(
    os.getenv("LINGBOT_VA_DEX_MODEL_NAME", "lingbot-va-dex-action58"),
    "lingbot-va-dex-random-action58",
    "lingbot-va-base",
)

va_dex_cfg.attn_window = 72
va_dex_cfg.frame_chunk_size = 2
va_dex_cfg.env_type = 'dex_zarr'

va_dex_cfg.height = 256
va_dex_cfg.width = 256
va_dex_cfg.action_dim = 58
va_dex_cfg.action_per_frame = int(os.getenv("LINGBOT_VA_DEX_ACTION_PER_FRAME", "8"))
va_dex_cfg.obs_cam_keys = [
    'observation.images.top',
    'observation.images.left_wrist',
    'observation.images.right_wrist',
]
va_dex_cfg.guidance_scale = 5
va_dex_cfg.action_guidance_scale = 1

va_dex_cfg.num_inference_steps = 25
va_dex_cfg.video_exec_step = -1
va_dex_cfg.action_num_inference_steps = 50

va_dex_cfg.snr_shift = 5.0
va_dex_cfg.action_snr_shift = 1.0

va_dex_cfg.used_action_channel_ids = list(range(va_dex_cfg.action_dim))
va_dex_cfg.inverse_used_action_channel_ids = list(range(va_dex_cfg.action_dim))

va_dex_cfg.action_norm_method = 'quantiles'
va_dex_cfg.norm_stat = _load_norm_stat(va_dex_cfg.dataset_path)
