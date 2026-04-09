# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import os
from pathlib import Path

import torch
from easydict import EasyDict


def _is_pretrained_model_root(path):
    required_files = (
        ("vae", "config.json"),
        ("text_encoder", "config.json"),
        ("tokenizer", "tokenizer_config.json"),
        ("transformer", "config.json"),
    )
    return all((path / subdir / filename).is_file()
               for subdir, filename in required_files)


def resolve_pretrained_model_path(*candidate_paths):
    env_path = (os.getenv("LINGBOT_VA_MODEL_PATH")
                or os.getenv("WAN22_PRETRAINED_MODEL_NAME_OR_PATH"))
    if env_path:
        return os.path.abspath(os.path.expanduser(env_path))

    repo_root = Path(__file__).resolve().parents[2]
    search_roots = (
        repo_root,
        repo_root / "download",
        repo_root.parent,
        repo_root.parent / "download",
        Path.cwd(),
        Path.cwd() / "download",
        Path.cwd().parent,
        Path.cwd().parent / "download",
    )

    for candidate_path in candidate_paths:
        expanded_path = Path(candidate_path).expanduser()
        if expanded_path.is_absolute() and _is_pretrained_model_root(
                expanded_path):
            return str(expanded_path.resolve())

        for root in search_roots:
            resolved_path = (expanded_path if expanded_path.is_absolute() else
                             (root / expanded_path))
            if _is_pretrained_model_root(resolved_path):
                return str(resolved_path.resolve())

    if candidate_paths:
        fallback_path = Path(candidate_paths[0]).expanduser()
        if fallback_path.is_absolute():
            return str(fallback_path)
        return str((repo_root.parent / "download" / fallback_path).resolve())

    return "/path/to/pretrained/model"

va_shared_cfg = EasyDict()

va_shared_cfg.host = '0.0.0.0'
va_shared_cfg.port = 29536

va_shared_cfg.param_dtype = torch.bfloat16
va_shared_cfg.save_root = './train_out'

va_shared_cfg.patch_size = (1, 2, 2)

va_shared_cfg.enable_offload = False
