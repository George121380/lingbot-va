#!/usr/bin/env python3
"""Prepare a LingBot-VA checkpoint whose transformer action_dim is 58."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


ACTION_KEYS = {
    "action_embedder.weight",
    "action_embedder.bias",
    "action_proj_out.weight",
    "action_proj_out.bias",
}


def random_tensor(shape, *, dtype, device, std: float, generator: torch.Generator) -> torch.Tensor:
    value = torch.empty(shape, dtype=torch.float32, device=device)
    value.normal_(mean=0.0, std=std, generator=generator)
    return value.to(dtype=dtype)


def random_like_resized(old: torch.Tensor, shape, generator: torch.Generator) -> torch.Tensor:
    std = float(old.float().std().item()) if old.numel() > 1 else 0.02
    if not torch.isfinite(torch.tensor(std)) or std < 1e-6:
        std = 0.02
    return random_tensor(shape, dtype=old.dtype, device=old.device, std=std, generator=generator)


def resize_action_state(state: dict[str, torch.Tensor], action_dim: int, mode: str, seed: int) -> dict[str, torch.Tensor]:
    required = ["action_embedder.weight", "action_proj_out.weight", "action_proj_out.bias"]
    missing = [key for key in required if key not in state]
    if missing:
        raise KeyError(f"Transformer state is missing action keys: {missing}")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    old_embed_w = state["action_embedder.weight"]
    old_proj_w = state["action_proj_out.weight"]
    old_proj_b = state["action_proj_out.bias"]
    old_action_dim = old_embed_w.shape[1]
    inner_dim = old_embed_w.shape[0]

    new_embed_w = random_like_resized(old_embed_w, (inner_dim, action_dim), generator)
    new_proj_w = random_like_resized(old_proj_w, (action_dim, old_proj_w.shape[1]), generator)
    new_proj_b = torch.zeros((action_dim,), dtype=old_proj_b.dtype, device=old_proj_b.device)

    if mode == "finetune_init":
        copy_dim = min(old_action_dim, action_dim)
        new_embed_w[:, :copy_dim] = old_embed_w[:, :copy_dim]
        new_proj_w[:copy_dim, :] = old_proj_w[:copy_dim, :]
        new_proj_b[:copy_dim] = old_proj_b[:copy_dim]

    state["action_embedder.weight"] = new_embed_w
    if "action_embedder.bias" in state and mode == "pretrain_random_action":
        state["action_embedder.bias"] = torch.zeros_like(state["action_embedder.bias"])
    state["action_proj_out.weight"] = new_proj_w
    state["action_proj_out.bias"] = new_proj_b
    return state


def resolve_model_dirs(src: Path, dst: Path) -> tuple[Path, Path]:
    if (src / "transformer").is_dir():
        return src, dst
    if src.name == "transformer" and src.is_dir():
        return src, dst / "transformer"
    raise FileNotFoundError(f"--src must be a model root containing transformer/ or a transformer dir: {src}")


def prepare(args: argparse.Namespace) -> None:
    src = Path(args.src).expanduser().resolve()
    dst = Path(args.dst).expanduser().resolve()
    copy_src, copy_dst = resolve_model_dirs(src, dst)

    if dst.exists():
        if not args.overwrite:
            raise FileExistsError(f"Destination exists: {dst}. Pass --overwrite to replace it.")
        shutil.rmtree(dst)
    copy_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(copy_src, copy_dst)

    transformer_dir = dst / "transformer" if (dst / "transformer").is_dir() else dst
    config_path = transformer_dir / "config.json"
    weight_path = transformer_dir / "diffusion_pytorch_model.safetensors"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing transformer config: {config_path}")
    if not weight_path.is_file():
        raise FileNotFoundError(f"Only safetensors transformer weights are supported, missing: {weight_path}")

    state = load_file(str(weight_path), device="cpu")
    state = resize_action_state(state, args.action_dim, args.mode, args.seed)
    save_file(state, str(weight_path))

    config = json.load(config_path.open("r"))
    config["action_dim"] = args.action_dim
    config.pop("_name_or_path", None)
    with config_path.open("w") as f:
        json.dump(config, f, indent=2)

    print(f"Wrote {args.action_dim}-dim action checkpoint to {dst}")
    print(f"Initialization mode: {args.mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="Source LingBot-VA model root or transformer dir.")
    parser.add_argument("--dst", required=True, help="Destination model root.")
    parser.add_argument("--action-dim", type=int, default=58)
    parser.add_argument(
        "--mode",
        choices=["finetune_init", "pretrain_random_action"],
        default="pretrain_random_action",
        help="How to initialize action_embedder/action_proj_out.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    prepare(parse_args())


if __name__ == "__main__":
    main()
