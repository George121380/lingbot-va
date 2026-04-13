#!/usr/bin/env python3
"""Extract LingBot-VA Wan2.2 VAE latents for a converted dex LeRobot dataset.

This script consumes the dataset produced by convert_dex_zarr_to_lerobot.py and
writes per-camera latent .pth files under latents/, matching
MultiLatentLeRobotDataset's file lookup convention.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

try:
    import cv2
except ImportError as exc:  # pragma: no cover - checked at runtime.
    raise ImportError(
        "extract_dex_video_latents.py requires cv2 for mp4 decoding. "
        "Install opencv-python-headless in the environment used for latent extraction."
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan_va.modules.utils import load_text_encoder, load_tokenizer, load_vae  # noqa: E402

try:  # noqa: E402
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
except Exception:  # pragma: no cover - depends on diffusers version.

    def prompt_clean(text: str) -> str:
        return text


DEFAULT_MODEL_ROOT = "/share-2/code/fanqilin/peiqi/download/lingbot-va-base"
REQUIRED_LATENT_KEYS = {
    "latent",
    "latent_num_frames",
    "latent_height",
    "latent_width",
    "video_num_frames",
    "video_height",
    "video_width",
    "text_emb",
    "text",
    "frame_ids",
    "start_frame",
    "end_frame",
    "fps",
    "ori_fps",
}


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_episode_filter(raw: str | None) -> set[int] | None:
    if not raw:
        return None
    out: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
            if end < start:
                raise ValueError(f"Invalid episode range: {part}")
            out.update(range(start, end + 1))
        else:
            out.add(int(part))
    return out


def get_torch_dtype(name: str) -> torch.dtype:
    if name == "auto":
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def resolve_model_root(raw: str | None) -> Path:
    model_root = Path(
        raw
        or os.getenv("LINGBOT_VA_MODEL_PATH")
        or os.getenv("WAN22_PRETRAINED_MODEL_NAME_OR_PATH")
        or DEFAULT_MODEL_ROOT
    ).expanduser().resolve()
    required = [
        model_root / "vae" / "config.json",
        model_root / "text_encoder" / "config.json",
        model_root / "tokenizer" / "tokenizer_config.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Invalid model root for latent extraction. Missing:\n"
            + "\n".join(missing)
        )
    return model_root


def collect_video_keys(info: dict, requested: Iterable[str] | None) -> list[str]:
    if requested:
        return list(requested)
    return [
        key
        for key, value in info["features"].items()
        if isinstance(value, dict) and value.get("dtype") == "video"
    ]


def make_frame_ids(
    *,
    start_frame: int,
    end_frame: int,
    ori_fps: int,
    target_fps: int,
) -> np.ndarray:
    if end_frame <= start_frame:
        raise ValueError(
            f"Invalid action_config frame range: {start_frame}:{end_frame}"
        )
    if target_fps <= 0 or ori_fps <= 0:
        raise ValueError(f"fps values must be positive, got {target_fps}/{ori_fps}")
    if target_fps > ori_fps:
        raise ValueError(
            f"target_fps={target_fps} cannot exceed ori_fps={ori_fps}"
        )
    if ori_fps % target_fps != 0:
        raise ValueError(
            "LingBot-VA action alignment expects a constant integer frame stride; "
            f"got ori_fps={ori_fps}, target_fps={target_fps}."
        )

    stride = ori_fps // target_fps
    frame_ids = np.arange(start_frame, end_frame, stride, dtype=np.int64)
    if frame_ids.size == 0:
        raise ValueError(
            f"No sampled frames for action_config range {start_frame}:{end_frame}"
        )

    # Wan2.2 causal VAE has temporal compression 4x. The released example
    # stores sampled video lengths as 4m+1 so latent frames align with
    # (video_num_frames - 1) // 4 + 1.
    usable = ((frame_ids.size - 1) // 4) * 4 + 1
    frame_ids = frame_ids[:usable]
    return frame_ids


def read_sampled_video(
    video_path: Path,
    frame_ids: np.ndarray,
    *,
    height: int,
    width: int,
) -> tuple[torch.Tensor, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    target_pos = 0
    cur_frame = 0
    frames: list[np.ndarray] = []
    try:
        while target_pos < len(frame_ids):
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if cur_frame == int(frame_ids[target_pos]):
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                if frame_rgb.shape[0] != height or frame_rgb.shape[1] != width:
                    frame_rgb = cv2.resize(
                        frame_rgb,
                        (width, height),
                        interpolation=cv2.INTER_LINEAR,
                    )
                frames.append(np.ascontiguousarray(frame_rgb))
                target_pos += 1
            cur_frame += 1
    finally:
        cap.release()

    if len(frames) != len(frame_ids):
        raise RuntimeError(
            f"{video_path}: decoded {len(frames)} sampled frames, "
            f"expected {len(frame_ids)}. Last requested frame: {int(frame_ids[-1])}"
        )

    video = torch.from_numpy(np.stack(frames)).float().permute(3, 0, 1, 2)
    if video.shape[-2:] != (height, width):
        video = F.interpolate(
            video,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
    return video.unsqueeze(0), source_height, source_width


@torch.no_grad()
def encode_text(
    *,
    text: str,
    tokenizer,
    text_encoder,
    device: torch.device,
    dtype: torch.dtype,
    max_sequence_length: int,
) -> torch.Tensor:
    clean_text = prompt_clean(text)
    text_inputs = tokenizer(
        [clean_text],
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids.to(device)
    mask = text_inputs.attention_mask.to(device)
    seq_lens = mask.gt(0).sum(dim=1).long()
    prompt_embeds = text_encoder(input_ids, mask).last_hidden_state
    prompt_embeds = prompt_embeds.to(dtype=dtype)
    prompt_embeds = torch.stack(
        [
            torch.cat(
                [
                    emb[:seq_len],
                    emb.new_zeros(max_sequence_length - seq_len, emb.size(1)),
                ]
            )
            for emb, seq_len in zip(prompt_embeds, seq_lens)
        ],
        dim=0,
    )
    return prompt_embeds[0].to(torch.bfloat16).cpu()


@torch.no_grad()
def encode_video_latent(
    *,
    video: torch.Tensor,
    vae,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    video = video.to(device=device, dtype=dtype) / 255.0 * 2.0 - 1.0
    if hasattr(vae, "_encode"):
        # AutoencoderKLWan._encode implements the official causal streaming
        # schedule: first 1 frame, then chunks of 4 frames. Calling the encoder
        # once through WanVAEStreamingWrapper on a long sequence leaves the first
        # temporal downsample block un-applied and creates shape mismatches.
        enc_out = vae._encode(video)
    else:  # pragma: no cover - fallback for future diffusers API changes.
        enc_out = vae.encode(video).latent_dist.parameters
    mu, _logvar = torch.chunk(enc_out, 2, dim=1)
    latents_mean = torch.tensor(
        vae.config.latents_mean,
        device=mu.device,
        dtype=torch.float32,
    )
    latents_std = torch.tensor(
        vae.config.latents_std,
        device=mu.device,
        dtype=torch.float32,
    )
    mu_norm = (mu.float() - latents_mean.view(1, -1, 1, 1, 1)) / (
        latents_std.view(1, -1, 1, 1, 1) + 1e-8
    )
    return mu_norm[0].to(torch.bfloat16).cpu()


def flatten_latent(latent_chw: torch.Tensor) -> torch.Tensor:
    if latent_chw.ndim != 4:
        raise ValueError(f"Expected latent [C,F,H,W], got {tuple(latent_chw.shape)}")
    latent_fhwc = latent_chw.permute(1, 2, 3, 0).contiguous()
    return latent_fhwc.view(-1, latent_fhwc.shape[-1])


def action_text_for_segment(episode: dict, action_cfg: dict) -> str:
    text = action_cfg.get("action_text")
    if text:
        return text
    tasks = episode.get("tasks") or []
    if tasks:
        return tasks[0]
    return ""


def should_skip(path: Path, overwrite: bool) -> bool:
    if overwrite or not path.is_file():
        return False
    data = torch.load(path, map_location="cpu", weights_only=False)
    missing = REQUIRED_LATENT_KEYS - set(data.keys())
    if missing:
        raise AssertionError(f"Existing latent is incomplete: {path}, missing={missing}")
    return True


def extract_latents(args: argparse.Namespace) -> None:
    dataset = Path(args.dataset).expanduser().resolve()
    info = json.load((dataset / "meta" / "info.json").open("r"))
    episodes = load_jsonl(dataset / "meta" / "episodes.jsonl")
    video_keys = collect_video_keys(info, args.video_key)
    if not video_keys:
        raise ValueError("No video features found in meta/info.json")

    ori_fps = int(args.ori_fps or info["fps"])
    target_fps = int(args.target_fps)
    height = int(args.height or info["features"][video_keys[0]]["shape"][1])
    width = int(args.width or info["features"][video_keys[0]]["shape"][2])
    episode_filter = parse_episode_filter(args.episodes)

    model_root = resolve_model_root(args.model_root)
    dtype = get_torch_dtype(args.dtype)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    print(f"Dataset: {dataset}")
    print(f"Model root: {model_root}")
    print(f"Device/dtype: {device}/{dtype}")
    print(f"Latent fps: {target_fps} from ori_fps={ori_fps}")
    print(f"Video size for VAE: {height}x{width}")
    print(f"Video keys: {', '.join(video_keys)}")

    tokenizer = load_tokenizer(str(model_root / "tokenizer"))
    text_encoder = load_text_encoder(str(model_root / "text_encoder"), dtype, device)
    text_encoder.eval()
    vae = load_vae(str(model_root / "vae"), dtype, device)
    vae.eval()

    text_cache: dict[str, torch.Tensor] = {}
    written = 0
    skipped = 0

    for episode in episodes:
        episode_index = int(episode["episode_index"])
        if episode_filter is not None and episode_index not in episode_filter:
            continue
        chunk = episode_index // int(info["chunks_size"])
        episode_length = int(episode["length"])
        for action_cfg in episode["action_config"]:
            start_frame = int(action_cfg["start_frame"])
            end_frame = int(action_cfg["end_frame"])
            if start_frame < 0 or end_frame > episode_length:
                raise ValueError(
                    f"Episode {episode_index}: action_config {start_frame}:{end_frame} "
                    f"outside episode length {episode_length}"
                )
            frame_ids = make_frame_ids(
                start_frame=start_frame,
                end_frame=end_frame,
                ori_fps=ori_fps,
                target_fps=target_fps,
            )
            expected_latent_frames = (len(frame_ids) - 1) // 4 + 1
            text = action_text_for_segment(episode, action_cfg)
            if text not in text_cache:
                text_cache[text] = encode_text(
                    text=text,
                    tokenizer=tokenizer,
                    text_encoder=text_encoder,
                    device=device,
                    dtype=dtype,
                    max_sequence_length=args.max_sequence_length,
                )
            text_emb = text_cache[text]

            for video_key in video_keys:
                video_path = dataset / info["video_path"].format(
                    episode_chunk=chunk,
                    video_key=video_key,
                    episode_index=episode_index,
                )
                if not video_path.is_file():
                    raise FileNotFoundError(f"Missing video: {video_path}")
                latent_path = (
                    dataset
                    / "latents"
                    / f"chunk-{chunk:03d}"
                    / video_key
                    / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
                )
                if should_skip(latent_path, args.overwrite):
                    skipped += 1
                    continue

                video, source_height, source_width = read_sampled_video(
                    video_path,
                    frame_ids,
                    height=height,
                    width=width,
                )
                latent_chw = encode_video_latent(
                    video=video,
                    vae=vae,
                    device=device,
                    dtype=dtype,
                )
                latent_num_frames = int(latent_chw.shape[1])
                latent_height = int(latent_chw.shape[2])
                latent_width = int(latent_chw.shape[3])
                if latent_num_frames != expected_latent_frames:
                    raise AssertionError(
                        f"{video_path}: latent frames {latent_num_frames} != "
                        f"expected {expected_latent_frames} from sampled frames {len(frame_ids)}"
                    )

                latent_data = {
                    "latent": flatten_latent(latent_chw),
                    "latent_num_frames": latent_num_frames,
                    "latent_height": latent_height,
                    "latent_width": latent_width,
                    "video_num_frames": int(len(frame_ids)),
                    "video_height": int(height if args.height else source_height),
                    "video_width": int(width if args.width else source_width),
                    "text_emb": text_emb,
                    "text": text,
                    "frame_ids": frame_ids,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "fps": target_fps,
                    "ori_fps": ori_fps,
                }
                latent_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(latent_data, latent_path)
                written += 1
                print(
                    f"wrote {latent_path.relative_to(dataset)} "
                    f"frames={len(frame_ids)} latent=({latent_num_frames},{latent_height},{latent_width})"
                )

    print(f"Done. written={written} skipped={skipped}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Converted LeRobot dataset root.")
    parser.add_argument("--model-root", default=None, help="Wan2.2/LingBot-VA model root containing vae/text_encoder/tokenizer.")
    parser.add_argument("--target-fps", type=int, default=15, help="FPS used for latent extraction.")
    parser.add_argument("--ori-fps", type=int, default=None, help="Original episode fps. Defaults to meta/info.json fps.")
    parser.add_argument("--height", type=int, default=None, help="VAE input height. Defaults to dataset video height.")
    parser.add_argument("--width", type=int, default=None, help="VAE input width. Defaults to dataset video width.")
    parser.add_argument("--device", default=None, help="Torch device. Defaults to cuda if available, else cpu.")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--video-key", action="append", help="Limit to one video key. Can be passed multiple times.")
    parser.add_argument("--episodes", default=None, help="Optional episode filter, e.g. '0,2-4'.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing latent files.")
    return parser.parse_args()


def main() -> None:
    extract_latents(parse_args())


if __name__ == "__main__":
    main()
