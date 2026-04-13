#!/usr/bin/env python3
"""Check a converted dex LeRobot dataset for LingBot-VA training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq


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


def check_video(path: Path, expected_frames: int, expected_fps: int, expected_hw: tuple[int, int]) -> None:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise AssertionError(f"Video cannot be opened: {path}")
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if frames != expected_frames:
        raise AssertionError(f"{path}: frame count {frames} != {expected_frames}")
    if abs(fps - expected_fps) > 0.2:
        raise AssertionError(f"{path}: fps {fps} != {expected_fps}")
    if (height, width) != expected_hw:
        raise AssertionError(f"{path}: size {(height, width)} != {expected_hw}")


def check_latent(path: Path, action_cfg: dict, expected_hw: tuple[int, int]) -> None:
    import torch

    data = torch.load(path, map_location="cpu", weights_only=False)
    missing = REQUIRED_LATENT_KEYS - set(data.keys())
    if missing:
        raise AssertionError(f"{path}: missing latent keys {sorted(missing)}")

    start_frame = int(action_cfg["start_frame"])
    end_frame = int(action_cfg["end_frame"])
    if int(data["start_frame"]) != start_frame or int(data["end_frame"]) != end_frame:
        raise AssertionError(
            f"{path}: stored frame range {data['start_frame']}:{data['end_frame']} "
            f"!= action_config {start_frame}:{end_frame}"
        )

    frame_ids = np.asarray(data["frame_ids"], dtype=np.int64)
    if frame_ids.ndim != 1 or len(frame_ids) == 0:
        raise AssertionError(f"{path}: frame_ids must be a non-empty 1D array")
    if int(frame_ids[0]) != start_frame:
        raise AssertionError(f"{path}: first frame_id {frame_ids[0]} != {start_frame}")
    if int(frame_ids[-1]) >= end_frame:
        raise AssertionError(f"{path}: last frame_id {frame_ids[-1]} >= end_frame {end_frame}")
    if len(frame_ids) % 4 != 1:
        raise AssertionError(f"{path}: sampled frame count {len(frame_ids)} is not 4m+1")
    if int(data["video_num_frames"]) != len(frame_ids):
        raise AssertionError(
            f"{path}: video_num_frames {data['video_num_frames']} != len(frame_ids) {len(frame_ids)}"
        )
    if len(frame_ids) > 1:
        diffs = np.diff(frame_ids)
        if not np.all(diffs == diffs[0]):
            raise AssertionError(f"{path}: frame_ids are not sampled with a constant stride")
        ori_fps = int(data["ori_fps"])
        fps = int(data["fps"])
        if ori_fps % fps == 0 and int(diffs[0]) != ori_fps // fps:
            raise AssertionError(
                f"{path}: frame stride {diffs[0]} != ori_fps/fps {ori_fps // fps}"
            )

    latent_num_frames = int(data["latent_num_frames"])
    expected_latent_frames = (len(frame_ids) - 1) // 4 + 1
    if latent_num_frames != expected_latent_frames:
        raise AssertionError(
            f"{path}: latent_num_frames {latent_num_frames} != {expected_latent_frames}"
        )

    latent = data["latent"]
    if not torch.is_tensor(latent):
        raise AssertionError(f"{path}: latent must be a torch.Tensor")
    if latent.dtype != torch.bfloat16:
        raise AssertionError(f"{path}: latent dtype {latent.dtype} != torch.bfloat16")
    latent_height = int(data["latent_height"])
    latent_width = int(data["latent_width"])
    if latent.ndim != 2:
        raise AssertionError(f"{path}: latent shape {tuple(latent.shape)} must be 2D")
    expected_tokens = latent_num_frames * latent_height * latent_width
    if latent.shape[0] != expected_tokens:
        raise AssertionError(
            f"{path}: latent tokens {latent.shape[0]} != {expected_tokens}"
        )

    text_emb = data["text_emb"]
    if not torch.is_tensor(text_emb) or text_emb.ndim != 2:
        raise AssertionError(f"{path}: text_emb must be a 2D torch.Tensor")
    if text_emb.dtype != torch.bfloat16:
        raise AssertionError(f"{path}: text_emb dtype {text_emb.dtype} != torch.bfloat16")
    if (int(data["video_height"]), int(data["video_width"])) != expected_hw:
        raise AssertionError(
            f"{path}: video size {(data['video_height'], data['video_width'])} != {expected_hw}"
        )


def check_dataset(root: Path, check_latents: bool) -> None:
    info = json.load((root / "meta" / "info.json").open("r"))
    episodes = load_jsonl(root / "meta" / "episodes.jsonl")
    stats = load_jsonl(root / "meta" / "episodes_stats.jsonl")
    tasks = load_jsonl(root / "meta" / "tasks.jsonl")

    assert info["codebase_version"] == "v2.1"
    assert info["features"]["action"]["shape"] == [58]
    assert info["features"]["observation.state"]["shape"] == [58]
    assert len(tasks) == info["total_tasks"]
    assert len(episodes) == info["total_episodes"]
    assert len(stats) == info["total_episodes"]

    video_keys = [key for key, value in info["features"].items() if value["dtype"] == "video"]
    total_rows = 0
    expected_hw = tuple(info["features"][video_keys[0]]["shape"][1:])

    for episode in episodes:
        ep_idx = episode["episode_index"]
        chunk = ep_idx // info["chunks_size"]
        length = episode["length"]
        parquet_path = root / info["data_path"].format(
            episode_chunk=chunk,
            episode_index=ep_idx,
        )
        if not parquet_path.is_file():
            raise AssertionError(f"Missing parquet: {parquet_path}")
        table = pq.read_table(parquet_path)
        if table.num_rows != length:
            raise AssertionError(f"{parquet_path}: rows {table.num_rows} != episode length {length}")
        for col in ["action", "observation.state"]:
            values = np.asarray(table[col].to_pylist(), dtype=np.float32)
            if values.shape != (length, 58):
                raise AssertionError(f"{parquet_path}: {col} shape {values.shape} != {(length, 58)}")
            if not np.isfinite(values).all():
                raise AssertionError(f"{parquet_path}: {col} contains non-finite values")

        for video_key in video_keys:
            video_path = root / info["video_path"].format(
                episode_chunk=chunk,
                episode_index=ep_idx,
                video_key=video_key,
            )
            if not video_path.is_file():
                raise AssertionError(f"Missing video: {video_path}")
            check_video(video_path, length, info["fps"], expected_hw)

            if check_latents:
                for action_cfg in episode["action_config"]:
                    latent_path = (
                        root / "latents" / f"chunk-{chunk:03d}" / video_key /
                        f"episode_{ep_idx:06d}_{action_cfg['start_frame']}_{action_cfg['end_frame']}.pth"
                    )
                    if not latent_path.is_file():
                        raise AssertionError(f"Missing latent: {latent_path}")
                    check_latent(latent_path, action_cfg, expected_hw)

        total_rows += length

    if total_rows != info["total_frames"]:
        raise AssertionError(f"total rows {total_rows} != info.total_frames {info['total_frames']}")
    norm_path = root / "meta" / "dex_action_norm.json"
    if not norm_path.is_file():
        raise AssertionError(f"Missing action norm stats: {norm_path}")
    norm = json.load(norm_path.open("r"))
    if len(norm["q01"]) != 58 or len(norm["q99"]) != 58:
        raise AssertionError("dex_action_norm.json q01/q99 must have 58 values")

    print(f"OK: {root}")
    print(f"  episodes={len(episodes)} frames={total_rows} videos={len(video_keys)} fps={info['fps']}")
    print(f"  action_dim=58 latents_checked={check_latents}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Converted LeRobot dataset root.")
    parser.add_argument("--check-latents", action="store_true", help="Also require latent .pth files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    check_dataset(Path(args.dataset).expanduser().resolve(), args.check_latents)


if __name__ == "__main__":
    main()
