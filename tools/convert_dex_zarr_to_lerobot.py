#!/usr/bin/env python3
"""Convert Teleop dexterous-hand zarr data to LingBot-VA LeRobot layout.

Run this script in the data environment:

    conda run -n dp python tools/convert_dex_zarr_to_lerobot.py ...
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr
from scipy.spatial.transform import Rotation


CAMERA_MAP = {
    "image_data": "observation.images.top",
    "image_data_wrist_left": "observation.images.left_wrist",
    "image_data_wrist_right": "observation.images.right_wrist",
}

DEFAULT_EMPTY_EMB = Path(
    "/share-2/code/fanqilin/peiqi/download/example-dataset/empty_emb.pt"
)


def pose_to_mat(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose)
    leading_shape = pose.shape[:-1]
    pos = pose[..., :3]
    rot = Rotation.from_rotvec(pose[..., 3:].reshape(-1, 3))
    mat = np.zeros(leading_shape + (4, 4), dtype=np.float64)
    mat[..., :3, :3] = rot.as_matrix().reshape(leading_shape + (3, 3))
    mat[..., :3, 3] = pos
    mat[..., 3, 3] = 1.0
    return mat


def normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    return vec / np.maximum(norm, eps)


def rot6d_from_mat(mat: np.ndarray) -> np.ndarray:
    # Match Teleop-franka-test/utils/pose_utils.py: mat[..., :2, :].reshape(..., 6)
    batch_dim = mat.shape[:-2]
    return mat[..., :2, :].copy().reshape(batch_dim + (6,))


def pose9d_from_mat(mat: np.ndarray) -> np.ndarray:
    return np.concatenate([mat[..., :3, 3], rot6d_from_mat(mat[..., :3, :3])], axis=-1)


def arm_state_to_9d(arm_state: np.ndarray, robot_num: int = 2) -> np.ndarray:
    arm_state = np.asarray(arm_state, dtype=np.float64).reshape(-1, robot_num, 6)
    return pose9d_from_mat(pose_to_mat(arm_state)).reshape(arm_state.shape[0], robot_num * 9)


def relative_actions(hand: np.ndarray, arm: np.ndarray, robot_num: int = 2) -> np.ndarray:
    next_hand = np.concatenate([hand[1:], hand[-1:]], axis=0)
    next_arm = np.concatenate([arm[1:], arm[-1:]], axis=0)

    hand_delta = next_hand - hand
    rel_arms = []
    for arm_id in range(robot_num):
        cur = arm[:, arm_id * 6:(arm_id + 1) * 6]
        nxt = next_arm[:, arm_id * 6:(arm_id + 1) * 6]
        rel_mat = np.linalg.inv(pose_to_mat(cur)) @ pose_to_mat(nxt)
        rel_arms.append(pose9d_from_mat(rel_mat))
    return np.concatenate(rel_arms + [hand_delta], axis=-1).astype(np.float32)


def absolute_actions(hand: np.ndarray, arm: np.ndarray, robot_num: int = 2) -> np.ndarray:
    next_hand = np.concatenate([hand[1:], hand[-1:]], axis=0)
    next_arm = np.concatenate([arm[1:], arm[-1:]], axis=0)
    arms_9d = arm_state_to_9d(next_arm, robot_num=robot_num)
    return np.concatenate([arms_9d, next_hand], axis=-1).astype(np.float32)


def fixed_size_list_array(values: np.ndarray) -> pa.FixedSizeListArray:
    values = np.asarray(values, dtype=np.float32)
    flat = pa.array(values.reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, values.shape[1])


def scalar_stats(values: np.ndarray) -> dict:
    values = np.asarray(values)
    return {
        "min": [float(np.min(values))],
        "max": [float(np.max(values))],
        "mean": [float(np.mean(values))],
        "std": [float(np.std(values))],
        "count": [int(values.shape[0])],
    }


def vector_stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": values.min(axis=0).astype(float).tolist(),
        "max": values.max(axis=0).astype(float).tolist(),
        "mean": values.mean(axis=0).astype(float).tolist(),
        "std": values.std(axis=0).astype(float).tolist(),
        "count": [int(values.shape[0])],
    }


def safe_quantiles(values: np.ndarray) -> tuple[list[float], list[float]]:
    q01 = np.quantile(values, 0.01, axis=0)
    q99 = np.quantile(values, 0.99, axis=0)
    narrow = (q99 - q01) < 1e-6
    center = (q01 + q99) / 2.0
    q01 = np.where(narrow, center - 1.0, q01)
    q99 = np.where(narrow, center + 1.0, q99)
    return q01.astype(float).tolist(), q99.astype(float).tolist()


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def append_jsonl(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(data) + "\n")


def write_parquet(
    path: Path,
    *,
    episode_index: int,
    global_start: int,
    task_index: int,
    fps: int,
    action: np.ndarray,
    state: np.ndarray,
) -> None:
    length = action.shape[0]
    frame_index = np.arange(length, dtype=np.int64)
    table = pa.table({
        "episode_index": pa.array(np.full(length, episode_index, dtype=np.int64)),
        "index": pa.array(np.arange(global_start, global_start + length, dtype=np.int64)),
        "frame_index": pa.array(frame_index),
        "task_index": pa.array(np.full(length, task_index, dtype=np.int64)),
        "timestamp": pa.array(frame_index.astype(np.float32) / float(fps), type=pa.float32()),
        "action": fixed_size_list_array(action),
        "observation.state": fixed_size_list_array(state),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def open_video_writer(path: Path, fps: int, width: int, height: int) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {path}")
    return writer


def write_video(
    path: Path,
    image_array,
    start: int,
    end: int,
    *,
    fps: int,
    height: int,
    width: int,
    source_bgr: bool,
) -> None:
    writer = open_video_writer(path, fps=fps, width=width, height=height)
    try:
        for idx in range(start, end):
            frame = np.asarray(image_array[idx])
            if frame.shape[0] != height or frame.shape[1] != width:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
            if not source_bgr:
                frame = frame[:, :, ::-1]
            writer.write(np.ascontiguousarray(frame))
    finally:
        writer.release()


def resolve_split_dirs(input_root: Path, splits: list[str]) -> list[tuple[str, Path]]:
    if (input_root / ".zgroup").exists():
        if len(splits) != 1:
            raise ValueError("When --input points directly at a zarr group, use one split only.")
        return [(splits[0], input_root)]

    out = []
    for split in splits:
        split_dir = input_root / split
        if not (split_dir / ".zgroup").exists():
            raise FileNotFoundError(f"Missing zarr split directory: {split_dir}")
        out.append((split, split_dir))
    return out


def make_info(
    *,
    total_episodes: int,
    total_frames: int,
    total_videos: int,
    total_chunks: int,
    fps: int,
    height: int,
    width: int,
    action_dim: int,
) -> dict:
    video_info = {
        "video.height": height,
        "video.width": width,
        "video.codec": "mp4v",
        "video.pix_fmt": "yuv420p",
        "video.is_depth_map": False,
        "video.fps": fps,
        "video.channels": 3,
        "has_audio": False,
    }
    features = {}
    for video_key in CAMERA_MAP.values():
        features[video_key] = {
            "dtype": "video",
            "shape": [3, height, width],
            "names": ["rgb", "height", "width"],
            "info": video_info,
        }
    action_names = {"motors": [f"action_{i:02d}" for i in range(action_dim)]}
    state_names = {"motors": [f"state_{i:02d}" for i in range(action_dim)]}
    features.update({
        "action": {"dtype": "float32", "shape": [action_dim], "names": action_names},
        "observation.state": {"dtype": "float32", "shape": [action_dim], "names": state_names},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    })
    return {
        "codebase_version": "v2.1",
        "robot_type": "dual_arm_dexterous_hand",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": total_videos,
        "total_chunks": total_chunks,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }


def convert(args: argparse.Namespace) -> None:
    input_root = Path(args.input).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {output_root}. Pass --overwrite to replace it.")
        for child in output_root.iterdir():
            if args.preserve_latents and child.name == "latents":
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    output_root.mkdir(parents=True, exist_ok=True)

    height, width = args.resize
    splits = [part.strip() for part in args.splits.split(",") if part.strip()]
    split_dirs = resolve_split_dirs(input_root, splits)
    action_dim = 58
    task_index = 0
    global_index = 0
    episode_index = 0
    all_actions = []
    total_frames = 0

    append_jsonl(output_root / "meta" / "tasks.jsonl", {
        "task_index": task_index,
        "task": args.task_text,
    })

    for split_name, split_dir in split_dirs:
        root = zarr.open(str(split_dir), mode="r")
        missing = [key for key in ["arm_data", "hand_data", "episode_ends"] if key not in root]
        missing += [key for key in CAMERA_MAP if key not in root]
        if missing:
            raise KeyError(f"{split_dir} is missing required zarr arrays: {missing}")

        episode_ends = np.asarray(root["episode_ends"][:], dtype=np.int64)
        episode_starts = np.concatenate([[0], episode_ends[:-1]])
        print(f"[{split_name}] {len(episode_ends)} episodes, {int(episode_ends[-1]) if len(episode_ends) else 0} frames")

        for local_ep, (start, end) in enumerate(zip(episode_starts, episode_ends)):
            start = int(start)
            end = int(end)
            length = end - start
            if length <= 0:
                raise ValueError(f"Invalid episode length in {split_dir}: start={start}, end={end}")

            arm = np.asarray(root["arm_data"][start:end], dtype=np.float64)
            hand = np.asarray(root["hand_data"][start:end], dtype=np.float64)
            if arm.shape[1] != 12 or hand.shape[1] != 40:
                raise ValueError(
                    f"Expected arm_data dim 12 and hand_data dim 40, got {arm.shape} and {hand.shape}"
                )

            state = np.concatenate([arm_state_to_9d(arm), hand], axis=-1).astype(np.float32)
            if args.action_type == "relative":
                action = relative_actions(hand, arm)
            else:
                action = absolute_actions(hand, arm)
            if action.shape[1] != action_dim or state.shape[1] != action_dim:
                raise AssertionError(f"Unexpected action/state shapes: {action.shape}, {state.shape}")

            chunk = episode_index // 1000
            parquet_path = output_root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"
            write_parquet(
                parquet_path,
                episode_index=episode_index,
                global_start=global_index,
                task_index=task_index,
                fps=args.fps,
                action=action,
                state=state,
            )

            for zarr_key, video_key in CAMERA_MAP.items():
                video_path = (
                    output_root / "videos" / f"chunk-{chunk:03d}" /
                    video_key / f"episode_{episode_index:06d}.mp4"
                )
                write_video(
                    video_path,
                    root[zarr_key],
                    start,
                    end,
                    fps=args.fps,
                    height=height,
                    width=width,
                    source_bgr=args.source_bgr,
                )

            episode = {
                "episode_index": episode_index,
                "tasks": [args.task_text],
                "length": length,
                "action_config": [{
                    "start_frame": 0,
                    "end_frame": length,
                    "action_text": args.task_text,
                    "skill": "",
                }],
            }
            append_jsonl(output_root / "meta" / "episodes.jsonl", episode)
            append_jsonl(output_root / "meta" / "episodes_ori.jsonl", {
                "episode_index": episode_index,
                "tasks": [args.task_text],
                "length": length,
            })

            frame_index = np.arange(length, dtype=np.int64)
            stats = {
                "episode_index": scalar_stats(np.full(length, episode_index)),
                "index": scalar_stats(np.arange(global_index, global_index + length)),
                "frame_index": scalar_stats(frame_index),
                "task_index": scalar_stats(np.full(length, task_index)),
                "timestamp": scalar_stats(frame_index.astype(np.float32) / float(args.fps)),
                "action": vector_stats(action),
                "observation.state": vector_stats(state),
            }
            append_jsonl(output_root / "meta" / "episodes_stats.jsonl", {
                "episode_index": episode_index,
                "stats": stats,
            })

            all_actions.append(action)
            global_index += length
            total_frames += length
            episode_index += 1
            print(f"  episode {episode_index - 1:06d}: source_ep={local_ep}, frames={length}")

    all_actions_np = np.concatenate(all_actions, axis=0)
    q01, q99 = safe_quantiles(all_actions_np)
    norm_stats = {
        "action_dim": action_dim,
        "action_type": args.action_type,
        "q01": q01,
        "q99": q99,
        "mean": all_actions_np.mean(axis=0).astype(float).tolist(),
        "std": all_actions_np.std(axis=0).astype(float).tolist(),
    }
    write_json(output_root / "meta" / "dex_action_norm.json", norm_stats)

    total_episodes = episode_index
    total_chunks = int(math.ceil(total_episodes / 1000)) if total_episodes else 0
    info = make_info(
        total_episodes=total_episodes,
        total_frames=total_frames,
        total_videos=total_episodes * len(CAMERA_MAP),
        total_chunks=total_chunks,
        fps=args.fps,
        height=height,
        width=width,
        action_dim=action_dim,
    )
    write_json(output_root / "meta" / "info.json", info)

    if args.empty_emb_source:
        empty_src = Path(args.empty_emb_source).expanduser()
        if empty_src.is_file():
            shutil.copy2(empty_src, output_root / "empty_emb.pt")
        else:
            print(f"warning: empty_emb source not found, skipped: {empty_src}")

    print(f"Converted {total_episodes} episodes / {total_frames} frames to {output_root}")
    print(f"Wrote action quantiles to {output_root / 'meta' / 'dex_action_norm.json'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Teleop zarr root or split directory.")
    parser.add_argument("--output", required=True, help="Output LeRobot dataset directory.")
    parser.add_argument("--splits", default="train", help="Comma-separated zarr splits under --input.")
    parser.add_argument("--task-text", default="pour egg", help="Language task text for tasks/action_config.")
    parser.add_argument("--fps", type=int, default=30, help="Output video and metadata fps.")
    parser.add_argument("--resize", type=int, nargs=2, default=[256, 256], metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--action-type", choices=["relative", "absolute"], default="relative")
    parser.add_argument("--source-bgr", action="store_true", help="Set if source zarr images are BGR.")
    parser.add_argument("--empty-emb-source", default=str(DEFAULT_EMPTY_EMB), help="Optional empty_emb.pt to copy.")
    parser.add_argument("--overwrite", action="store_true", help="Replace output directory if it exists.")
    parser.add_argument("--preserve-latents", action="store_true", help="When overwriting, keep an existing latents/ directory.")
    return parser.parse_args()


def main() -> None:
    convert(parse_args())


if __name__ == "__main__":
    main()
