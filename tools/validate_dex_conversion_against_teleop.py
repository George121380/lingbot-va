#!/usr/bin/env python3
"""Validate converted dex data against Teleop-franka-test action/state code."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import zarr


def load_teleop_fns(teleop_root: Path):
    sys.path.insert(0, str(teleop_root))
    from utils.pose_utils import action_6d_to_9d, get_delta_action, translate_arm_state

    return translate_arm_state, get_delta_action, action_6d_to_9d


def expected_state(arm: np.ndarray, hand: np.ndarray, translate_arm_state) -> np.ndarray:
    arm_9d = np.stack([translate_arm_state(arm_t) for arm_t in arm], axis=0)
    return np.concatenate([arm_9d, hand], axis=-1).astype(np.float32)


def expected_action(
    arm: np.ndarray,
    hand: np.ndarray,
    *,
    action_type: str,
    get_delta_action,
    action_6d_to_9d,
) -> np.ndarray:
    next_hand = np.concatenate([hand[1:], hand[-1:]], axis=0)
    next_arm = np.concatenate([arm[1:], arm[-1:]], axis=0)

    if action_type == "absolute":
        return np.stack([
            action_6d_to_9d(next_hand[i:i + 1], next_arm[i:i + 1], 2)[0]
            for i in range(len(hand))
        ], axis=0).astype(np.float32)

    return np.stack([
        get_delta_action(hand[i], arm[i], next_hand[i:i + 1], next_arm[i:i + 1], 2)[0]
        for i in range(len(hand))
    ], axis=0).astype(np.float32)


def read_parquet_arrays(dataset_root: Path, episode_index: int) -> tuple[np.ndarray, np.ndarray]:
    chunk = episode_index // 1000
    path = dataset_root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    table = pq.read_table(path)
    action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    return action, state


def validate(args: argparse.Namespace) -> None:
    source_split = Path(args.source_zarr_split).expanduser().resolve()
    dataset_root = Path(args.converted_dataset).expanduser().resolve()
    teleop_root = Path(args.teleop_root).expanduser().resolve()
    translate_arm_state, get_delta_action, action_6d_to_9d = load_teleop_fns(teleop_root)

    root = zarr.open(str(source_split), mode="r")
    hand_all = np.asarray(root["hand_data"][:], dtype=np.float64)
    arm_all = np.asarray(root["arm_data"][:], dtype=np.float64)
    episode_ends = np.asarray(root["episode_ends"][:], dtype=np.int64)
    episode_starts = np.concatenate([[0], episode_ends[:-1]])

    worst_action = 0.0
    worst_state = 0.0
    for local_episode, (start, end) in enumerate(zip(episode_starts, episode_ends)):
        episode_index = args.episode_offset + local_episode
        arm = arm_all[int(start):int(end)]
        hand = hand_all[int(start):int(end)]
        expected_a = expected_action(
            arm,
            hand,
            action_type=args.action_type,
            get_delta_action=get_delta_action,
            action_6d_to_9d=action_6d_to_9d,
        )
        expected_s = expected_state(arm, hand, translate_arm_state)
        actual_a, actual_s = read_parquet_arrays(dataset_root, episode_index)
        if actual_a.shape != expected_a.shape or actual_s.shape != expected_s.shape:
            raise AssertionError(
                f"episode {episode_index}: shape mismatch "
                f"action {actual_a.shape} vs {expected_a.shape}, "
                f"state {actual_s.shape} vs {expected_s.shape}"
            )
        action_diff = float(np.max(np.abs(actual_a - expected_a)))
        state_diff = float(np.max(np.abs(actual_s - expected_s)))
        worst_action = max(worst_action, action_diff)
        worst_state = max(worst_state, state_diff)
        if action_diff > args.atol or state_diff > args.atol:
            raise AssertionError(
                f"episode {episode_index}: diffs exceed tolerance {args.atol}: "
                f"action={action_diff}, state={state_diff}"
            )

    print(f"OK: Teleop parity for {len(episode_ends)} episodes")
    print(f"  max_abs_action_diff={worst_action}")
    print(f"  max_abs_state_diff={worst_state}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-zarr-split", required=True, help="Source zarr split dir, e.g. .../train.")
    parser.add_argument("--converted-dataset", required=True, help="Converted LeRobot dataset root.")
    parser.add_argument(
        "--teleop-root",
        default="/share-2/code/fanqilin/peiqi/Teleop-franka-test",
        help="Teleop-franka-test repo root.",
    )
    parser.add_argument("--action-type", choices=["relative", "absolute"], default="relative")
    parser.add_argument("--episode-offset", type=int, default=0)
    parser.add_argument("--atol", type=float, default=1e-6)
    return parser.parse_args()


def main() -> None:
    validate(parse_args())


if __name__ == "__main__":
    main()
