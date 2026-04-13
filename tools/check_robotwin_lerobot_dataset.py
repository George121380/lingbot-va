#!/usr/bin/env python3
"""Validate a RobotWin LeRobot post-training dataset for LingBot-VA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch


REQUIRED_META_FILES = (
    "meta/info.json",
    "meta/episodes.jsonl",
    "meta/episodes_stats.jsonl",
    "meta/tasks.jsonl",
)
REQUIRED_VIDEO_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
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
EXPECTED_ACTION_DIM = 16
def load_jsonl(path: Path) -> list[dict]:
    with path.open("r") as f:
        return [json.loads(line) for line in f if line.strip()]


def discover_dataset_roots(dataset: Path) -> list[Path]:
    if (dataset / "meta" / "info.json").is_file():
        return [dataset]
    return sorted({info_path.parent.parent for info_path in dataset.rglob("meta/info.json")})


def validate_latent_file(path: Path, action_cfg: dict, errors: list[str]) -> None:
    if not path.is_file():
        errors.append(f"missing latent: {path}")
        return

    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        errors.append(f"failed to load latent {path}: {type(exc).__name__}: {exc}")
        return

    missing = sorted(REQUIRED_LATENT_KEYS - set(data.keys()))
    if missing:
        errors.append(f"{path}: missing latent keys {missing}")
        return

    start_frame = int(action_cfg["start_frame"])
    end_frame = int(action_cfg["end_frame"])
    if int(data["start_frame"]) != start_frame or int(data["end_frame"]) != end_frame:
        errors.append(
            f"{path}: stored frame range {data['start_frame']}:{data['end_frame']} "
            f"!= action_config {start_frame}:{end_frame}"
        )

    frame_ids = np.asarray(data["frame_ids"], dtype=np.int64)
    if frame_ids.ndim != 1 or len(frame_ids) == 0:
        errors.append(f"{path}: frame_ids must be a non-empty 1D array")
        return
    if int(frame_ids[0]) != start_frame:
        errors.append(f"{path}: first frame_id {frame_ids[0]} != {start_frame}")
    if int(frame_ids[-1]) >= end_frame:
        errors.append(f"{path}: last frame_id {frame_ids[-1]} >= {end_frame}")
    if len(frame_ids) % 4 != 1:
        errors.append(f"{path}: sampled frame count {len(frame_ids)} is not 4m+1")
    if len(frame_ids) > 1:
        diffs = np.diff(frame_ids)
        if not np.all(diffs == diffs[0]):
            errors.append(f"{path}: frame_ids are not sampled with a constant stride")

    if int(data["video_num_frames"]) != len(frame_ids):
        errors.append(
            f"{path}: video_num_frames {data['video_num_frames']} != len(frame_ids) {len(frame_ids)}"
        )

    latent_num_frames = int(data["latent_num_frames"])
    expected_latent_frames = (len(frame_ids) - 1) // 4 + 1
    if latent_num_frames != expected_latent_frames:
        errors.append(
            f"{path}: latent_num_frames {latent_num_frames} != {expected_latent_frames}"
        )

    latent = data["latent"]
    if not torch.is_tensor(latent) or latent.ndim != 2:
        errors.append(f"{path}: latent must be a 2D torch.Tensor")
    elif latent.dtype != torch.bfloat16:
        errors.append(f"{path}: latent dtype {latent.dtype} != torch.bfloat16")

    text_emb = data["text_emb"]
    if not torch.is_tensor(text_emb) or text_emb.ndim != 2:
        errors.append(f"{path}: text_emb must be a 2D torch.Tensor")
    elif text_emb.dtype != torch.bfloat16:
        errors.append(f"{path}: text_emb dtype {text_emb.dtype} != torch.bfloat16")

    latent_height = int(data["latent_height"])
    latent_width = int(data["latent_width"])
    expected_tokens = latent_num_frames * latent_height * latent_width
    if torch.is_tensor(latent) and latent.ndim == 2 and latent.shape[0] != expected_tokens:
        errors.append(f"{path}: latent tokens {latent.shape[0]} != {expected_tokens}")

    video_hw = (int(data["video_height"]), int(data["video_width"]))
    if video_hw[0] <= 0 or video_hw[1] <= 0:
        errors.append(f"{path}: latent video size must be positive, got {video_hw}")


def validate_task_root(task_root: Path, *, check_latents: bool,
                       check_videos: bool) -> tuple[dict, list[str]]:
    errors: list[str] = []

    for rel_path in REQUIRED_META_FILES:
        if not (task_root / rel_path).is_file():
            errors.append(f"missing required file: {task_root / rel_path}")
    if errors:
        return {}, errors

    try:
        info = json.load((task_root / "meta" / "info.json").open("r"))
        episodes = load_jsonl(task_root / "meta" / "episodes.jsonl")
        stats = load_jsonl(task_root / "meta" / "episodes_stats.jsonl")
        tasks = load_jsonl(task_root / "meta" / "tasks.jsonl")
    except Exception as exc:
        errors.append(f"failed to load metadata: {type(exc).__name__}: {exc}")
        return {}, errors

    if info.get("codebase_version") != "v2.1":
        errors.append(f"{task_root}: codebase_version={info.get('codebase_version')} != v2.1")
    if info.get("features", {}).get("action", {}).get("shape") != [EXPECTED_ACTION_DIM]:
        errors.append(
            f"{task_root}: action shape {info.get('features', {}).get('action', {}).get('shape')} "
            f"!= [{EXPECTED_ACTION_DIM}]"
        )

    features = info.get("features", {})
    for video_key in REQUIRED_VIDEO_KEYS:
        if video_key not in features:
            errors.append(f"{task_root}: missing video feature {video_key}")

    if len(tasks) != info.get("total_tasks"):
        errors.append(f"{task_root}: tasks count {len(tasks)} != info.total_tasks {info.get('total_tasks')}")
    if len(episodes) != info.get("total_episodes"):
        errors.append(
            f"{task_root}: episodes count {len(episodes)} != info.total_episodes {info.get('total_episodes')}"
        )
    if len(stats) != info.get("total_episodes"):
        errors.append(
            f"{task_root}: episode_stats count {len(stats)} != info.total_episodes {info.get('total_episodes')}"
        )

    total_rows = 0
    for episode in episodes:
        ep_idx = int(episode["episode_index"])
        chunk = ep_idx // int(info["chunks_size"])
        length = int(episode["length"])
        parquet_path = task_root / info["data_path"].format(
            episode_chunk=chunk,
            episode_index=ep_idx,
        )
        if not parquet_path.is_file():
            errors.append(f"missing parquet: {parquet_path}")
            continue

        try:
            table = pq.read_table(parquet_path, columns=["action"])
        except Exception as exc:
            errors.append(f"failed to read parquet {parquet_path}: {type(exc).__name__}: {exc}")
            continue

        if table.num_rows != length:
            errors.append(f"{parquet_path}: rows {table.num_rows} != episode length {length}")
            continue

        actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        if actions.shape != (length, EXPECTED_ACTION_DIM):
            errors.append(
                f"{parquet_path}: action shape {actions.shape} != {(length, EXPECTED_ACTION_DIM)}"
            )
        elif not np.isfinite(actions).all():
            errors.append(f"{parquet_path}: action contains non-finite values")

        action_configs = episode.get("action_config", [])
        if not action_configs:
            errors.append(f"{task_root}: episode {ep_idx} has empty action_config")
            continue

        for action_cfg in action_configs:
            start_frame = int(action_cfg["start_frame"])
            end_frame = int(action_cfg["end_frame"])
            if not (0 <= start_frame < end_frame <= length):
                errors.append(
                    f"{task_root}: episode {ep_idx} has invalid action range {start_frame}:{end_frame} for length {length}"
                )
                continue

            for video_key in REQUIRED_VIDEO_KEYS:
                if check_videos:
                    video_path = task_root / info["video_path"].format(
                        episode_chunk=chunk,
                        episode_index=ep_idx,
                        video_key=video_key,
                    )
                    if not video_path.is_file():
                        errors.append(f"missing video: {video_path}")

                if check_latents:
                    latent_path = (
                        task_root / "latents" / f"chunk-{chunk:03d}" / video_key /
                        f"episode_{ep_idx:06d}_{start_frame}_{end_frame}.pth"
                    )
                    validate_latent_file(latent_path, action_cfg, errors)

        total_rows += length

    if total_rows != info.get("total_frames"):
        errors.append(f"{task_root}: total rows {total_rows} != info.total_frames {info.get('total_frames')}")

    summary = {
        "task_root": str(task_root),
        "episodes": len(episodes),
        "frames": total_rows,
        "errors": len(errors),
    }
    return summary, errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="RobotWin dataset root or a single task root.")
    parser.add_argument("--report", default="", help="Optional JSON report path.")
    parser.add_argument("--skip-latents", action="store_true", help="Skip latent file validation.")
    parser.add_argument("--check-videos", action="store_true", help="Also require source videos to exist.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset).expanduser().resolve()

    if not dataset_root.exists():
        raise SystemExit(f"Dataset path does not exist: {dataset_root}")

    roots = discover_dataset_roots(dataset_root)
    if not roots:
        raise SystemExit(f"No LeRobot task roots found under {dataset_root}")

    report = {
        "dataset_root": str(dataset_root),
        "task_count": len(roots),
        "failures": [],
        "successes": [],
    }

    empty_emb_path = dataset_root / "empty_emb.pt"
    if empty_emb_path.is_file():
        try:
            empty_emb = torch.load(empty_emb_path, map_location="cpu", weights_only=False)
            if not torch.is_tensor(empty_emb) or empty_emb.ndim != 2:
                report["failures"].append({
                    "task_root": str(dataset_root),
                    "errors": [f"empty_emb.pt must be a 2D tensor: {empty_emb_path}"],
                })
        except Exception as exc:
            report["failures"].append({
                "task_root": str(dataset_root),
                "errors": [f"failed to load empty_emb.pt: {type(exc).__name__}: {exc}"],
            })
    elif len(roots) > 1:
        report["failures"].append({
            "task_root": str(dataset_root),
            "errors": [f"missing empty_emb.pt at dataset root: {empty_emb_path}"],
        })

    for task_root in roots:
        summary, errors = validate_task_root(
            task_root,
            check_latents=not args.skip_latents,
            check_videos=args.check_videos,
        )
        if errors:
            report["failures"].append({
                "task_root": str(task_root),
                "summary": summary,
                "errors": errors,
            })
            print(f"FAIL: {task_root}")
            for error in errors[:10]:
                print(f"  - {error}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more")
        else:
            report["successes"].append(summary)
            print(f"OK: {task_root} episodes={summary['episodes']} frames={summary['frames']}")

    if args.report:
        report_path = Path(args.report).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w") as f:
            json.dump(report, f, indent=2)
        print(f"Report written to {report_path}")

    if report["failures"]:
        raise SystemExit(f"Validation failed for {len(report['failures'])} dataset roots.")

    print(f"Validation passed for {len(report['successes'])} dataset roots.")


if __name__ == "__main__":
    main()
