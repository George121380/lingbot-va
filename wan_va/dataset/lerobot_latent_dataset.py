# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import get_episode_data_index
from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats
import numpy as np
from pathlib import Path
from collections.abc import Callable
import os
from tqdm import tqdm
from multiprocessing import Pool, get_context
from functools import partial
import torch
import torch.nn.functional as F_pad_module
from einops import rearrange
from torch.utils.data import DataLoader
from scipy.spatial.transform import Rotation as R
from lerobot.constants import HF_LEROBOT_HOME
import json


def collate_variable_length(batch):
    """Custom collate function that pads variable-length frame tensors to the
    per-batch maximum and creates validity masks.

    Each sample dict is expected to contain:
        latents      (C, F, H, W)       – F varies across samples
        actions      (C, F_a, N, 1)     – F_a varies across samples
        actions_mask (C, F_a, N, 1)     – same shape as actions
        text_emb     (S, D)             – fixed shape
    """
    max_latent_f = max(s['latents'].shape[1] for s in batch)
    max_action_f = max(s['actions'].shape[1] for s in batch)

    padded_latents = []
    padded_actions = []
    padded_actions_masks = []
    latent_masks = []
    text_embs = []
    latent_num_frames_list = []
    action_num_frames_list = []

    for s in batch:
        f_l = s['latents'].shape[1]
        f_a = s['actions'].shape[1]

        # Pad latents along F dimension (dim=1 of C,F,H,W)
        pad_l = max_latent_f - f_l
        if pad_l > 0:
            # F.pad order: last dim first → (W_left, W_right, H_left, H_right, F_left, F_right)
            padded_latents.append(
                F_pad_module.pad(s['latents'], (0, 0, 0, 0, 0, pad_l)))
        else:
            padded_latents.append(s['latents'])

        # Create latent_mask: True for valid frames, False for padded
        mask = torch.zeros(1, max_latent_f, 1, 1, dtype=torch.bool)
        mask[:, :f_l] = True
        latent_masks.append(mask)
        latent_num_frames_list.append(f_l)

        # Pad actions and actions_mask along F dimension (dim=1 of C,F,N,1)
        pad_a = max_action_f - f_a
        if pad_a > 0:
            padded_actions.append(
                F_pad_module.pad(s['actions'], (0, 0, 0, 0, 0, pad_a)))
            padded_actions_masks.append(
                F_pad_module.pad(s['actions_mask'], (0, 0, 0, 0, 0, pad_a)))
        else:
            padded_actions.append(s['actions'])
            padded_actions_masks.append(s['actions_mask'])
        action_num_frames_list.append(f_a)

        text_embs.append(s['text_emb'])

    return {
        'latents': torch.stack(padded_latents),
        'actions': torch.stack(padded_actions),
        'actions_mask': torch.stack(padded_actions_masks),
        'text_emb': torch.stack(text_embs),
        'latent_mask': torch.stack(latent_masks),
        'latent_num_frames': torch.tensor(latent_num_frames_list, dtype=torch.long),
        'action_num_frames': torch.tensor(action_num_frames_list, dtype=torch.long),
    }


class DatasetShardError(RuntimeError):
    """Base error raised while initializing a single LeRobot shard."""


class IncompleteDatasetShardError(DatasetShardError):
    """The shard is missing files required for training."""

def recursive_find_file(directory, filename='info.json'):
    result = []
    try:
        for root, dirs, files in os.walk(directory):
            if filename in files:
                full_path = os.path.join(root, filename)
                result.append(full_path)
    except PermissionError:
        print(f"Error: can not access {directory}")
    except Exception as e:
        print(f"Error: {e}")
    return result

def construct_lerobot(
    repo_id,
    config,
):
    return LatentLeRobotDataset(
        repo_id=repo_id,
        config=config,
    )


def construct_lerobot_safe(repo_id, config):
    try:
        return {
            "repo_id": repo_id,
            "dataset": construct_lerobot(repo_id=repo_id, config=config),
            "error": None,
            "recoverable": False,
        }
    except Exception as exc:
        return {
            "repo_id": repo_id,
            "dataset": None,
            "error": f"{type(exc).__name__}: {exc}",
            "recoverable": isinstance(exc, IncompleteDatasetShardError),
        }


def _dataset_report_path(config):
    report_path = getattr(config, "dataset_init_report_path", "")
    if report_path:
        return Path(report_path).expanduser().resolve()
    save_root = getattr(config, "save_root", "./train_out")
    return (Path(save_root) / "dataset_init_report.json").resolve()


def _write_dataset_report(config, repo_list, valid_datasets, skipped_datasets):
    if getattr(config, "rank", 0) != 0:
        return None

    report_path = _dataset_report_path(config)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "dataset_path": str(config.dataset_path),
        "allow_incomplete_datasets": bool(
            getattr(config, "allow_incomplete_datasets", False)),
        "discovered_repo_count": len(repo_list),
        "initialized_repo_count": len(valid_datasets),
        "failed_repo_count": len(skipped_datasets),
        "failures": skipped_datasets,
    }
    with report_path.open("w") as f:
        json.dump(report, f, indent=2)
    return report_path


def construct_lerobot_multi_processor(config, 
                                      num_init_worker=8,
                                      ):
    datasets_out_lst = []
    construct_func = partial(
        construct_lerobot_safe,
        config=config,
    )
    repo_list = recursive_find_file(config.dataset_path, 'info.json')
    repo_list = sorted(v.split('/meta/info.json')[0] for v in repo_list)
    if not repo_list:
        raise RuntimeError(
            f"No LeRobot datasets were found under {config.dataset_path}")
    if num_init_worker <= 1:
        datasets_out_lst = [construct_func(repo_id) for repo_id in repo_list]
    else:
        # Dataset discovery runs after CUDA/FSDP initialization in training, so
        # using spawn avoids forking a process that already owns CUDA context.
        with get_context("spawn").Pool(num_init_worker) as pool:
            datasets_out_lst = pool.map(construct_func, repo_list)

    valid_datasets = []
    skipped_datasets = []
    for result in datasets_out_lst:
        if result["dataset"] is None:
            skipped_datasets.append({
                "repo_id": result["repo_id"],
                "error": result["error"],
                "recoverable": result["recoverable"],
            })
            continue
        valid_datasets.append(result["dataset"])

    report_path = _write_dataset_report(config, repo_list, valid_datasets,
                                        skipped_datasets)

    if skipped_datasets:
        all_recoverable = all(result["recoverable"] for result in skipped_datasets)
        allow_incomplete = bool(
            getattr(config, "allow_incomplete_datasets", False))
        summary_lines = [
            f"Detected {len(skipped_datasets)} dataset shard failures while scanning {config.dataset_path}.",
        ]
        if report_path is not None:
            summary_lines.append(f"Detailed report: {report_path}")
        for result in skipped_datasets[:10]:
            summary_lines.append(f"- {result['repo_id']}: {result['error']}")
        remaining = len(skipped_datasets) - 10
        if remaining > 0:
            summary_lines.append(f"... and {remaining} more")

        if allow_incomplete and all_recoverable:
            print(
                "Skipping incomplete LeRobot datasets because "
                "LINGBOT_VA_ALLOW_INCOMPLETE_DATASETS=1:")
            details_start = 2 if report_path is not None else 1
            for line in summary_lines[details_start:]:
                print(f"  {line}")
        else:
            if allow_incomplete and not all_recoverable:
                summary_lines.insert(
                    1,
                    "At least one failure is not a missing/incomplete-shard error, so training will stop.",
                )
            else:
                summary_lines.insert(
                    1,
                    "Training is in strict mode. Set LINGBOT_VA_ALLOW_INCOMPLETE_DATASETS=1 only for smoke tests on partial data.",
                )
            raise RuntimeError("\n".join(summary_lines))

    if not valid_datasets:
        message_lines = [
            f"Failed to initialize any LeRobot datasets under {config.dataset_path}.",
        ]
        if report_path is not None:
            message_lines.append(f"Detailed report: {report_path}")
        raise RuntimeError("\n".join(message_lines))

    return valid_datasets

def get_relative_pose(pose):
    if torch.is_tensor(pose):
        pose = pose.detach().cpu().numpy()
    
    rot = R.from_quat(pose[:, 3:7])
    first_rot = R.from_quat(np.tile(pose[:1, 3:7], (pose.shape[0], 1)))
    trans = pose[:, :3]
    relative_trans = trans - trans[0:1]

    relative_rot = first_rot.inv() * rot
    relative_quat = relative_rot.as_quat()

    relative_pose = np.concatenate([relative_trans, relative_quat], axis=1)
    return torch.from_numpy(relative_pose)

class MultiLatentLeRobotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config,
        num_init_worker=None,
    ):
        if num_init_worker is None:
            num_init_worker = getattr(config, 'num_init_worker', 8)
        num_init_worker = max(1, min(int(num_init_worker), os.cpu_count() or 1))
        self._datasets = construct_lerobot_multi_processor(config, 
                                                           num_init_worker, 
                                                           )
        self.item_id_to_dataset_id, self.acc_dset_num = (
            self._get_item_id_to_dataset_id()
        )

    def __len__(
        self,
    ):
        return sum(len(v) for v in self._datasets)

    def _get_item_id_to_dataset_id(self):
        item_id_to_dataset_id = {}
        acc_dset_num = {}
        acc_nums = [0]
        id = 0
        for dset_id, dset in enumerate(self._datasets):
            acc_nums.append(acc_nums[-1] + len(dset))
            for _ in range(len(dset)):
                item_id_to_dataset_id[id] = dset_id
                id += 1
        for did in range(len(self._datasets)):
            acc_dset_num[did] = acc_nums[did]
        return item_id_to_dataset_id, acc_dset_num

    def __getitem__(self, idx) -> dict:
        assert idx < len(self)
        cur_dset = self._datasets[self.item_id_to_dataset_id[idx]]
        local_idx = idx - self.acc_dset_num[self.item_id_to_dataset_id[idx]]
        return cur_dset[local_idx]

class LatentLeRobotDataset(LeRobotDataset):
    def __init__(
        self,
        repo_id,
        config=None,
    ):
        self.repo_id = repo_id
        self.root = HF_LEROBOT_HOME / repo_id
        self.image_transforms = None
        self.delta_timestamps = None
        self.episodes = None
        self.tolerance_s = 1e-4
        self.revision = "v2.1"
        self.video_backend = 'pyav'
        self.delta_indices = None
        self.batch_encoding_size = 1
        self.episodes_since_last_encoding = 0
        self.image_writer = None
        self.episode_buffer = None
        self.root.mkdir(exist_ok=True, parents=True)
        self.meta = LeRobotDatasetMetadata(
            self.repo_id, self.root, self.revision, force_cache_sync=False
        )
        if self.episodes is not None and self.meta._version >= packaging.version.parse("v2.1"):
            episodes_stats = [self.meta.episodes_stats[ep_idx] for ep_idx in self.episodes]
            self.stats = aggregate_stats(episodes_stats)
        
        try:
            assert all((self.root / fpath).is_file() for fpath in self.get_episodes_file_paths())
            self.hf_dataset = self.load_hf_dataset()
        except (AssertionError, FileNotFoundError, NotADirectoryError):
            raise IncompleteDatasetShardError(
                f"Dataset files are incomplete under {self.root}. "
                "Finish the download or allow this shard to be skipped."
            )
        self.episode_data_index = get_episode_data_index(self.meta.episodes, self.episodes)
        
        self.latent_path = Path(repo_id) / 'latents'
        self.empty_emb = torch.load(config.empty_emb_path, weights_only=False)
        self.config = config
        self.cfg_prob = config.cfg_prob
        self.used_video_keys = config.obs_cam_keys
        self.q01 = np.array(config.norm_stat['q01'], dtype='float')[None]
        self.q99 = np.array(config.norm_stat['q99'], dtype='float')[None]
        # Keep the HF dataset action-only and avoid datasets' torch formatter:
        # in some torchvision builds the formatter imports VideoReader even
        # though LingBot-VA trains from precomputed latents, not raw videos.
        self._hf_action_view = self.hf_dataset.select_columns(['action'])
        self.parse_meta()
        if not self.new_metas:
            raise IncompleteDatasetShardError(
                f"No usable latent/action segments found under {repo_id}")

    def __len__(self):
        return len(self.new_metas)

    def parse_meta(self):
        out = []
        missing_segments = []
        for key, value in self.meta.episodes.items():
            episode_index = value["episode_index"]
            tasks = value["tasks"]
            action_config = value["action_config"]
            for acfg in action_config:
                cur_meta = {
                    "episode_index": episode_index,
                    "tasks": tasks,
                }
                cur_meta.update(acfg)

                missing_files = self._check_meta(
                    cur_meta["start_frame"],
                    cur_meta["end_frame"],
                    cur_meta["episode_index"],
                )

                if missing_files:
                    missing_segments.append({
                        "episode_index": episode_index,
                        "start_frame": cur_meta["start_frame"],
                        "end_frame": cur_meta["end_frame"],
                        "missing_files": missing_files,
                    })
                    continue

                out.append(cur_meta)

        if missing_segments and not getattr(self.config,
                                            "allow_incomplete_datasets",
                                            False):
            summary_lines = [
                f"Found {len(missing_segments)} action segments with missing latent files under {self.repo_id}.",
            ]
            for segment in missing_segments[:10]:
                summary_lines.append(
                    f"episode {segment['episode_index']} "
                    f"{segment['start_frame']}:{segment['end_frame']} missing:"
                )
                for missing_file in segment["missing_files"][:3]:
                    summary_lines.append(f"  {missing_file}")
                remaining_files = len(segment["missing_files"]) - 3
                if remaining_files > 0:
                    summary_lines.append(f"  ... and {remaining_files} more files")
            remaining_segments = len(missing_segments) - 10
            if remaining_segments > 0:
                summary_lines.append(f"... and {remaining_segments} more segments")
            raise IncompleteDatasetShardError("\n".join(summary_lines))

        self.new_metas = out

    def _check_meta(self, start_frame, end_frame, episode_index):
        episode_chunk = self.meta.get_episode_chunk(episode_index)
        latent_path = Path(self.latent_path) / f"chunk-{episode_chunk:03d}"
        missing_files = []
        for key in self.used_video_keys:
            cur_path = latent_path / key
            latent_file = (
                cur_path / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
            )
            if not os.path.exists(latent_file):
                missing_files.append(str(latent_file))
        return missing_files

    def _get_global_idx(self, episode_index: int, local_index: int):
        ep_start = self.episode_data_index["from"][episode_index]
        return local_index + ep_start

    def _get_range_hf_data(self, start_frame, end_frame):
        batch = self._hf_action_view[start_frame:end_frame]
        return {
            'action':
            torch.from_numpy(np.asarray(batch['action'], dtype=np.float32))
        }

    def _flatten_latent_dict(self, latent_dict):
        out = {}
        for key, value in latent_dict.items():
            for inner_key, inner_value in value.items():
                new_key = f"{key}.{inner_key}"
                out[new_key] = inner_value
        return out

    def _get_range_latent_data(self, start_frame, end_frame, episode_index):
        episode_chunk = self.meta.get_episode_chunk(episode_index)
        latent_path = Path(self.latent_path) / f"chunk-{episode_chunk:03d}"
        out = {}
        for key in self.used_video_keys:
            cur_path = latent_path / key
            latent_file = (
                cur_path / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
            )
            assert os.path.exists(latent_file)
            latent_data = torch.load(latent_file, weights_only=False)
            out[key] = latent_data
        
        return self._flatten_latent_dict(out)
    
        
    def _cat_video_latents(self,
                           data_dict
                           ):
        latent_lst = []
        for key in self.used_video_keys:
            latent= data_dict[f"{key}.latent"]
            latent_num_frames = data_dict[f"{key}.latent_num_frames"]
            latent_height = data_dict[f"{key}.latent_height"]
            latent_width = data_dict[f"{key}.latent_width"]
            latent = rearrange(latent, 
                                 '(f h w) c -> f h w c', 
                                 f=latent_num_frames, 
                                 h=latent_height, 
                                 w=latent_width)
            latent_lst.append(latent)
        if self.config.env_type == 'robotwin_tshape':
            wrist_latent = torch.cat(latent_lst[1:], dim=2)
            cat_latent = torch.cat([wrist_latent, latent_lst[0]], dim=1)
        else:
            cat_latent = torch.cat(latent_lst, dim=2)

        text_emb = data_dict[f"{self.used_video_keys[0]}.text_emb"]
        if torch.rand(1).item() < self.cfg_prob:
            text_emb = self.empty_emb

        out_dict = dict(
            latents = cat_latent,
            text_emb = text_emb,
        )
        return out_dict
    
    def _action_post_process(self, local_start_frame, local_end_frame, latent_frame_ids, action):
        act_shift = int(latent_frame_ids[0] - local_start_frame)
        frame_stride = latent_frame_ids[1] - latent_frame_ids[0] if len(latent_frame_ids) > 1 else 1
        if torch.is_tensor(action):
            action = action.detach().cpu().numpy()
        else:
            action = np.asarray(action)
        action = action[act_shift:]
        if self.config.env_type == 'robotwin_tshape': ## TODO support get_relative_pose for other dataset, currently only support robotwin 
            left_action = get_relative_pose(action[:, :7])
            right_action = get_relative_pose(action[:, 8:15])
            action = np.concatenate([left_action, action[:, 7:8], right_action, action[:, 15:16]], axis=1)
        elif self.config.env_type == 'dex_zarr':
            if action.shape[-1] != self.config.action_dim:
                raise ValueError(
                    f"dex_zarr action dim mismatch: parquet action has {action.shape[-1]} dims, "
                    f"but config.action_dim={self.config.action_dim}"
                )
        action = np.pad(action, pad_width=((frame_stride * 4, 0), (0, 0)), mode='constant', constant_values=0)

        latent_frame_num = (len(latent_frame_ids) - 1) // 4 + 1
        required_action_num = latent_frame_num * frame_stride * 4

        action = action[:required_action_num]
        action_mask = np.ones_like(action, dtype='bool')
        assert action.shape[0] == required_action_num


        action_paded = np.pad(action, ((0, 0), (0, 1)), mode='constant', constant_values=0)
        action_mask_padded = np.pad(action_mask, ((0, 0), (0, 1)), mode='constant', constant_values=0)

        action_aligned = action_paded[:, self.config.inverse_used_action_channel_ids]
        action_mask_aligned = action_mask_padded[:, self.config.inverse_used_action_channel_ids]
        action_aligned = (action_aligned - self.q01) / (
                self.q99 - self.q01 + 1e-6) * 2. - 1.
        action_aligned = rearrange(action_aligned, "(f n) c -> c f n 1", f=latent_frame_num)
        action_mask_aligned = rearrange(action_mask_aligned, "(f n) c -> c f n 1", f=latent_frame_num)
        action_aligned *= action_mask_aligned
        return torch.from_numpy(action_aligned).float(), torch.from_numpy(action_mask_aligned).bool()

    def __getitem__(self, idx) -> dict:
        idx = idx % len(self.new_metas)
        cur_meta = self.new_metas[idx]
        episode_index = cur_meta["episode_index"]
        start_frame = cur_meta["start_frame"]
        end_frame = cur_meta["end_frame"]
        local_start_frame = start_frame
        local_end_frame = end_frame

        ori_data_dict = self._get_range_latent_data(start_frame, end_frame, episode_index)

        latent_frame_ids = ori_data_dict[f"{self.used_video_keys[0]}.frame_ids"]
        start_frame = self._get_global_idx(episode_index, start_frame)
        end_frame = self._get_global_idx(episode_index, end_frame)

        hf_data_frames = self._get_range_hf_data(start_frame, end_frame)
        ori_data_dict.update(hf_data_frames)
        out_dict = self._cat_video_latents(ori_data_dict)

        out_dict['actions'], out_dict['actions_mask'] = self._action_post_process(local_start_frame, local_end_frame, latent_frame_ids, ori_data_dict['action'])

        out_dict['latents'] = out_dict['latents'].permute(3, 0, 1, 2)
        return out_dict

    def __len__(self):
        return len(self.new_metas)

if __name__ == '__main__':
    from wan_va.configs import VA_CONFIGS
    from tqdm import tqdm
    dset = MultiLatentLeRobotDataset(
        VA_CONFIGS['demo_train']
    )
    for key, value in dset[0].items():
        if isinstance(value, torch.Tensor):
            print(f'{key}: {value.shape} tensor')
        elif isinstance(value, np.ndarray):
            print(f'{key}: {value.shape} np')
        else:
            print(f'{key}: {value}')
    print(len(dset))
    dloader = DataLoader(
            dset,
            batch_size=1,
            shuffle=True,
            num_workers=32,
        )
    max_l = 0
    action_list = []
    for data in tqdm(dloader):
        _, _, F, H, W = data['latents'].shape
        max_l = max(max_l, F*H*W)
        action_list.append(data['actions'].flatten(2).permute(0, 2, 1).flatten(0, 1))
    action_all = torch.cat(action_list, dim=0)
    print(max_l)
    print(action_all.shape, action_all.mean(dim=0), action_all.min(dim=0)[0], action_all.max(dim=0)[0])
    
