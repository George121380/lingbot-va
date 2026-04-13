# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import os

import torch
import torch.distributed as dist


def _use_distributed_workers():
    return dist.is_initialized() and dist.get_world_size() > 1


def _configure_model(model, shard_fn, param_dtype, device, eval_mode=True):
    """
    TODO
    """
    if eval_mode:
        model.eval().requires_grad_(False)
    if _use_distributed_workers():
        dist.barrier()

    if _use_distributed_workers():
        model = shard_fn(model)
    else:
        model.to(param_dtype)
        model.to(device)

    return model


def init_distributed(world_size, local_rank, rank):
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if world_size <= 1:
        return
    init_kwargs = dict(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )

    # ``device_id=...`` triggers eager NCCL connection. That path crashes on some
    # driver/runtime stacks (reproduced here with torch 2.11.0 + cu128 on B300),
    # so keep the stable init path as the default and make eager init opt-in.
    use_device_id_init = os.getenv("WAN_VA_USE_DEVICE_ID_INIT", "").lower() in (
        "1",
        "true",
        "yes",
    )
    if use_device_id_init:
        try:
            dist.init_process_group(device_id=torch.device(f"cuda:{local_rank}"),
                                    **init_kwargs)
            return
        except TypeError:
            pass

    dist.init_process_group(**init_kwargs)

def dist_mean(local_tensor):
    if dist.is_initialized():
        dist.all_reduce(local_tensor, op=dist.ReduceOp.AVG)
    return local_tensor

def dist_max(local_tensor):
    if dist.is_initialized():
        dist.all_reduce(local_tensor, op=dist.ReduceOp.MAX)
    return local_tensor
