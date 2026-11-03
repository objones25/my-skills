#!/usr/bin/env python3
"""Multi-process data-parallel training: DDP by default, FSDP2 with --fsdp.

Launch one process per device with torchrun:

    torchrun --standalone --nproc_per_node=2 assets/ddp_template.py --epochs 3
    torchrun --standalone --nproc_per_node=8 assets/ddp_template.py --fsdp

It runs on CPU with the gloo backend when no accelerator is present, so the whole
program (process group, DistributedSampler, DDP all-reduce, no_sync accumulation,
rank-0 checkpoint, teardown) can be tested on a laptop: tests/test_ddp_template.py runs
it with two CPU ranks and checks that the ranks end with identical weights and saw
disjoint data.

Replace build_data and build_model with your problem. See references/12-distributed.md.
Verified on torch 2.15.0 and 2.14.1 (CPU, gloo, 2 ranks).
"""

from __future__ import annotations

import argparse
import contextlib
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler


def build_data(n=2048, n_features=32, n_classes=10, seed=0):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(n_features, n_classes, generator=g)
    x = torch.randn(n, n_features, generator=g)
    y = (x @ w).argmax(1)
    idx = torch.arange(n)                      # carried so the test can check disjoint shards
    return TensorDataset(x, y, idx), n_features, n_classes


def build_model(n_features: int, n_classes: int) -> nn.Module:
    return nn.Sequential(nn.Linear(n_features, 128), nn.ReLU(), nn.Linear(128, 128),
                         nn.ReLU(), nn.Linear(128, n_classes))


def setup() -> tuple[int, int, torch.device]:
    """Init the process group from torchrun's environment and pick this rank's device."""
    acc = torch.accelerator.current_accelerator(check_available=True)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if acc is not None and acc.type == "cuda":
        torch.cuda.set_device(local_rank)      # BEFORE building the model
        device, backend = torch.device("cuda", local_rank), "nccl"  # lint: ignore[CUDA_HARDCODE] branch only runs when the accelerator is CUDA
    else:
        device, backend = torch.device("cpu"), "gloo"
    dist.init_process_group(backend)
    return dist.get_rank(), dist.get_world_size(), device


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=64, help="per rank")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--accum-steps", type=int, default=2)
    p.add_argument("--fsdp", action="store_true", help="shard with FSDP2 fully_shard instead of DDP")
    p.add_argument("--out", type=Path, default=Path("ddp_out"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rank, world, device = setup()
    try:
        torch.manual_seed(args.seed)               # same init on every rank
        ds, n_features, n_classes = build_data(seed=args.seed)
        sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True,
                                     seed=args.seed, drop_last=True)
        loader = DataLoader(ds, batch_size=args.batch_size, sampler=sampler)   # NOT shuffle=True

        model = build_model(n_features, n_classes).to(device)
        if args.fsdp:
            from torch.distributed.device_mesh import init_device_mesh
            from torch.distributed.fsdp import fully_shard

            mesh = init_device_mesh(device.type, (world,), mesh_dim_names=("dp",))
            for layer in model:
                if isinstance(layer, nn.Linear):
                    fully_shard(layer, mesh=mesh)  # inner modules first
            fully_shard(model, mesh=mesh)          # then the root
        else:
            model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)

        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        loss_fn = nn.CrossEntropyLoss()
        seen: list[int] = []

        for epoch in range(args.epochs):
            sampler.set_epoch(epoch)               # otherwise every epoch repeats one order
            model.train()
            total = torch.zeros((), device=device)
            count = 0
            for i, (x, y, idx) in enumerate(loader):
                x, y = x.to(device), y.to(device)
                if epoch == 0:
                    seen.extend(idx.tolist())
                last_micro = (i + 1) % args.accum_steps == 0 or (i + 1) == len(loader)
                # Skip the gradient all-reduce on accumulation micro-steps.
                if args.fsdp:
                    model.set_requires_gradient_sync(last_micro)
                    sync = contextlib.nullcontext()
                else:
                    sync = contextlib.nullcontext() if last_micro else model.no_sync()
                with sync:
                    loss = loss_fn(model(x), y) / args.accum_steps
                    loss.backward()
                if last_micro:
                    opt.step()
                    opt.zero_grad()
                total += loss.detach() * args.accum_steps
                count += 1
            mean = total / count
            dist.all_reduce(mean, op=dist.ReduceOp.SUM)    # every rank reaches this collective
            if rank == 0:
                print(f"epoch {epoch} loss {mean.item() / world:.4f}", flush=True)

        # Checksum of the full weights on every rank: DDP keeps replicas identical.
        if args.fsdp:
            from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
            state = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True))
        else:
            state = model.module.state_dict()          # .module drops the "module." prefix
        checksum = sum(float(v.double().sum()) for v in state.values())
        print(f"rank {rank} checksum {checksum:.6f} first_epoch_samples {len(seen)} "
              f"min {min(seen)} max {max(seen)} sum {sum(seen)}", flush=True)

        if rank == 0:
            args.out.mkdir(parents=True, exist_ok=True)
            torch.save(state, args.out / "model.pth")
        dist.barrier()                                 # nobody exits before the save
    finally:
        dist.destroy_process_group()                   # also on the error path


if __name__ == "__main__":
    main()
