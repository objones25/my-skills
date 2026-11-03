#!/usr/bin/env python3
"""A complete, runnable PyTorch training script to adapt rather than retype.

Everything here follows the conventions in the skill's references: device-agnostic
placement, raw-logit head, AMP with the right dtype for the device, gradient
accumulation, cosine schedule, seeded DataLoader workers, best-checkpoint saving,
and resume.

Run it as-is to smoke-test an environment (it trains on synthetic data):

    python assets/train_template.py --epochs 3

Adapt it by replacing `build_data` and `build_model`. Nothing else needs to change.

Verified end to end on torch 2.15.0 and 2.14.1 (CPU). Resume is bit-for-bit: a run
killed after epoch N and resumed matches the uninterrupted run (tests/test_template.py).
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
import warnings
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)          # seeds CPU and all CUDA devices
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    """DataLoader workers reseed themselves - pin them so augmentation is reproducible."""
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    try:
        import numpy as np

        np.random.seed(worker_seed)
    except ImportError:
        pass


# --------------------------------------------------------------------------- #
# Replace these two functions with your problem
# --------------------------------------------------------------------------- #
def build_data(n_train=4096, n_val=1024, n_features=32, n_classes=10, seed=0):
    """Synthetic linearly-separable-ish classification data, so the template runs anywhere."""
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(n_features, n_classes, generator=g)

    def make(n):
        x = torch.randn(n, n_features, generator=g)
        y = (x @ w + 0.3 * torch.randn(n, n_classes, generator=g)).argmax(1)
        return TensorDataset(x, y)

    return make(n_train), make(n_val), n_features, n_classes


def build_model(n_features: int, n_classes: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(n_features, 256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(256, n_classes),   # raw logits - the loss owns the softmax
    )


# --------------------------------------------------------------------------- #
# Device and precision
# --------------------------------------------------------------------------- #
def pick_device() -> torch.device:
    return torch.accelerator.current_accelerator(check_available=True) or torch.device("cpu")


def pick_amp(device: torch.device, requested: str) -> tuple[torch.dtype | None, bool]:
    """Return (autocast dtype or None, whether a GradScaler is needed)."""
    if requested == "off" or not torch.amp.is_autocast_available(device.type):
        return None, False
    if requested == "fp16":
        # fp16 needs loss scaling. GradScaler has a CUDA path and, in recent releases, an
        # MPS code path; only the CUDA path is exercised by this template's tests.
        return torch.float16, device.type == "cuda"
    if device.type == "cuda":
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:                                  # Ampere and newer
            return torch.bfloat16, False
        return torch.float16, True                      # Volta / Turing
    if device.type == "mps":
        return torch.float16, False
    return torch.bfloat16, False                        # CPU


# --------------------------------------------------------------------------- #
# Train / eval
# --------------------------------------------------------------------------- #
def train_one_epoch(model, loader, loss_fn, optimizer, device, *, amp_dtype, scaler,
                    accum_steps, scheduler_per_batch, max_grad_norm, log_every, epoch):
    model.train()
    running = torch.zeros((), device=device)          # stays on device: no per-step sync
    seen = 0
    optimizer.zero_grad()

    for i, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            loss = loss_fn(model(x), y)

        # Divide so the accumulated gradient equals the mean over the effective batch.
        scaled = loss / accum_steps
        if scaler is not None:
            scaler.scale(scaled).backward()
        else:
            scaled.backward()

        is_step = (i + 1) % accum_steps == 0 or (i + 1) == len(loader)
        if is_step:
            if max_grad_norm:
                if scaler is not None:
                    scaler.unscale_(optimizer)          # unscale BEFORE clipping
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            if scheduler_per_batch is not None:
                scheduler_per_batch.step()

        running += loss.detach() * x.size(0)             # detach: keep no graph alive
        seen += x.size(0)
        if log_every and (i + 1) % log_every == 0:      # .item() syncs, so only here
            print(f"  epoch {epoch} batch {i+1}/{len(loader)}  loss {running.item()/seen:.4f}")

    return running.item() / seen


@torch.inference_mode()                                   # faster than no_grad
def evaluate(model, loader, loss_fn, device, *, amp_dtype):
    model.eval()
    loss_sum, correct, n = 0.0, 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(x)
            loss = loss_fn(logits, y)
        loss_sum += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += x.size(0)
    return loss_sum / n, correct / n


# --------------------------------------------------------------------------- #
# Checkpointing
# --------------------------------------------------------------------------- #
def unwrap(model: nn.Module) -> nn.Module:
    """torch.compile returns a wrapper whose state_dict keys gain an `_orig_mod.` prefix.
    Save and load the original module so checkpoints load with or without --compile."""
    return getattr(model, "_orig_mod", model)


def device_rng_state():
    """The accelerator's RNG (dropout on the device draws from it), or None on CPU."""
    if torch.cuda.is_available():
        return {"type": "cuda", "state": torch.cuda.get_rng_state_all()}
    if torch.backends.mps.is_available():
        return {"type": "mps", "state": torch.mps.get_rng_state()}
    return None


def set_device_rng_state(saved) -> None:
    """Restore only onto the same kind of device; other RNG streams cannot be converted."""
    current = device_rng_state()
    if saved is None or current is None:
        return
    same_device = isinstance(saved, dict) and saved.get("type") == current["type"]
    if same_device and saved["type"] == "cuda":
        same_device = len(saved["state"]) == len(current["state"])
    if not same_device:
        print("note: checkpoint RNG is from another device; resume is not bit-exact", file=sys.stderr)
        return
    if saved["type"] == "cuda":
        torch.cuda.set_rng_state_all(saved["state"])
    else:
        torch.mps.set_rng_state(saved["state"])


def save_ckpt(path, *, epoch, model, optimizer, scheduler, scaler, best_val,
              bad_epochs, total_epochs, generator):
    torch.save(
        {
            "epoch": epoch,
            "model": unwrap(model).state_dict(),          # state_dict, never the object
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "scaler": scaler.state_dict() if scaler else None,
            "best_val": best_val,
            "bad_epochs": bad_epochs,
            "total_epochs": total_epochs,
            # Without these, a resumed run reshuffles from the seed and replays epoch 0's order.
            "rng": {"torch": torch.get_rng_state(), "python": random.getstate(),
                    "loader": generator.get_state(), "device": device_rng_state()},
        },
        path,
    )


def load_ckpt(path, *, model, optimizer, scheduler, scaler, generator):
    # weights_only=False because the checkpoint carries Python's RNG state (a tuple of ints).
    # Only load checkpoints you wrote yourself.
    # Load to CPU: set_rng_state needs a CPU ByteTensor, and load_state_dict moves the rest.
    ckpt = torch.load(path, map_location="cpu", weights_only=False)  # lint: ignore[LOAD_WEIGHTS_ONLY_FALSE] own checkpoint carrying RNG state
    unwrap(model).load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and ckpt.get("scheduler"):
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler and ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])
    torch.set_rng_state(ckpt["rng"]["torch"])
    random.setstate(ckpt["rng"]["python"])
    generator.set_state(ckpt["rng"]["loader"])
    set_device_rng_state(ckpt["rng"].get("device"))
    return ckpt["epoch"] + 1, ckpt["best_val"], ckpt["bad_epochs"], ckpt["total_epochs"]


# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)   # AdamW's own default
    p.add_argument("--accum-steps", type=int, default=1)
    p.add_argument("--max-grad-norm", type=float, default=1.0, help="0 disables clipping")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--amp", choices=["auto", "off", "fp16"], default="auto")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--patience", type=int, default=5, help="early stop after N bad epochs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("checkpoints"))
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--stop-after", type=int, default=None,
                   help="exit after this many epochs in this run (simulates preemption)")
    args = p.parse_args()

    seed_all(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    device = pick_device()
    amp_dtype, need_scaler = pick_amp(device, args.amp)
    print(f"device={device}  autocast={amp_dtype}  scaler={need_scaler}  torch={torch.__version__}")

    train_ds, val_ds, n_features, n_classes = build_data(seed=args.seed)
    g = torch.Generator().manual_seed(args.seed)
    common = dict(num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
                  persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, worker_init_fn=seed_worker, generator=g,
                              **common)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **common)

    model = build_model(n_features, n_classes).to(device)   # modules move in place
    if args.compile:
        model = torch.compile(model)

    loss_fn = nn.CrossEntropyLoss()                          # raw logits + int64 targets
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(device.type) if need_scaler else None

    start_epoch, best_val, bad_epochs = 0, math.inf, 0
    if args.resume and args.resume.exists():
        start_epoch, best_val, bad_epochs, planned = load_ckpt(
            args.resume, model=model, optimizer=optimizer, scheduler=scheduler,
            scaler=scaler, generator=g)
        print(f"resumed from {args.resume} at epoch {start_epoch}")
        if planned != args.epochs:
            # A restored CosineAnnealingLR keeps the old T_max and climbs back to the peak
            # LR once it passes it. Rebuild the schedule for the new length instead.
            print(f"note: checkpoint planned {planned} epochs, now {args.epochs}; "
                  "rebuilding the cosine schedule for the new length")
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"]
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")           # stepping before optimizer.step()
                for _ in range(start_epoch):
                    scheduler.step()

    for epoch in range(start_epoch, args.epochs):
        t0 = time.perf_counter()
        tr_loss = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device,
            amp_dtype=amp_dtype, scaler=scaler, accum_steps=args.accum_steps,
            scheduler_per_batch=None,            # cosine steps per epoch, below
            max_grad_norm=args.max_grad_norm, log_every=0, epoch=epoch,
        )
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device, amp_dtype=amp_dtype)
        scheduler.step()                          # per-epoch scheduler

        improved = val_loss < best_val
        if improved:
            best_val, bad_epochs = val_loss, 0
            save_ckpt(args.out / "best.pth", epoch=epoch, model=model, optimizer=optimizer,
                      scheduler=scheduler, scaler=scaler, best_val=best_val,
                      bad_epochs=bad_epochs, total_epochs=args.epochs, generator=g)
        else:
            bad_epochs += 1
        save_ckpt(args.out / "last.pth", epoch=epoch, model=model, optimizer=optimizer,
                  scheduler=scheduler, scaler=scaler, best_val=best_val,
                  bad_epochs=bad_epochs, total_epochs=args.epochs, generator=g)

        print(f"epoch {epoch:3d}  train {tr_loss:.4f}  val {val_loss:.4f}  "
              f"acc {val_acc:.4f}  lr {scheduler.get_last_lr()[0]:.2e}  "
              f"{time.perf_counter() - t0:.1f}s{'  *' if improved else ''}")

        if args.stop_after is not None and epoch + 1 - start_epoch >= args.stop_after:
            print(f"stopping after {args.stop_after} epoch(s) this run; resume with --resume")
            break
        if bad_epochs >= args.patience:
            print(f"early stop: no improvement in {args.patience} epochs")
            break

    print(f"best val loss {best_val:.4f} -> {args.out / 'best.pth'}")


if __name__ == "__main__":
    main()
