#!/usr/bin/env python3
"""Report what this machine can do for massively parallel RL.

Prints installed versions (jax, flax, optax, gymnax, craftax, torch, mujoco, brax,
mujoco_playground, isaaclab), the accelerators each framework sees, and a one-line
verdict. Run it before quoting throughput or picking a framework: a JAX install that
silently fell back to CPU is the most common reason a "GPU" run is 100x slow.

Usage: python check_env.py
"""
from __future__ import annotations

import importlib
import importlib.metadata as md
import platform
import sys

PKGS = ["jax", "jaxlib", "flax", "optax", "distrax", "gymnax", "craftax", "kinetix-env",
        "brax", "mujoco", "mujoco-mjx", "playground", "torch", "isaaclab", "pufferlib"]


def version(dist: str) -> str | None:
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return None


def main() -> int:
    print(f"python {sys.version.split()[0]} on {platform.system()} {platform.machine()}")
    for p in PKGS:
        v = version(p)
        if v:
            print(f"  {p:<12} {v}")
    verdict = []

    if version("jax"):
        import jax
        devs = jax.devices()
        kinds = sorted({d.platform for d in devs})
        print(f"jax devices: {devs}")
        if kinds == ["cpu"]:
            verdict.append("JAX is CPU-only: fine for tests and tiny N, not for scaling runs")
        else:
            verdict.append(f"JAX accelerator: {kinds} x{len([d for d in devs if d.platform != 'cpu'])}")

    if version("torch"):
        import torch
        cuda = torch.cuda.is_available()
        mps = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
        print(f"torch cuda={cuda} mps={mps}"
              + (f" devices={[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}" if cuda else ""))
        if cuda:
            verdict.append("PyTorch CUDA available: Isaac Lab / FastTD3-style GPU training possible")
        elif mps:
            verdict.append("PyTorch MPS only: networks run on the Apple GPU, but GPU simulators "
                           "(Isaac Lab, MJX via JAX CUDA) do not")
        else:
            verdict.append("PyTorch CPU-only")

    if not verdict:
        verdict.append("neither jax nor torch installed")
    print("verdict: " + "; ".join(verdict))
    return 0


if __name__ == "__main__":
    sys.exit(main())
