#!/usr/bin/env python3
"""Report the PyTorch environment this code will actually run in.

Run this before quoting any API signature or recommending a device/dtype path.

    python scripts/check_env.py

Exit codes: 0 = torch imported fine, 1 = torch missing or broken.
"""

from __future__ import annotations

import importlib
import platform
import sys


def line(label: str, value: object) -> None:
    print(f"{label:<34} {value}")


def main() -> int:
    line("python", platform.python_version())
    line("platform", f"{platform.system()} {platform.machine()}")

    try:
        import torch
    except Exception as exc:  # noqa: BLE001 - we want the reason, whatever it is
        print(f"\ntorch NOT importable: {type(exc).__name__}: {exc}")
        print("Install with: pip install torch")
        return 1

    print()
    line("torch", torch.__version__)
    for name in ("torchvision", "torchaudio", "torchao", "tensorboard", "optuna", "peft"):
        try:
            line(name, importlib.import_module(name).__version__)
        except Exception:  # noqa: BLE001
            line(name, "-- not installed")

    print()
    acc = torch.accelerator.current_accelerator(check_available=True)
    device = acc or torch.device("cpu")
    line("accelerator", f"{device} (count={torch.accelerator.device_count()})")

    if device.type == "cuda":
        idx = torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(idx)
        line("gpu", torch.cuda.get_device_name(idx))
        line("compute capability", f"{major}.{minor}")
        line("cuda runtime", torch.version.cuda)
        free, total = torch.cuda.mem_get_info(idx)
        line("vram free / total", f"{free / 2**30:.1f} / {total / 2**30:.1f} GiB")
        line("bf16 supported", torch.cuda.is_bf16_supported())
        line("bf16 recommended (Ampere+)", major >= 8)
        line("GradScaler needed for fp16", True)
    elif device.type == "mps":
        line("float64 supported", False)
        line("GradScaler", "MPS code path in source; unverified by this skill")
        line("macOS requirement", "15+ for torch 2.15")
        line("recommended autocast dtype", "float16")
    else:
        line("recommended autocast dtype", "bfloat16 (CPU)")

    line("autocast available", torch.amp.is_autocast_available(device.type))
    print()

    line("torch.compile present", callable(getattr(torch, "compile", None)))  # presence only; a CPU compile also needs a C++ compiler
    line("triton", _version_or_missing("triton"))
    line("distributed available", torch.distributed.is_available())
    if torch.distributed.is_available():
        backends = [
            n
            for n, fn in (
                ("nccl", torch.distributed.is_nccl_available),
                ("gloo", torch.distributed.is_gloo_available),
                ("mpi", torch.distributed.is_mpi_available),
            )
            if fn()
        ]
        line("distributed backends", ", ".join(backends) or "none")

    print()
    line("deterministic algorithms", torch.are_deterministic_algorithms_enabled())
    line("cudnn.benchmark", torch.backends.cudnn.benchmark)
    line("float32 matmul precision", torch.get_float32_matmul_precision())

    major_minor = tuple(int(p) for p in torch.__version__.split(".")[:2])
    if major_minor != (2, 15):
        print(
            f"\nNOTE: this skill's references were verified against torch 2.15.0 (and re-checked\n"
            f"      on 2.14.1); you are on {torch.__version__}.\n"
            f"      Run `python scripts/verify_api.py --docs` to list what differs."
        )
    if major_minor >= (2, 15) and sys.version_info < (3, 11):
        print("\nNOTE: torch 2.15 requires Python 3.11 or newer.")
    return 0


def _version_or_missing(name: str) -> str:
    try:
        return importlib.import_module(name).__version__
    except Exception:  # noqa: BLE001
        return "-- not installed"


if __name__ == "__main__":
    sys.exit(main())
