"""Fixtures for assets/test_model_template.py.

Copy this file next to the template as ``conftest.py`` (or merge it into an
existing ``tests/conftest.py``). pytest imports a file named ``conftest.py``
automatically; it does not import ``conftest_torch.py`` by that name.

Fixtures provided:

    _seed_everything (autouse)   seeds Python, NumPy (if installed) and torch
                                 before every test
    device                       "cpu", plus "cuda" and "mps" when present;
                                 the accelerator cases carry the ``gpu`` marker
    deterministic                turns on torch.use_deterministic_algorithms
                                 for one test and restores the previous mode
    compile_ready                skips a test cleanly when torch.compile
                                 cannot run here, with the reason

Written against torch 2.15.0 and pytest 9.2.0; also run on torch 2.14.1.

Where this differs from the pytest-expert skill's ``assets/conftest_ml.py``,
and why:

1. Determinism is an opt-in fixture, not autouse. conftest_ml.py turns on
   ``use_deterministic_algorithms(True, warn_only=True)`` for every test and
   afterwards forces it to ``False``. Here the ``deterministic`` fixture uses
   ``warn_only=False`` so an op with no deterministic kernel raises
   ``RuntimeError`` in the test that asked for determinism, and it restores
   whatever mode was active before (including warn-only) instead of forcing
   ``False``.
2. ``CUBLAS_WORKSPACE_CONFIG`` is set once in ``pytest_configure``, before any
   test can create a CUDA context. cuBLAS reads it when the context is
   created, so setting it inside a per-test fixture (as conftest_ml.py does)
   only takes effect if that fixture happens to run before the first CUDA
   call. It is set with ``setdefault`` so a value you export yourself wins.
3. The ``device`` fixture adds Apple's MPS backend next to CUDA.
4. Markers (``slow``, ``gpu``) are registered here in ``pytest_configure`` so
   the template runs under ``--strict-markers`` with no ini file. If your
   project config already registers them, the duplicate line is harmless.
5. The seed is the same value (20260826) so results line up across both
   skills.
"""

from __future__ import annotations

import os
import random
import shutil
from collections.abc import Iterator

import pytest
import torch

SEED = 20260826


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: takes more than about a second")
    config.addinivalue_line("markers", "gpu: needs a CUDA or MPS device")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


@pytest.fixture(autouse=True)
def _seed_everything() -> None:
    """Seed every global RNG a model or data pipeline might draw from.

    torch.manual_seed seeds the CPU generator and every CUDA device. Python's
    ``random`` and NumPy's global generator are separate and need their own
    calls.
    """
    random.seed(SEED)
    torch.manual_seed(SEED)
    try:
        import numpy as np
    except ImportError:
        return
    np.random.seed(SEED)


def _mps_available() -> bool:
    return torch.backends.mps.is_available()


@pytest.fixture(
    params=[
        "cpu",
        pytest.param("cuda", marks=pytest.mark.gpu),
        pytest.param("mps", marks=pytest.mark.gpu),
    ]
)
def device(request: pytest.FixtureRequest) -> torch.device:
    """Run the requesting test once per device.

    Accelerator cases skip with a reason when the hardware is absent, so a
    laptop run reports them as skipped rather than silently passing.
    Deselect them entirely with ``-m "not gpu"``.
    """
    available = {
        "cpu": True,
        "cuda": torch.cuda.is_available(),
        "mps": _mps_available(),
    }
    if not available[request.param]:
        pytest.skip(f"{request.param} is not available on this machine")
    return torch.device(request.param)


@pytest.fixture
def deterministic() -> Iterator[None]:
    """Force deterministic kernels for one test, then restore the prior mode.

    ``torch.use_deterministic_algorithms(True)`` makes every op use a
    deterministic implementation where one exists and raise ``RuntimeError``
    where none does. ``cudnn.benchmark`` (re-timing conv algorithms per run)
    is switched off for the same reason.
    """
    was_enabled = torch.are_deterministic_algorithms_enabled()
    was_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    was_benchmark = torch.backends.cudnn.benchmark
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(was_enabled, warn_only=was_warn_only)
        torch.backends.cudnn.benchmark = was_benchmark


def _has_cxx_compiler() -> bool:
    return any(shutil.which(name) for name in ("c++", "g++", "clang++"))


@pytest.fixture
def compile_ready(device: torch.device) -> None:
    """Skip, with the reason, when torch.compile cannot run on this machine.

    torch.compile's default backend (Inductor) generates C++ for CPU, so a CPU
    run needs a C++ compiler on PATH. ``torch._dynamo.is_dynamo_supported`` is
    a private helper; it checks the Python version and platform.
    """
    if not torch._dynamo.is_dynamo_supported():
        pytest.skip("torch.compile is not supported on this Python or platform")
    if device.type == "cpu" and not _has_cxx_compiler():
        pytest.skip("Inductor on CPU needs a C++ compiler (c++, g++ or clang++) on PATH")
