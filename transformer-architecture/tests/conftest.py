"""Shared fixtures. Scripts are loaded by file path, so no sys.path edits are needed."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

SKILL_ROOT = Path(__file__).resolve().parent.parent


def load_module(relpath: str, name: str):
    spec = importlib.util.spec_from_file_location(name, SKILL_ROOT / relpath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module            # dataclasses resolve their module by name
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


@pytest.fixture(scope="session")
def skill_root() -> Path:
    return SKILL_ROOT


@pytest.fixture(scope="session")
def budget():
    return load_module("scripts/config_budget.py", "config_budget")


@pytest.fixture(scope="session")
def block():
    return load_module("scripts/reference_block.py", "reference_block")


@pytest.fixture(scope="session")
def probe():
    return load_module("scripts/residual_probe.py", "residual_probe")


@pytest.fixture(scope="session")
def modern():
    return load_module("scripts/modern_blocks.py", "modern_blocks")


@pytest.fixture(scope="session")
def losses():
    return load_module("scripts/post_training_losses.py", "post_training_losses")


@pytest.fixture(scope="session")
def spikes():
    return load_module("scripts/spike_monitor.py", "spike_monitor")
