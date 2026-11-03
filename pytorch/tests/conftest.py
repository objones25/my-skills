"""Shared fixtures. Scripts are loaded by file path, so no sys.path edits are needed."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

SKILL_ROOT = Path(__file__).resolve().parent.parent


def load_module(relpath: str, name: str):
    spec = importlib.util.spec_from_file_location(name, SKILL_ROOT / relpath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


@pytest.fixture(scope="session")
def skill_root() -> Path:
    return SKILL_ROOT


@pytest.fixture(scope="session")
def verify_api():
    return load_module("scripts/verify_api.py", "verify_api")


@pytest.fixture(scope="session")
def template():
    return load_module("assets/train_template.py", "train_template")
