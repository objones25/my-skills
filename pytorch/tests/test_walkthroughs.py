"""Every documented walkthrough output must be what the code actually prints."""
from __future__ import annotations

import importlib.util

import pytest

from walkthrough_runner import run_walkthrough

FILES = [
    "01-devices-amp.md", "02-tensors-autograd.md", "03-models-layers.md",
    "04-losses-optimizers.md", "05-data-training-loop.md", "06-tuning-reproducibility.md",
    pytest.param("07-fine-tuning.md", marks=pytest.mark.skipif(
        importlib.util.find_spec("torchvision") is None,
        reason="07 walkthrough builds a torchvision ResNet")),
    pytest.param("08-transforms-augmentation.md", marks=pytest.mark.skipif(
        importlib.util.find_spec("torchvision") is None,
        reason="08 walkthrough runs torchvision.transforms.v2")),
    "09-attention-transformers.md",
    "10-debugging.md", "11-compile-export.md", "12-distributed.md",
    "13-performance-profiling.md",
    pytest.param("14-quantization.md", marks=pytest.mark.skipif(
        importlib.util.find_spec("torchao") is None,
        reason="14 walkthrough steps 1-3 run torchao quantize_; later steps reuse their objects")),
    "15-custom-ops-autograd.md", "16-testing-pytorch-code.md",
]
OPTIONAL = tuple(m for m in ("peft",) if importlib.util.find_spec(m) is None)


@pytest.mark.parametrize("name", FILES)
def test_walkthrough_output_matches_documented_output(skill_root, name):
    mismatches = run_walkthrough(skill_root / "references" / name, skip_modules=OPTIONAL)
    assert mismatches == []
