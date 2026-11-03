"""residual_probe.py: hook plumbing, and position 0 excluded by default."""
from __future__ import annotations

import contextlib
import io

import torch


def test_self_test_captures_every_block(probe):
    with contextlib.redirect_stdout(io.StringIO()):
        assert probe.self_test() == 0


def test_first_token_is_excluded_by_default(probe, block):
    model = block.Model(block.Cfg()).eval()
    ids = torch.randint(0, 128, (1, 6))
    with torch.no_grad():
        model.embed.weight[ids[0, 0]] *= 1000.0        # a massive activation at position 0
    kept = probe.probe_blocks(model, ids, blocks=list(model.blocks), skip_first=0)
    skipped = probe.probe_blocks(model, ids, blocks=list(model.blocks))
    assert skipped[0]["norm_in"] < kept[0]["norm_in"] / 10
