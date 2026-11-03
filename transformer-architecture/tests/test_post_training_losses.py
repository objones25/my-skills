"""post_training_losses.py: properties that separate the objectives."""
from __future__ import annotations

import contextlib
import io
import math

import pytest
import torch


def test_verify_passes(losses):
    with contextlib.redirect_stdout(io.StringIO()):
        assert losses.verify() == 0


def test_dpo_at_initialization_is_log_two(losses):
    z = torch.zeros(3)
    assert losses.dpo_loss(z, z, z, z)[0].item() == pytest.approx(math.log(2))


@pytest.mark.parametrize(("rewards", "keep"), [([1, 1, 1, 1], False), ([0, 0, 0, 0], False),
                                              ([1, 0, 0, 0], True)])
def test_dynamic_sampling_keeps_only_mixed_groups(losses, rewards, keep):
    assert losses.keep_group(torch.tensor(rewards, dtype=torch.float32)) is keep


def test_grpo_per_token_penalty_scales_inversely_with_length(losses):
    g = losses._token_grad("grpo", [10, 100], [-1.0, -1.0])
    assert (g[0, 0] / g[1, 0]).item() == pytest.approx(10.0)


def test_dr_grpo_weights_tokens_equally(losses):
    g = losses._token_grad("dr_grpo", [10, 100], [-1.0, -1.0], max_tokens=128)
    assert g[0, 0].item() == pytest.approx(g[1, 0].item())


def test_unknown_variant_is_rejected(losses):
    with pytest.raises(ValueError, match="ppo2"):
        losses.policy_loss(torch.zeros(1, 2), torch.zeros(1, 2), torch.ones(1), torch.ones(1, 2), "ppo2")


def test_every_explanation_names_a_mechanism(losses):
    assert all(len(text.split()) > 40 for text in losses.EXPLAIN.values())
