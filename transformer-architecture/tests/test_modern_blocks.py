"""modern_blocks.py: every property holds on the reference code, and each detector fires."""
from __future__ import annotations

import contextlib
import io
import math

import pytest
import torch


def test_verify_passes(modern):
    with contextlib.redirect_stdout(io.StringIO()):
        assert modern.verify() == 0


@pytest.mark.parametrize("name", ["moe-bias-in-weights", "mla-rope-on-latent",
                                  "moe-topk-renorm-mismatch", "sink-added-to-logits"])
def test_bug_detector_catches_its_bug(modern, name):
    caught, _ = modern.BUG_DETECTORS[name]()
    assert caught


def test_every_detector_has_an_explanation(modern):
    assert set(modern.BUG_DETECTORS) == set(modern.BUGS)


def test_mla_decode_matches_full_forward(modern):
    m = modern.MLA()
    x = torch.randn(modern.B, modern.T, modern.D)
    torch.testing.assert_close(m.decode(x), m(x), atol=1e-4, rtol=1e-4)


def test_moe_bias_is_a_buffer_not_a_parameter(modern):
    m = modern.MoE()
    assert "bias" in dict(m.named_buffers())
    assert "bias" not in dict(m.named_parameters())


def test_sliding_window_mask_allows_exactly_w_keys_once_full(modern):
    mask = modern.causal_mask(10, 10, window=4)
    assert mask.sum(-1).tolist() == [1, 2, 3, 4, 4, 4, 4, 4, 4, 4]


def test_router_z_loss_of_uniform_logits(modern):
    assert modern.router_z_loss(torch.zeros(3, 16)).item() == pytest.approx(math.log(16) ** 2)
