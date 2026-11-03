"""Behavior tests for a PyTorch classifier. Copy this file next to your model.

Setup:
    1. Copy this file into your tests folder.
    2. Copy assets/conftest_torch.py into the same folder as ``conftest.py``
       (or merge its fixtures into the conftest you already have).
    3. Replace the block marked REPLACE THESE with your own model factory,
       batch factory and loss.
    4. Run ``pytest -q`` there. ``-m "not slow"`` skips the torch.compile test.

Each test checks one behavior that, when broken, still lets training run and
quietly learn less (or nothing). The failure messages name the offending
parameter or row so you can go straight to the bug.

Written against torch 2.15.0 and pytest 9.2.0; also run on torch 2.14.1.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

# ---------------------------------------------------------------------------
# REPLACE THESE: a placeholder model, batch and loss. Keep the three names.
# ---------------------------------------------------------------------------
IN_FEATURES = 8
NUM_CLASSES = 3
BATCH_SIZE = 16


class TinyClassifier(nn.Module):
    """Placeholder. Has BatchNorm and Dropout so the mode tests have teeth."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(IN_FEATURES, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(32, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, IN_FEATURES) -> (B, NUM_CLASSES)
        return self.net(x)


def make_model() -> nn.Module:
    """Return a freshly initialized model. Called once per test."""
    return TinyClassifier()


def make_batch(batch_size: int = BATCH_SIZE) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (inputs, int64 class targets) on the CPU."""
    x = torch.randn(batch_size, IN_FEATURES)
    y = torch.randint(0, NUM_CLASSES, (batch_size,))
    return x, y


def loss_fn(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, target)


# ---------------------------------------------------------------------------
# Fixtures and helpers. Usually no edits needed below this line.
# ---------------------------------------------------------------------------
DROPOUT_TYPES = (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d,
                 nn.AlphaDropout, nn.FeatureAlphaDropout)


@pytest.fixture
def model(device: torch.device) -> nn.Module:
    return make_model().to(device)


@pytest.fixture
def batch(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    x, y = make_batch()
    return x.to(device), y.to(device)


@pytest.fixture
def dropout_model(model: nn.Module) -> nn.Module:
    """The model, or a skip when it has no dropout layer to test."""
    if not any(isinstance(m, DROPOUT_TYPES) for m in model.modules()):
        pytest.skip("model has no dropout layer, so train mode is expected to be deterministic")
    return model


def _train(model: nn.Module, x: torch.Tensor, y: torch.Tensor, steps: int) -> list[float]:
    """Run ``steps`` Adam steps on one batch and return the loss before each step."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    model.train()
    losses = []
    for _ in range(steps):
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


def _seeded_run(seed: int, device: torch.device) -> list[float]:
    torch.manual_seed(seed)
    model = make_model().to(device)
    x, y = make_batch()
    return _train(model, x.to(device), y.to(device), steps=5)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size", [1, BATCH_SIZE], ids=["single-row", "full-batch"])
def test_output_shape_is_batch_by_classes(model, device, batch_size):
    """Batch size 1 is where a stray squeeze() or hardcoded dim breaks.

    Runs in eval mode: BatchNorm in train mode cannot normalize a batch of 1.
    """
    x, _ = make_batch(batch_size)
    model.eval()
    with torch.no_grad():
        out = model(x.to(device))
    assert out.shape == (batch_size, NUM_CLASSES)


@pytest.mark.parametrize(
    ("reduce", "normalized_value"),
    [
        pytest.param(lambda out: out.sum(dim=1), 1.0, id="softmax"),
        pytest.param(lambda out: out.logsumexp(dim=1), 0.0, id="log_softmax"),
    ],
)
def test_head_emits_raw_logits(model, batch, reduce, normalized_value):
    """cross_entropy applies log_softmax itself; a normalized head applies it twice."""
    x, _ = batch
    model.eval()
    with torch.no_grad():
        per_row = reduce(model(x))
    assert not torch.allclose(per_row, torch.full_like(per_row, normalized_value), atol=1e-4)


def test_every_parameter_gets_a_finite_gradient(model, batch):
    x, y = batch
    model.train()
    loss_fn(model(x), y).backward()
    missing_or_bad = [
        name
        for name, p in model.named_parameters()
        if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())
    ]
    assert missing_or_bad == []


def test_loss_drops_when_overfitting_one_batch(model, batch):
    """A model that cannot memorize 16 examples has a wiring bug, not a tuning problem."""
    x, y = batch
    losses = _train(model, x, y, steps=100)
    assert losses[-1] < 0.5 * losses[0], f"loss went {losses[0]:.4f} -> {losses[-1]:.4f}"


def test_training_is_reproducible_under_the_same_seed(device, deterministic):
    first = _seeded_run(seed=0, device=device)
    second = _seeded_run(seed=0, device=device)
    assert first == pytest.approx(second, rel=0, abs=0)


def test_eval_mode_is_deterministic(model, batch):
    x, _ = batch
    model.eval()
    with torch.no_grad():
        first, second = model(x), model(x)
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_train_mode_is_stochastic_when_dropout_is_present(dropout_model, batch):
    x, _ = batch
    dropout_model.train()
    with torch.no_grad():
        first, second = dropout_model(x), dropout_model(x)
    assert not torch.equal(first, second)


def test_state_dict_round_trip_reproduces_outputs(model, batch, device, tmp_path):
    """One train-mode forward first, so BatchNorm running stats differ from init.

    A tensor stored as a plain attribute instead of a buffer or parameter is
    missing from state_dict and makes this test fail.
    """
    x, _ = batch
    model.train()
    with torch.no_grad():
        model(x)
    path = tmp_path / "model.pt"
    torch.save(model.state_dict(), path)

    reloaded = make_model().to(device)
    reloaded.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    reloaded.eval()
    with torch.no_grad():
        torch.testing.assert_close(reloaded(x), model(x), rtol=0, atol=0)


def test_eval_output_for_a_row_ignores_the_other_rows(model, batch):
    """In eval mode row i must not depend on rows j != i (BatchNorm in train mode breaks this)."""
    x, _ = batch
    other_rows_changed = torch.cat([x[:1], torch.randn_like(x[1:])])
    model.eval()
    with torch.no_grad():
        alone = model(x)[0]
        with_others_changed = model(other_rows_changed)[0]
    torch.testing.assert_close(with_others_changed, alone)


@pytest.mark.slow
def test_compiled_model_matches_eager(model, batch, compile_ready):
    """Fused kernels round differently, so the tolerance is stated, not exact."""
    x, _ = batch
    model.eval()
    compiled = torch.compile(model)
    with torch.no_grad():
        torch.testing.assert_close(compiled(x), model(x), rtol=1e-4, atol=1e-5)
