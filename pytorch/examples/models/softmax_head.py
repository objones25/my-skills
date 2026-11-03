"""Planted bug: softmax on the head, then CrossEntropyLoss applies log-softmax again.

Gradients are finite and reach every layer, so grads and nan pass. The loss is squeezed
into a narrow band (its floor is far above zero), so overfit on one batch is stuck.
"""
import torch
import torch.nn as nn


def make_model() -> nn.Module:
    return nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 4),
                         nn.Softmax(dim=-1))          # BUG: the loss owns the softmax


def make_batch():
    g = torch.Generator().manual_seed(0)
    return torch.randn(32, 16, generator=g), torch.randint(0, 4, (32,), generator=g)


loss_fn = nn.CrossEntropyLoss()
