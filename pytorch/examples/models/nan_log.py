"""Planted bug: a log of a value that can be negative produces NaN in one module."""
import torch
import torch.nn as nn


class LogFeatures(nn.Module):
    def forward(self, x):
        return torch.log(x)               # BUG: x can be negative; NaN from here on


def make_model() -> nn.Module:
    return nn.Sequential(nn.Linear(16, 32), LogFeatures(), nn.Linear(32, 4))


def make_batch():
    g = torch.Generator().manual_seed(0)
    return torch.randn(32, 16, generator=g), torch.randint(0, 4, (32,), generator=g)


loss_fn = nn.CrossEntropyLoss()
