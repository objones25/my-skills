"""A correct small classifier: the baseline every diagnose.py check passes on."""
import torch
import torch.nn as nn


def make_model() -> nn.Module:
    return nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 4))   # raw logits


def make_batch():
    g = torch.Generator().manual_seed(0)
    return torch.randn(32, 16, generator=g), torch.randint(0, 4, (32,), generator=g)


loss_fn = nn.CrossEntropyLoss()
