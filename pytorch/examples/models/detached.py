"""Planted bug: a stray .detach() cuts the encoder out of the graph.

The head still trains, so the loss moves; the encoder's gradients are None.
"""
import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(16, 32), nn.ReLU())
        self.head = nn.Linear(32, 4)

    def forward(self, x):
        h = self.encoder(x).detach()      # BUG: e.g. left over from logging features
        return self.head(h)


def make_model() -> nn.Module:
    return Net()


def make_batch():
    g = torch.Generator().manual_seed(0)
    return torch.randn(32, 16, generator=g), torch.randint(0, 4, (32,), generator=g)


loss_fn = nn.CrossEntropyLoss()
