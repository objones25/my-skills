"""Planted bug: hidden layers kept in a plain Python list.

The model runs and the loss falls a little (the last layer still trains), so nothing
crashes. The hidden layers are invisible to model.parameters(), so the optimizer never
updates them, .to(device) never moves them, and state_dict() never saves them.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = [nn.Linear(16, 16) for _ in range(3)]   # BUG: should be nn.ModuleList
        self.head = nn.Linear(16, 4)

    def forward(self, x):
        for layer in self.hidden:
            x = F.relu(layer(x))
        return self.head(x)


def make_model() -> nn.Module:
    return Net()


def make_batch():
    g = torch.Generator().manual_seed(0)
    return torch.randn(32, 16, generator=g), torch.randint(0, 4, (32,), generator=g)


loss_fn = nn.CrossEntropyLoss()
