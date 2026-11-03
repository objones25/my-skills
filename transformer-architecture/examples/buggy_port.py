"""A plausible-looking port with two silent bugs, for `reference_block.py --verify --impl`.

Both functions produce correctly shaped tensors and a model that trains. See
examples/02-reviewing-a-port.md for what the verifier reports and why each one matters.
"""
import torch
import torch.nn as nn


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """(batch, n_kv_heads, seq, head_dim) -> (batch, n_kv_heads * n_rep, seq, head_dim)."""
    return x.repeat(1, n_rep, 1, 1)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / (x.pow(2).mean(-1, keepdim=True).sqrt() + self.eps) * self.weight
