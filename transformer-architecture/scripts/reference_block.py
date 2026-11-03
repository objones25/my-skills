#!/usr/bin/env python3
"""
A minimal, correct decoder block: RMSNorm + RoPE + GQA + SwiGLU + KV cache.

Two jobs. First, it is the canonical implementation to copy from or diff
against. Second, `--verify` runs the property tests that catch the bugs each
component invites, run it after you port this into your own codebase and the
tests will tell you whether the port is right.

    python reference_block.py --verify        # property tests, real output
    python reference_block.py --shapes        # every tensor shape, one example
    python reference_block.py --bug rope-split-order   # explain one bug
    python reference_block.py --verify --impl my_impl.py   # test YOUR apply_rope,
                                                # repeat_kv and/or RMSNorm instead

The whole file uses one toy config so the shapes stay small enough to read:

    d_model=64  n_layers=4  n_heads=4  n_kv_heads=2  head_dim=16
    d_ff=176    vocab=128   seq_len=6  batch=2

Every shape comment is (batch, ...) with those numbers substituted.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Cfg:
    d_model: int = 64
    n_layers: int = 4
    n_heads: int = 4
    n_kv_heads: int = 2
    head_dim: int = 16
    d_ff: int = 176
    vocab: int = 128
    seq_len: int = 6
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    tie_embeddings: bool = True
    qk_norm: bool = False             # RMSNorm on q and k per head, before RoPE


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """
    x / sqrt(mean(x^2) + eps) * gain

    No mean subtraction and no bias, that is the entire difference from
    LayerNorm, and the reason it is cheaper. Zhang & Sennrich 2019 report
    7-64% faster than LayerNorm depending on the model.

    The eps goes INSIDE the sqrt, added to the mean square. Putting it
    outside changes the scale of small-norm vectors and is a real bug that
    passes every shape check.

    Cast to float32 before squaring. PyTorch already accumulates a bf16
    reduction in fp32, so the accumulation is not the problem; the squaring
    and the bf16 output are. Over 10,000 vectors of width 4096, the bf16 path's
    relative error on mean(x^2) averages 1.5e-3 and peaks near 4e-3; an fp32
    upcast adds only ~1e-7 beyond the input's own bf16 rounding. The bf16 error
    is zero-mean noise, up to ~0.2% in the norm scale at every norm site.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dtype) * self.weight)


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------


def rope_tables(head_dim: int, seq_len: int, theta: float, device=None,
                dtype=torch.float32):
    """
    Precompute cos and sin, each (seq_len, head_dim).

    theta_i = theta ** (-2i/head_dim) for i in [0, head_dim/2).
    Low i oscillates fast (short-range), high i slowly (long-range).

    Note the table is built for the *halves* pairing used below, so each
    frequency appears twice, in position i and position i + head_dim/2.
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float()
                                / head_dim))                    # (head_dim/2,)
    pos = torch.arange(seq_len, device=device).float()           # (seq_len,)
    freqs = torch.outer(pos, inv_freq)                           # (seq_len, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)                      # (seq_len, head_dim)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """[x1, x2] -> [-x2, x1], splitting on the last dim."""
    d = x.shape[-1] // 2
    return torch.cat((-x[..., d:], x[..., :d]), dim=-1)


def apply_rope(q, k, cos, sin):
    """
    q, k: (batch, n_heads, seq, head_dim). cos, sin: (seq, head_dim).

    Two conventions exist and they are NOT interchangeable:
      halves     pairs dim i with dim i + head_dim/2   <- used here, and by
                 Hugging Face Llama, so checkpoints are compatible
      interleaved pairs dim 2i with dim 2i+1           <- the RoFormer paper,
                 and some from-scratch codebases
    Either is a valid rotation and either trains fine from scratch. Mixing
    them silently destroys a pretrained checkpoint: the model still runs and
    the loss is merely bad, not NaN, which is what makes it expensive to find.

    RoPE touches Q and K only. V is never rotated, the position signal
    belongs in the score, not in the content being retrieved.
    """
    cos = cos[None, None, :, :]      # (1, 1, seq, head_dim), broadcasts over heads
    sin = sin[None, None, :, :]
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


# ---------------------------------------------------------------------------
# Grouped-query attention
# ---------------------------------------------------------------------------


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    (batch, n_kv_heads, seq, head_dim) -> (batch, n_kv_heads * n_rep, seq, head_dim)

    KV head j serves query heads [j*n_rep, (j+1)*n_rep), contiguous blocks,
    not interleaved. `x.repeat_interleave(n_rep, dim=1)` gives the same
    layout; `x.repeat(1, n_rep, 1, 1)` tiles the whole set instead and is the
    standard GQA bug. Both have the right shape.

    Be precise about what the bug costs. `verify()` proves the two differ by
    exactly a permutation of query heads: permute the head blocks of W_q and
    the matching column blocks of W_o and the outputs match to 1.8e-07. So
    from scratch it trains fine, because the model just learns the permuted
    labelling. It is fatal at the boundary, where every other implementation
    assumes the contiguous layout: exported checkpoints, imported
    checkpoints, `enable_gqa=True`, and any fused kernel. It also materialises
    the full expanded K and V, which `expand` does not.
    """
    if n_rep == 1:
        return x
    b, n_kv, s, d = x.shape
    return x[:, :, None, :, :].expand(b, n_kv, n_rep, s, d).reshape(b, n_kv * n_rep, s, d)


class Attention(nn.Module):
    def __init__(self, c: Cfg):
        super().__init__()
        assert c.n_heads % c.n_kv_heads == 0, "GQA groups must be equal-sized"
        self.c = c
        self.n_rep = c.n_heads // c.n_kv_heads
        q_out, kv_out = c.n_heads * c.head_dim, c.n_kv_heads * c.head_dim
        self.q_proj = nn.Linear(c.d_model, q_out, bias=False)
        self.k_proj = nn.Linear(c.d_model, kv_out, bias=False)
        self.v_proj = nn.Linear(c.d_model, kv_out, bias=False)
        self.o_proj = nn.Linear(q_out, c.d_model, bias=False)
        if c.qk_norm:
            # QK-norm (Henry et al. 2020; ViT-22B; OLMo 2, Qwen3): one gain vector per
            # projection, shared across heads, 2 * head_dim parameters per layer.
            self.q_norm = RMSNorm(c.head_dim, c.norm_eps)
            self.k_norm = RMSNorm(c.head_dim, c.norm_eps)

    def forward(self, x, cos, sin, cache=None):
        b, s, _ = x.shape                       # (2, 6, 64)
        c = self.c

        # Project, then split heads. The split must come BEFORE RoPE: RoPE
        # rotates within each head's head_dim slots, so applying it to the
        # flat (b, s, n_heads*head_dim) tensor rotates across head boundaries.
        q = self.q_proj(x).view(b, s, c.n_heads, c.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, c.n_kv_heads, c.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, c.n_kv_heads, c.head_dim).transpose(1, 2)
        # q (2, 4, 6, 16)   k, v (2, 2, 6, 16)

        if c.qk_norm:                           # normalize per head, then rotate
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rope(q, k, cos, sin)

        if cache is not None:
            # Cache K and V only. The current token's query is all attention
            # needs from the query side; past queries are never revisited.
            # Append AFTER rotating, so each cached key keeps the rotation
            # for its own absolute position.
            if cache.get("k") is not None:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v

        k = repeat_kv(k, self.n_rep)            # (2, 4, kv_len, 16)
        v = repeat_kv(v, self.n_rep)

        # is_causal is only correct when q and k have equal length. During
        # incremental decode q is length 1 against a longer k, and a causal
        # mask on that rectangle would mask out the entire history.
        is_causal = cache is None or k.shape[2] == q.shape[2]
        out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        # (2, 4, 6, 16) -> back to (2, 6, 64)
        out = out.transpose(1, 2).reshape(b, s, c.n_heads * c.head_dim)
        return self.o_proj(out)


def z_loss(logits: torch.Tensor, coef: float = 1e-4) -> torch.Tensor:
    """PaLM's auxiliary loss, coef * mean(log Z)^2 with Z the softmax normalizer.

    Add it to the cross-entropy to keep log Z near 0. PaLM and Wortsman et al.
    use 1e-4; Chameleon uses 1e-5. logits: (..., vocab).
    """
    return coef * torch.logsumexp(logits.float(), dim=-1).pow(2).mean()


def softcap(logits: torch.Tensor, cap: float) -> torch.Tensor:
    """Gemma 2's logit softcapping: cap * tanh(logits / cap). 50 on attention, 30 on output."""
    return cap * torch.tanh(logits / cap)


# ---------------------------------------------------------------------------
# SwiGLU MLP
# ---------------------------------------------------------------------------


def swiglu_d_ff(d_model: int, expansion: float = 4.0, multiple_of: int = 256) -> int:
    """
    A gated MLP has three matrices where a plain one has two, so to hold the
    parameter count fixed you scale the hidden width by 2/3 (Shazeer 2020,
    which took T5-base's d_ff from 3072 to 2048). Then round up to a multiple
    of 256 so the GEMM tiles cleanly.
    """
    d = int(2 * expansion * d_model / 3)
    return multiple_of * ((d + multiple_of - 1) // multiple_of)


class SwiGLU(nn.Module):
    """
    down( silu(gate(x)) * up(x) )

    silu(z) = z * sigmoid(z), i.e. Swish with beta=1. Shazeer 2020 tested
    this against ReLU/GELU/plain-GLU on T5 and got the best log-perplexity
    of the family (1.636 vs 1.677 for ReLU at matched parameters), while
    declining to explain why: "we attribute their success, as all else, to
    divine benevolence."

    No biases. Every gated-MLP model since Llama omits them.
    """

    def __init__(self, c: Cfg):
        super().__init__()
        self.gate_proj = nn.Linear(c.d_model, c.d_ff, bias=False)
        self.up_proj = nn.Linear(c.d_model, c.d_ff, bias=False)
        self.down_proj = nn.Linear(c.d_ff, c.d_model, bias=False)

    def forward(self, x):                       # (2, 6, 64)
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Block and model
# ---------------------------------------------------------------------------


class Block(nn.Module):
    """
    Pre-norm: x = x + sublayer(norm(x)).

    The residual path is clean end to end, no norm sits on it, which is why
    pre-norm trains without warmup where post-norm needs it (Xiong et al.
    2020). The cost is that nothing renormalizes the stream between blocks,
    so its norm grows with depth. See references/stability.md.
    """

    def __init__(self, c: Cfg):
        super().__init__()
        self.attn_norm = RMSNorm(c.d_model, c.norm_eps)
        self.attn = Attention(c)
        self.mlp_norm = RMSNorm(c.d_model, c.norm_eps)
        self.mlp = SwiGLU(c)

    def forward(self, x, cos, sin, cache=None):
        x = x + self.attn(self.attn_norm(x), cos, sin, cache)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class Model(nn.Module):
    def __init__(self, c: Cfg):
        super().__init__()
        self.c = c
        self.embed = nn.Embedding(c.vocab, c.d_model)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layers))
        self.final_norm = RMSNorm(c.d_model, c.norm_eps)
        self.lm_head = nn.Linear(c.d_model, c.vocab, bias=False)
        if c.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        cos, sin = rope_tables(c.head_dim, c.seq_len * 4, c.rope_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init)
        # GPT-2's residual scaling: every one of the 2*n_layers writes into the
        # stream is shrunk by 1/sqrt(2*n_layers) so the accumulated variance at
        # the top does not grow with depth. Applied to output projections only.
        for name, p in self.named_parameters():
            if name.endswith(("o_proj.weight", "down_proj.weight")):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * c.n_layers))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, ids, caches=None, pos_offset: int = 0):
        s = ids.shape[1]
        cos = self.cos[pos_offset:pos_offset + s]
        sin = self.sin[pos_offset:pos_offset + s]
        x = self.embed(ids)
        for i, blk in enumerate(self.blocks):
            x = blk(x, cos, sin, None if caches is None else caches[i])
        return self.lm_head(self.final_norm(x))


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


def _repeat_bug_is_a_permutation(c: Cfg):
    """
    Show that `.repeat(1, n_rep, 1, 1)` is not "wrong attention" but a
    relabelling of query heads. Correct: query head j reads KV head
    j // n_rep. Buggy: query head i reads KV head i % n_kv. Reorder the head
    blocks of W_q, and the matching column blocks of W_o, by the perfect
    shuffle and the two produce identical output.

    Returns (diff with identical weights, diff with permuted weights, perm).
    """
    import copy
    torch.manual_seed(0)
    n_rep, n_kv, hd = c.n_heads // c.n_kv_heads, c.n_kv_heads, c.head_dim
    perm = [r * n_kv + g for g in range(n_kv) for r in range(n_rep)]
    a = Attention(c).eval()
    cos, sin = rope_tables(hd, c.seq_len, c.rope_theta)
    x = torch.randn(2, c.seq_len, c.d_model)

    def buggy(att):
        b, s, _ = x.shape
        q = att.q_proj(x).view(b, s, c.n_heads, hd).transpose(1, 2)
        k = att.k_proj(x).view(b, s, n_kv, hd).transpose(1, 2)
        v = att.v_proj(x).view(b, s, n_kv, hd).transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)
        k, v = k.repeat(1, n_rep, 1, 1), v.repeat(1, n_rep, 1, 1)   # the bug
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return att.o_proj(o.transpose(1, 2).reshape(b, s, -1))

    b_ = copy.deepcopy(a)
    with torch.no_grad():
        qw = a.q_proj.weight.view(c.n_heads, hd, c.d_model)
        b_.q_proj.weight.copy_(qw[perm].reshape(c.n_heads * hd, c.d_model))
        ow = a.o_proj.weight.view(c.d_model, c.n_heads, hd)
        b_.o_proj.weight.copy_(ow[:, perm, :].reshape(c.d_model, c.n_heads * hd))
        ref = a(x, cos, sin)
        return ((ref - buggy(a)).abs().max().item(),
                (ref - buggy(b_)).abs().max().item(), perm)


def _check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    return bool(ok)


def verify() -> int:
    torch.manual_seed(0)
    c = Cfg()
    fails = 0
    print("RMSNorm")
    x = torch.randn(2, 6, c.d_model)
    n = RMSNorm(c.d_model, c.norm_eps)
    manual = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + c.norm_eps)
    fails += not _check("matches x/sqrt(mean(x^2)+eps) with unit gain",
                        torch.allclose(n(x), manual, atol=1e-5),
                        f"max abs diff {(n(x)-manual).abs().max():.2e}")
    fails += not _check("mean is NOT removed (unlike LayerNorm)",
                        not torch.allclose(n(x).mean(-1), torch.zeros(2, 6), atol=1e-3),
                        f"output mean {n(x).mean().item():+.4f}")
    zero = n(torch.zeros(1, c.d_model))
    fails += not _check("an all-zero vector stays finite (eps inside the sqrt)",
                        torch.isfinite(zero).all().item())
    xz = torch.zeros(1, c.d_model, requires_grad=True)
    n(xz).sum().backward()
    fails += not _check("...and gets a finite gradient (catches x / (sqrt(ms) + eps))",
                        torch.isfinite(xz.grad).all().item(),
                        "sqrt has an infinite derivative at 0; eps must be under it")

    print("\nRoPE")
    cos, sin = rope_tables(c.head_dim, 64, c.rope_theta)
    q = torch.randn(1, 1, 1, c.head_dim)
    k = torch.randn(1, 1, 1, c.head_dim)

    def score(m, n_):
        qm, _ = apply_rope(q, q, cos[m:m + 1], sin[m:m + 1])
        _, kn = apply_rope(k, k, cos[n_:n_ + 1], sin[n_:n_ + 1])
        return (qm * kn).sum().item()

    s_3_5, s_10_12, s_3_6 = score(3, 5), score(10, 12), score(3, 6)
    fails += not _check("score depends only on relative distance",
                        abs(s_3_5 - s_10_12) < 1e-4,
                        f"q@3.k@5 = {s_3_5:+.6f}   q@10.k@12 = {s_10_12:+.6f}")
    fails += not _check("...and a different distance gives a different score",
                        abs(s_3_5 - s_3_6) > 1e-3,
                        f"q@3.k@6 = {s_3_6:+.6f}")
    fails += not _check("rotation preserves norm",
                        torch.allclose(apply_rope(q, k, cos[7:8], sin[7:8])[0].norm(),
                                       q.norm(), atol=1e-5))

    print("\nGQA")
    fails += not _check("repeat_kv maps kv head j to query heads [j*n_rep, (j+1)*n_rep)",
                        torch.equal(repeat_kv(torch.arange(4).float()
                                              .view(1, 4, 1, 1), 2).flatten(),
                                    torch.tensor([0., 0., 1., 1., 2., 2., 3., 3.])),
                        "got " + str(repeat_kv(torch.arange(4).float()
                                               .view(1, 4, 1, 1), 2).flatten().tolist()))
    wrong = torch.arange(4).float().view(1, 4, 1, 1).repeat(1, 2, 1, 1).flatten().tolist()
    fails += not _check(".repeat() gives the wrong layout at the same shape",
                        wrong == [0., 1., 2., 3., 0., 1., 2., 3.], f"{wrong}")
    mha_cfg = Cfg(n_kv_heads=4)
    a_gqa, a_mha = Attention(Cfg()), Attention(mha_cfg)
    fails += not _check("n_kv_heads == n_heads reduces to plain MHA",
                        a_mha.n_rep == 1 and a_gqa.n_rep == 2,
                        f"n_rep: GQA {a_gqa.n_rep}, MHA {a_mha.n_rep}")
    if "repeat_kv" in IMPL_SWAPPED:
        print("  [skip] ...and .repeat() is exactly a query-head permutation  "
              "(a property of the reference repeat_kv; not meaningful for yours)")
    else:
        d_same, d_perm, perm = _repeat_bug_is_a_permutation(c)
        fails += not _check("...and .repeat() is exactly a query-head permutation",
                            d_same > 1e-2 and d_perm < 1e-5,
                            f"perm {perm}: same weights differ by {d_same:.2e}, "
                            f"permuted W_q/W_o match to {d_perm:.2e}")

    print("\nSwiGLU sizing")
    d = swiglu_d_ff(4096)
    plain = 2 * 4096 * (4 * 4096)
    gated = 3 * 4096 * d
    fails += not _check("2/3 rule holds parameters roughly fixed vs a 4x plain MLP",
                        abs(gated - plain) / plain < 0.06,
                        f"plain 4x = {plain/1e6:.1f}M, gated d_ff={d} = {gated/1e6:.1f}M "
                        f"({gated/plain:.3f}x)")

    print("\nCausality and KV cache")
    m = Model(c).eval()
    ids = torch.randint(0, c.vocab, (2, c.seq_len))
    with torch.no_grad():
        full = m(ids)
        tampered = ids.clone()
        tampered[:, -1] = (tampered[:, -1] + 7) % c.vocab
        after = m(tampered)
    fails += not _check("changing the last token leaves earlier logits identical",
                        torch.allclose(full[:, :-1], after[:, :-1], atol=1e-5),
                        f"max abs diff on prefix {(full[:,:-1]-after[:,:-1]).abs().max():.2e}")

    with torch.no_grad():
        caches = [{"k": None, "v": None} for _ in range(c.n_layers)]
        step = []
        for t in range(c.seq_len):
            step.append(m(ids[:, t:t + 1], caches=caches, pos_offset=t))
        step = torch.cat(step, dim=1)
    diff = (full - step).abs().max().item()
    fails += not _check("token-by-token decode with cache == one full forward",
                        diff < 1e-4, f"max abs diff {diff:.2e}")

    print("\nWeight tying and init")
    fails += not _check("lm_head and embedding are the same tensor",
                        m.lm_head.weight.data_ptr() == m.embed.weight.data_ptr())
    o_std = m.blocks[0].attn.o_proj.weight.std().item()
    q_std = m.blocks[0].attn.q_proj.weight.std().item()
    expect = 0.02 / math.sqrt(2 * c.n_layers)
    fails += not _check("residual output projections are scaled by 1/sqrt(2*n_layers)",
                        abs(o_std - expect) / expect < 0.15,
                        f"o_proj std {o_std:.4f} (target {expect:.4f}), q_proj std {q_std:.4f}")

    print("\nQK-norm, z-loss, softcap")
    a_qk = Attention(Cfg(qk_norm=True))
    xq = torch.randn(2, c.seq_len, c.d_model) * 50          # large activations
    qn = a_qk.q_norm(a_qk.q_proj(xq).view(2, c.seq_len, c.n_heads, c.head_dim).transpose(1, 2))
    kn = a_qk.k_norm(a_qk.k_proj(xq).view(2, c.seq_len, c.n_kv_heads, c.head_dim).transpose(1, 2))
    max_logit = (qn @ repeat_kv(kn, a_qk.n_rep).transpose(-1, -2)).abs().max().item() / math.sqrt(c.head_dim)
    fails += not _check("QK-norm bounds |logit| by sqrt(head_dim) at unit gain",
                        max_logit <= math.sqrt(c.head_dim) + 1e-4,
                        f"max |logit| {max_logit:.3f} <= {math.sqrt(c.head_dim):.3f}")
    extra = sum(p.numel() for p in a_qk.parameters()) - sum(p.numel() for p in Attention(c).parameters())
    fails += not _check("per-head QK-norm adds 2 * head_dim parameters per layer", extra == 2 * c.head_dim,
                        f"{extra} = 2 x {c.head_dim}")
    lg = torch.randn(4, c.vocab)
    zl = z_loss(lg, 1e-4).item()
    want = 1e-4 * torch.logsumexp(lg, -1).pow(2).mean().item()
    fails += not _check("z_loss = coef * mean(logsumexp^2)", abs(zl - want) < 1e-9, f"{zl:.3e}")
    fails += not _check("softcap keeps |logit| below the cap",
                        softcap(torch.tensor([1e4, -1e4]), 30.0).abs().max().item() <= 30.0)

    print("\nGradient flow")
    m2 = Model(c)
    # Next-token targets: position t predicts token t+1. Initial loss sits near ln(vocab).
    loss = F.cross_entropy(m2(ids)[:, :-1].reshape(-1, c.vocab), ids[:, 1:].reshape(-1))
    loss.backward()
    dead = [n_ for n_, p in m2.named_parameters()
            if p.grad is None or p.grad.abs().max() == 0]
    fails += not _check("every parameter receives gradient", not dead,
                        f"loss {loss.item():.4f}, "
                        + (f"dead: {dead[:3]}" if dead else f"ln(vocab)={math.log(c.vocab):.4f}"))

    print("\nBug detectors (each check above must FAIL on the bug it targets)")
    for name, detect in BUG_DETECTORS.items():
        caught, detail = detect()
        fails += not _check(f"{name}: caught", caught, detail)

    print(f"\n{'all checks passed' if not fails else f'{fails} CHECK(S) FAILED'}")
    return fails


# ---------------------------------------------------------------------------
# Bug detectors: run a property check against a deliberately buggy version.
# A check that cannot fail on its own bug proves nothing.
# ---------------------------------------------------------------------------


def _detect_rope_before_split():
    """RoPE over the flat (n_heads*head_dim) vector, then split: per-head scores lose
    the relative-distance property."""
    c = Cfg()
    torch.manual_seed(0)
    flat = c.n_heads * c.head_dim
    cos, sin = rope_tables(flat, 64, c.rope_theta)
    q, k = torch.randn(1, 1, 1, flat), torch.randn(1, 1, 1, flat)

    def head0_score(m, n_):
        qm, _ = apply_rope(q, q, cos[m:m + 1], sin[m:m + 1])
        _, kn = apply_rope(k, k, cos[n_:n_ + 1], sin[n_:n_ + 1])
        return (qm[..., :c.head_dim] * kn[..., :c.head_dim]).sum().item()

    a, b = head0_score(3, 5), head0_score(10, 12)
    return abs(a - b) > 1e-3, f"head-0 score at (3,5) {a:+.4f} vs (10,12) {b:+.4f}"


def _detect_eps_outside_sqrt():
    x = torch.zeros(1, 8, requires_grad=True)
    after = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True)) + 1e-6       # eps added to the output
    denom = x / (x.pow(2).mean(-1, keepdim=True).sqrt() + 1e-6)           # eps added to the sqrt
    denom.sum().backward()
    fwd_nan = not torch.isfinite(after).all().item()
    grad_nan = not torch.isfinite(x.grad).all().item()
    return fwd_nan and grad_nan, "zero input: rsqrt(ms)+eps gives NaN output; x/(sqrt(ms)+eps) gives NaN gradient"


def _decode_diff(force_causal=False, restart_positions=False):
    c = Cfg()
    torch.manual_seed(0)
    m = Model(c).eval()
    ids = torch.randint(0, c.vocab, (2, c.seq_len))
    original = F.scaled_dot_product_attention

    def sdpa(q, k, v, is_causal=False, **kw):
        return original(q, k, v, is_causal=is_causal or force_causal, **kw)

    with torch.no_grad():
        full = m(ids)
        caches = [{"k": None, "v": None} for _ in range(c.n_layers)]
        F.scaled_dot_product_attention = sdpa
        try:
            step = torch.cat([m(ids[:, t:t + 1], caches=caches,
                                pos_offset=0 if restart_positions else t)
                              for t in range(c.seq_len)], dim=1)
        finally:
            F.scaled_dot_product_attention = original
    return (full - step).abs().max().item()


def _detect_causal_mask_in_decode():
    d = _decode_diff(force_causal=True)
    return d > 1e-3, f"cache vs full forward differ by {d:.2e}"


def _detect_positions_restart_in_cache():
    d = _decode_diff(restart_positions=True)
    return d > 1e-3, f"cache vs full forward differ by {d:.2e}"


def _detect_pad_equals_eos():
    """Masking pad tokens from the loss also masks EOS when pad_id == eos_id."""
    eos, pad_distinct = 1, 0
    seq = torch.tensor([5, 6, 7, eos, eos, eos])        # response, EOS, then padding
    def supervised_eos(pad_id):
        padded = seq.clone()
        padded[4:] = pad_id
        labels = padded.masked_fill(padded == pad_id, -100)
        return int((labels == eos).sum())
    same, distinct = supervised_eos(eos), supervised_eos(pad_distinct)
    return same == 0 and distinct == 1, f"EOS labels kept: pad==eos {same}, distinct pad {distinct}"


def _detect_repeat_layout():
    wrong = torch.arange(4).float().view(1, 4, 1, 1).repeat(1, 2, 1, 1).flatten().tolist()
    return wrong != [0., 0., 1., 1., 2., 2., 3., 3.], f".repeat() layout {wrong}"


BUG_DETECTORS = {
    "rope-split-order": _detect_rope_before_split,
    "rmsnorm-eps": _detect_eps_outside_sqrt,
    "cache-causal-mask": _detect_causal_mask_in_decode,
    "cache-position-restart": _detect_positions_restart_in_cache,
    "pad-equals-eos": _detect_pad_equals_eos,
    "gqa-repeat": _detect_repeat_layout,
}


IMPL_SWAPPED: set[str] = set()


def load_impl(path: str) -> list[str]:
    """Swap this file's apply_rope / repeat_kv / RMSNorm for the user's, by name."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("user_impl", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    swapped = [n for n in ("apply_rope", "repeat_kv", "RMSNorm") if hasattr(mod, n)]
    if not swapped:
        raise SystemExit(f"{path} defines none of apply_rope, repeat_kv, RMSNorm")
    for name in swapped:
        globals()[name] = getattr(mod, name)
    IMPL_SWAPPED.update(swapped)
    return swapped


# ---------------------------------------------------------------------------


def shapes() -> int:
    c = Cfg()
    torch.manual_seed(0)
    print(f"config: d_model={c.d_model} n_heads={c.n_heads} n_kv_heads={c.n_kv_heads} "
          f"head_dim={c.head_dim} d_ff={c.d_ff} batch=2 seq={c.seq_len}\n")
    x = torch.randn(2, c.seq_len, c.d_model)
    a = Attention(c)
    b, s, _ = x.shape
    rows = [("input x", x.shape)]
    q = a.q_proj(x); k = a.k_proj(x); v = a.v_proj(x)
    rows += [("q_proj(x)", q.shape), ("k_proj(x)", k.shape), ("v_proj(x)", v.shape)]
    q = q.view(b, s, c.n_heads, c.head_dim).transpose(1, 2)
    k = k.view(b, s, c.n_kv_heads, c.head_dim).transpose(1, 2)
    v = v.view(b, s, c.n_kv_heads, c.head_dim).transpose(1, 2)
    rows += [("q split to heads", q.shape), ("k split to heads", k.shape)]
    cos, sin = rope_tables(c.head_dim, c.seq_len, c.rope_theta)
    rows += [("cos / sin table", cos.shape)]
    q, k = apply_rope(q, k, cos, sin)
    rows += [("q after RoPE", q.shape), ("k after RoPE (v untouched)", v.shape)]
    kr, vr = repeat_kv(k, a.n_rep), repeat_kv(v, a.n_rep)
    rows += [("k after repeat_kv", kr.shape)]
    scores = q @ kr.transpose(-1, -2) / math.sqrt(c.head_dim)
    rows += [("q @ k^T / sqrt(head_dim)", scores.shape)]
    o = (scores.softmax(-1) @ vr)
    rows += [("softmax(scores) @ v", o.shape)]
    o = o.transpose(1, 2).reshape(b, s, -1)
    rows += [("merged heads", o.shape), ("o_proj(...)", a.o_proj(o).shape)]
    mlp = SwiGLU(c)
    rows += [("gate_proj(x)", mlp.gate_proj(x).shape),
             ("down_proj(...)", mlp(x).shape)]
    w = max(len(r[0]) for r in rows)
    for name, shp in rows:
        print(f"  {name:<{w}}  {tuple(shp)}")
    print("\nThe two reshapes that matter: split heads BEFORE RoPE, and expand "
          "KV heads AFTER RoPE.")
    return 0


BUGS = {
    "rope-split-order": """\
Applying RoPE to the flat (batch, seq, n_heads*head_dim) tensor instead of
after splitting into heads. The rotation then pairs dimension i with
dimension i + n_heads*head_dim/2, which straddles head boundaries: head 0's
first half gets rotated against head 2's first half. Shapes are unchanged and
the model trains, but scores inside a head no longer depend only on relative
distance (--verify's bug detector shows it). The quality cost has not been
measured here, so do not quote a number for it.""",
    "rope-halves-vs-interleaved": """\
Building the cos/sin table for interleaved pairs (2i, 2i+1) but applying it
with the halves convention (i, i + head_dim/2), or vice versa. Both are valid
rotations, so training from scratch works either way and nothing errors. It
only shows up when you load someone else's checkpoint, or export yours: the
weights are silently incompatible.""",
    "gqa-repeat": """\
Using x.repeat(1, n_rep, 1, 1) instead of expand-then-reshape. Both produce
(batch, n_heads, seq, head_dim). repeat() tiles the whole KV set, so query
head i reads KV head i % n_kv; the correct layout has query head j read KV
head j // n_rep.

Be precise about the cost. Those differ by a permutation of query heads, and
query heads are independent and symmetric at init, so from-scratch training is
unharmed: --verify shows that permuting the head blocks of W_q and the
matching column blocks of W_o makes the two paths agree to 1.8e-07. What it
breaks is every boundary that assumes the contiguous layout: exported and
imported checkpoints, enable_gqa=True, fused GQA kernels. It also materialises
the expanded K and V, which expand() does not, wasting activation memory
proportional to n_rep. Fix it, but do not report it as a quality regression.""",
    "cache-causal-mask": """\
Leaving is_causal=True during incremental decode. With a length-1 query
against a length-t key, a causal mask on that 1-by-t rectangle allows only the
first key position, so the model attends to token 0 and nothing else. Prefill
looks perfect because there q and k have equal length; generation degrades
into nonsense after the first token.""",
    "rmsnorm-eps": """\
Two ways to put eps outside the sqrt, with different symptoms on an all-zero
row (padding positions, a dead channel, a zero-initialized branch):
  x * rsqrt(mean(x^2)) + eps    eps is an output offset; the forward pass is NaN.
  x / (sqrt(mean(x^2)) + eps)   forward is finite (0/eps = 0) and within ~1e-6 of
                                the correct value elsewhere, but the gradient is NaN,
                                because d sqrt(u)/du is infinite at u = 0.
The second passes forward-only tests; --verify checks the gradient too.

The related bf16 issue is real but smaller than usually claimed. PyTorch
already accumulates bf16 reductions in fp32, so accumulation precision is not
the problem. Squaring and the bf16 output are: the bf16 path's relative error
on mean(x^2) averages 1.5e-3 and peaks near 4e-3, zero-mean noise of up to
~0.2% in the norm scale at each of the two norm sites per layer. An fp32 upcast
removes it.""",
    "tying-without-scaling": """\
Tying the embedding to the unembedding without checking the init scale. The
tied matrix has to serve as both a lookup table (rows are token identities)
and a projection (columns are logit directions). Press & Wolf 2016 report the
tied matrix behaves like the *output* embedding, so initialise it at
output-projection scale (std 0.02 here) rather than at a unit-variance
embedding scale, which makes the initial logits large.""",
    "tying-init-order": """\
Tying in __init__ and then calling self.apply(init_fn). The shared tensor is
visited twice and drawn twice; whichever module apply() reaches last wins, and
module order follows the order you wrote the attribute assignments in. So the
effective init scale of your largest tensor depends on line ordering. Tie
after apply(), or assert the tie held:
    assert model.lm_head.weight.data_ptr() == model.embed.weight.data_ptr()""",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true", help="run the property tests")
    ap.add_argument("--shapes", action="store_true", help="print every tensor shape")
    ap.add_argument("--bug", choices=sorted(BUGS), help="explain one classic bug")
    ap.add_argument("--list-bugs", action="store_true")
    ap.add_argument("--impl", metavar="PATH",
                    help="python file defining apply_rope / repeat_kv / RMSNorm to test instead")
    a = ap.parse_args()
    if a.impl:
        print(f"testing your {', '.join(load_impl(a.impl))} from {a.impl}\n")
    if a.list_bugs:
        for k in sorted(BUGS):
            print(f"  {k}")
        return 0
    if a.bug:
        print(f"{a.bug}\n\n{BUGS[a.bug]}")
        return 0
    if a.shapes:
        return shapes()
    return verify()


if __name__ == "__main__":
    sys.exit(main())
