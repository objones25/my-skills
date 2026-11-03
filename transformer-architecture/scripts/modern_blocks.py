#!/usr/bin/env python3
"""
Reference implementations and property tests for the components that 2025-2026
models added on top of the dense block in reference_block.py:

    gated attention     sigmoid gate on the SDPA output, per head (Qwen3-Next, Qiu et al. 2025)
    attention sinks     a learned per-head logit in the softmax denominator (gpt-oss)
    sliding window      causal mask limited to the last W keys, with a rolling cache
    MoE routing         top-k experts, shared experts, DeepSeek-V3 bias-based balancing,
                        Switch load-balancing loss, router z-loss
    MLA                 multi-head latent attention with decoupled RoPE and the
                        absorbed decode path (DeepSeek-V2/V3)
    Muon                the update-RMS rule behind torch.optim.Muon's adjust_lr_fn (checked on
                        torch 2.15; older versions may not ship torch.optim.Muon)

    python modern_blocks.py --verify          # every property test, real output
    python modern_blocks.py --shapes          # tensor shapes on one toy example
    python modern_blocks.py --bug moe-bias-in-weights   # explain one bug

Toy sizes, used everywhere so shapes stay readable:

    d_model=64  n_heads=4  head_dim=16  seq_len=8  batch=2
    MoE: n_experts=8  top_k=2  n_shared=1  expert d_ff=32
    MLA: kv_lora_rank=24  q nope/rope dims 16/8  v_head_dim=16

Each shape comment is (batch, ...) with those numbers substituted.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

B, T, D, H, HD = 2, 8, 64, 4, 16


def rope(x: torch.Tensor, pos: torch.Tensor, theta: float = 10000.0) -> torch.Tensor:
    """Halves-pairing RoPE on the last dim of x (..., T, dim) at integer positions pos (T,)."""
    dim = x.shape[-1]
    inv = 1.0 / theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)      # (dim/2,)
    ang = pos.float()[:, None] * inv[None, :]                                       # (T, dim/2)
    cos, sin = torch.cat([ang, ang], -1).cos(), torch.cat([ang, ang], -1).sin()     # (T, dim)
    x1, x2 = x[..., : dim // 2], x[..., dim // 2:]
    return x * cos + torch.cat([-x2, x1], -1) * sin


def causal_mask(t_q: int, t_k: int, window: int | None = None, offset: int = 0) -> torch.Tensor:
    """True where query i (absolute position offset+i) may attend key j.
    Causal: j <= i. Sliding window W: also i - j < W, so each query sees at most W keys
    including itself (Mistral's convention; check yours, implementations differ by one)."""
    qi = torch.arange(t_q)[:, None] + offset
    kj = torch.arange(t_k)[None, :]
    ok = kj <= qi
    if window is not None:
        ok &= (qi - kj) < window
    return ok


# ---------------------------------------------------------------------------
# Attention with optional output gate and sinks
# ---------------------------------------------------------------------------


class Attention(nn.Module):
    """
    Plain multi-head attention plus the two 2025 additions, both off by default.

    gate="elementwise": Y <- Y * sigmoid(X W_g), W_g (d_model, n_heads*head_dim). Qiu et al.'s
        best variant: gate after SDPA, computed from the layer input X, so it depends on the
        query token and can push a head's output toward zero.
    gate="headwise": one sigmoid scalar per head, W_g (d_model, n_heads). Cheaper, close in
        quality in their tables.
    sinks=True: learned s_h per head, p_ij = exp(l_ij) / (exp(s_h) + sum_k exp(l_ik)).
        Rows sum to less than 1, so a head can put mass "nowhere" instead of on token 0.
    """

    def __init__(self, gate: str | None = None, sinks: bool = False, window: int | None = None):
        super().__init__()
        self.q, self.k, self.v, self.o = (nn.Linear(D, H * HD, bias=False) for _ in range(4))
        self.gate_mode = gate
        if gate == "elementwise":
            self.g = nn.Linear(D, H * HD, bias=False)
        elif gate == "headwise":
            self.g = nn.Linear(D, H, bias=False)
        self.sinks = nn.Parameter(torch.zeros(H)) if sinks else None
        self.window = window

    def probs(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.q(x).view(b, t, H, HD).transpose(1, 2)                  # (2, 4, 8, 16)
        k = self.k(x).view(b, t, H, HD).transpose(1, 2)                  # (2, 4, 8, 16)
        pos = torch.arange(t)
        q, k = rope(q, pos), rope(k, pos)
        logits = q @ k.transpose(-1, -2) / math.sqrt(HD)                 # (2, 4, 8, 8)
        logits = logits.masked_fill(~causal_mask(t, t, self.window), float("-inf"))
        if self.sinks is None:
            return logits.softmax(-1)
        sink = self.sinks.view(1, H, 1, 1).expand(b, H, t, 1)           # (2, 4, 8, 1)
        full = torch.cat([logits, sink], -1).softmax(-1)                 # (2, 4, 8, 9)
        return full[..., :-1]                                            # drop the sink column

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        v = self.v(x).view(b, t, H, HD).transpose(1, 2)                  # (2, 4, 8, 16)
        y = (self.probs(x) @ v).transpose(1, 2)                          # (2, 8, 4, 16)
        if self.gate_mode == "elementwise":
            y = y * torch.sigmoid(self.g(x)).view(b, t, H, HD)
        elif self.gate_mode == "headwise":
            y = y * torch.sigmoid(self.g(x)).view(b, t, H, 1)
        return self.o(y.reshape(b, t, H * HD))                           # (2, 8, 64)


def sliding_window_decode(attn: Attention, x: torch.Tensor) -> torch.Tensor:
    """Token-by-token decode keeping only the last `window` keys and values (a rolling cache)."""
    b, t, _ = x.shape
    w = attn.window
    ks, vs, outs = [], [], []
    for i in range(t):
        xi = x[:, i:i + 1]                                               # (2, 1, 64)
        pos = torch.tensor([i])
        q = rope(attn.q(xi).view(b, 1, H, HD).transpose(1, 2), pos)      # (2, 4, 1, 16)
        ks.append(rope(attn.k(xi).view(b, 1, H, HD).transpose(1, 2), pos))
        vs.append(attn.v(xi).view(b, 1, H, HD).transpose(1, 2))
        ks, vs = ks[-w:], vs[-w:]                                        # evict beyond the window
        k, v = torch.cat(ks, 2), torch.cat(vs, 2)                        # (2, 4, <=w, 16)
        p = (q @ k.transpose(-1, -2) / math.sqrt(HD)).softmax(-1)        # every cached key is visible
        outs.append(attn.o((p @ v).transpose(1, 2).reshape(b, 1, H * HD)))
    return torch.cat(outs, 1)


# ---------------------------------------------------------------------------
# Mixture of experts
# ---------------------------------------------------------------------------


@dataclass
class MoECfg:
    n_experts: int = 8
    top_k: int = 2
    n_shared: int = 1
    d_ff: int = 32
    scoring: str = "sigmoid"        # "sigmoid" (DeepSeek-V3) or "softmax" (Mixtral, Qwen3, V2)
    norm_topk: bool = True          # renormalize the selected weights to sum to 1
    bias_update: float = 1e-3       # gamma in DeepSeek-V3's aux-loss-free balancing


class Expert(nn.Module):
    def __init__(self, d_ff: int):
        super().__init__()
        self.gate, self.up = nn.Linear(D, d_ff, bias=False), nn.Linear(D, d_ff, bias=False)
        self.down = nn.Linear(d_ff, D, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class MoE(nn.Module):
    """
    Routed + shared experts.

    scores s = sigmoid(x W_r) or softmax(x W_r)                       (tokens, n_experts)
    selection: top_k of (s + b), where b is a per-expert bias that is NOT trained by gradient
    weights:   g = s at the selected experts, divided by their sum if norm_topk
    output:    sum_selected g_i E_i(x) + sum_shared E_s(x)

    DeepSeek-V3 uses b only to choose experts; the weights come from the unbiased s. After
    each step, b_i -= gamma for overloaded experts and += gamma for underloaded ones. That is
    `update_bias`. b is a buffer, which is how Hugging Face stores it too
    (`e_score_correction_bias`), so optimizers and weight decay never touch it.
    """

    def __init__(self, c: MoECfg = MoECfg()):
        super().__init__()
        self.c = c
        self.router = nn.Linear(D, c.n_experts, bias=False)
        self.experts = nn.ModuleList(Expert(c.d_ff) for _ in range(c.n_experts))
        self.shared = nn.ModuleList(Expert(c.d_ff) for _ in range(c.n_shared))
        self.register_buffer("bias", torch.zeros(c.n_experts))

    def route(self, x: torch.Tensor, bias_in_weights: bool = False):
        logits = self.router(x)                                          # (tokens, 8)
        s = torch.sigmoid(logits) if self.c.scoring == "sigmoid" else logits.softmax(-1)
        idx = (s + self.bias).topk(self.c.top_k, dim=-1).indices         # (tokens, 2)
        g = (s + self.bias if bias_in_weights else s).gather(-1, idx)    # (tokens, 2)
        if self.c.norm_topk:
            g = g / g.sum(-1, keepdim=True)
        return logits, s, idx, g

    def forward(self, x: torch.Tensor, bias_in_weights: bool = False) -> torch.Tensor:
        shape = x.shape
        x = x.reshape(-1, D)                                             # (16, 64)
        _, _, idx, g = self.route(x, bias_in_weights)
        out = torch.zeros_like(x)
        for e, expert in enumerate(self.experts):                        # dispatch by expert
            tok, slot = (idx == e).nonzero(as_tuple=True)
            if tok.numel():
                out.index_add_(0, tok, g[tok, slot, None] * expert(x[tok]))
        for expert in self.shared:
            out = out + expert(x)
        return out.view(shape)

    def forward_dense(self, x: torch.Tensor) -> torch.Tensor:
        """Every expert on every token, masked. Slow, obviously correct; the oracle for forward()."""
        shape = x.shape
        x = x.reshape(-1, D)
        _, _, idx, g = self.route(x)
        w = torch.zeros(x.shape[0], self.c.n_experts).scatter(-1, idx, g)  # (16, 8)
        y = torch.stack([e(x) for e in self.experts], 1)                   # (16, 8, 64)
        out = (w[..., None] * y).sum(1) + sum(e(x) for e in self.shared)
        return out.view(shape)

    def load(self, x: torch.Tensor) -> torch.Tensor:
        """Fraction of routed slots each expert received."""
        _, _, idx, _ = self.route(x.reshape(-1, D))
        return torch.bincount(idx.flatten(), minlength=self.c.n_experts).float() / idx.numel()

    @torch.no_grad()
    def update_bias(self, x: torch.Tensor) -> None:
        load = self.load(x)
        self.bias += self.c.bias_update * torch.sign(load.mean() - load)


def switch_balance_loss(logits: torch.Tensor, idx: torch.Tensor, n_experts: int) -> torch.Tensor:
    """Switch Transformer load-balancing loss N * sum_i f_i * P_i, f_i = fraction of slots
    routed to i, P_i = mean router probability for i. Equals 1 at perfect balance."""
    f = torch.bincount(idx.flatten(), minlength=n_experts).float() / idx.numel()
    p = logits.softmax(-1).mean(0)
    return n_experts * (f * p).sum()


def router_z_loss(logits: torch.Tensor) -> torch.Tensor:
    """ST-MoE router z-loss: mean over tokens of logsumexp(logits)^2."""
    return torch.logsumexp(logits, -1).pow(2).mean()


# ---------------------------------------------------------------------------
# Multi-head latent attention
# ---------------------------------------------------------------------------

KV_LORA, NOPE, ROPE_D, V_D = 24, 16, 8, 16


class MLA(nn.Module):
    """
    DeepSeek-V2 MLA without the query low-rank path (V2-Lite style).

    c_kv   = W_dkv x                    (latent, KV_LORA wide)   cached
    k_rope = RoPE(W_kr x)               (ROPE_D wide, one per token, shared by heads)   cached
    k_nope = W_uk c_kv,  v = W_uv c_kv  (per head)               recomputable
    q      = W_q x -> [q_nope | q_rope],  q_rope gets RoPE
    score  = q_nope . k_nope + q_rope . k_rope

    RoPE cannot touch k_nope: the absorbed path rewrites q_nope . (W_uk c) as
    (W_uk^T q_nope) . c, which only works if nothing position-dependent sits between
    W_uk and c. That is why the RoPE part is a separate, decoupled key.
    """

    def __init__(self):
        super().__init__()
        self.w_q = nn.Linear(D, H * (NOPE + ROPE_D), bias=False)
        self.w_dkv = nn.Linear(D, KV_LORA, bias=False)
        self.w_kr = nn.Linear(D, ROPE_D, bias=False)
        self.w_uk = nn.Linear(KV_LORA, H * NOPE, bias=False)
        self.w_uv = nn.Linear(KV_LORA, H * V_D, bias=False)
        self.o = nn.Linear(H * V_D, D, bias=False)

    def _q(self, x, pos):
        b, t, _ = x.shape
        q = self.w_q(x).view(b, t, H, NOPE + ROPE_D).transpose(1, 2)     # (2, 4, 8, 24)
        return q[..., :NOPE], rope(q[..., NOPE:], pos)                   # (2,4,8,16), (2,4,8,8)

    def cache_entries(self, x, pos):
        return self.w_dkv(x), rope(self.w_kr(x), pos)                    # (2,8,24), (2,8,8)

    def forward(self, x: torch.Tensor, rope_on_latent: bool = False) -> torch.Tensor:
        """Explicit path: expand per-head keys and values from the latent."""
        b, t, _ = x.shape
        pos = torch.arange(t)
        q_nope, q_rope = self._q(x, pos)
        c, k_rope = self.cache_entries(x, pos)
        k_nope = self.w_uk(c).view(b, t, H, NOPE).transpose(1, 2)        # (2, 4, 8, 16)
        if rope_on_latent:                                               # the bug, for the detector
            k_nope = rope(k_nope, pos)
        v = self.w_uv(c).view(b, t, H, V_D).transpose(1, 2)              # (2, 4, 8, 16)
        scores = q_nope @ k_nope.transpose(-1, -2) + q_rope @ k_rope[:, None].transpose(-1, -2)
        scores = scores / math.sqrt(NOPE + ROPE_D)
        scores = scores.masked_fill(~causal_mask(t, t), float("-inf"))
        y = (scores.softmax(-1) @ v).transpose(1, 2).reshape(b, t, H * V_D)
        return self.o(y)

    def absorbed_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Scores computed from the cache alone: q_nope is pulled through W_uk^T."""
        b, t, _ = x.shape
        pos = torch.arange(t)
        q_nope, q_rope = self._q(x, pos)
        c, k_rope = self.cache_entries(x, pos)
        w_uk = self.w_uk.weight.view(H, NOPE, KV_LORA)                   # (4, 16, 24)
        q_lat = torch.einsum("bhtn,hnc->bhtc", q_nope, w_uk)             # (2, 4, 8, 24)
        scores = q_lat @ c[:, None].transpose(-1, -2) + q_rope @ k_rope[:, None].transpose(-1, -2)
        return scores / math.sqrt(NOPE + ROPE_D)

    def explicit_scores(self, x: torch.Tensor, rope_on_latent: bool = False) -> torch.Tensor:
        b, t, _ = x.shape
        pos = torch.arange(t)
        q_nope, q_rope = self._q(x, pos)
        c, k_rope = self.cache_entries(x, pos)
        k_nope = self.w_uk(c).view(b, t, H, NOPE).transpose(1, 2)
        if rope_on_latent:
            k_nope = rope(k_nope, pos)
        return (q_nope @ k_nope.transpose(-1, -2) + q_rope @ k_rope[:, None].transpose(-1, -2)) \
            / math.sqrt(NOPE + ROPE_D)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Token by token, caching only (c_kv, k_rope): KV_LORA + ROPE_D numbers per token."""
        b, t, _ = x.shape
        cs, krs, outs = [], [], []
        w_uk = self.w_uk.weight.view(H, NOPE, KV_LORA)
        w_uv = self.w_uv.weight.view(H, V_D, KV_LORA)
        for i in range(t):
            xi, pos = x[:, i:i + 1], torch.tensor([i])
            q_nope, q_rope = self._q(xi, pos)                            # (2,4,1,16), (2,4,1,8)
            c, kr = self.cache_entries(xi, pos)
            cs.append(c)
            krs.append(kr)
            C, KR = torch.cat(cs, 1), torch.cat(krs, 1)                  # (2, i+1, 24), (2, i+1, 8)
            q_lat = torch.einsum("bhtn,hnc->bhtc", q_nope, w_uk)
            s = (q_lat @ C[:, None].transpose(-1, -2) + q_rope @ KR[:, None].transpose(-1, -2))
            p = (s / math.sqrt(NOPE + ROPE_D)).softmax(-1)               # (2, 4, 1, i+1)
            lat = p @ C[:, None]                                         # (2, 4, 1, 24) weighted latent
            y = torch.einsum("bhtc,hvc->bhtv", lat, w_uv)                # (2, 4, 1, 16) absorb W_uv
            outs.append(self.o(y.transpose(1, 2).reshape(b, 1, H * V_D)))
        return torch.cat(outs, 1)


# ---------------------------------------------------------------------------
# Muon update scale
# ---------------------------------------------------------------------------


def muon_update(shape: tuple[int, int], adjust_lr_fn: str | None, seed: int = 0) -> torch.Tensor:
    """One torch.optim.Muon step from zero momentum on a random gradient, lr=1, no weight
    decay, no Nesterov. Returns the update, which is adjust(shape) * NS(g)."""
    torch.manual_seed(seed)
    p = nn.Parameter(torch.zeros(shape))
    opt = torch.optim.Muon([p], lr=1.0, weight_decay=0.0, momentum=0.0, nesterov=False,
                           adjust_lr_fn=adjust_lr_fn)
    p.grad = torch.randn(shape)
    opt.step()
    return -p.detach()


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

RESULTS: list[bool] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def verify() -> int:
    RESULTS.clear()
    torch.manual_seed(0)
    x = torch.randn(B, T, D)

    print("\nGated attention")
    plain = Attention()
    for mode, extra in (("elementwise", D * H * HD), ("headwise", D * H)):
        gated = Attention(gate=mode)
        n = sum(p.numel() for p in gated.parameters()) - sum(p.numel() for p in plain.parameters())
        check(f"{mode} gate adds {extra:,} parameters", n == extra, f"{n:,}")
    gated = Attention(gate="headwise")
    gated.load_state_dict(plain.state_dict(), strict=False)
    nn.init.zeros_(gated.g.weight)
    diff = (gated(x) - 0.5 * plain(x)).abs().max().item()
    check("zero gate weights halve the output (sigmoid(0) = 0.5)", diff < 1e-6, f"max diff {diff:.1e}")
    xs = x.clone()
    xs[..., 0] = -1e4                                                    # one input feature, very negative
    with torch.no_grad():
        gated.g.weight[0, 0] = 1.0                                       # head 0's gate reads it: sigmoid -> 0
        v = gated.v(xs).view(B, T, H, HD).transpose(1, 2)
        y = (gated.probs(xs) @ v).transpose(1, 2)
        gate = torch.sigmoid(gated.g(xs)).view(B, T, H, 1)
    check("a head can abstain: gate -> 0 zeroes its output", (y * gate)[:, :, 0].abs().max() < 1e-6)
    ungated_row_sum = plain.probs(x).sum(-1)
    check("an ungated head cannot: softmax rows always sum to 1",
          torch.allclose(ungated_row_sum, torch.ones_like(ungated_row_sum)))

    print("\nAttention sinks")
    sink = Attention(sinks=True)
    sink.load_state_dict(plain.state_dict(), strict=False)
    with torch.no_grad():
        sink.sinks.fill_(-1e4)
    d = (sink(x) - plain(x)).abs().max().item()
    check("sink logit -> -inf recovers plain attention", d < 1e-6, f"max diff {d:.1e}")
    with torch.no_grad():
        sink.sinks.fill_(2.0)
    rows = sink.probs(x).sum(-1)
    check("with a sink, attention rows sum to less than 1", bool((rows < 1).all()),
          f"row sums {rows.min():.3f}..{rows.max():.3f}")
    with torch.no_grad():
        logits = torch.tensor([1.0, 0.5, -0.3])
        s = 2.0
        want = logits.exp() / (math.exp(s) + logits.exp().sum())
        got = torch.cat([logits, torch.tensor([s])]).softmax(-1)[:-1]
    check("sink column == exp(l) / (exp(s) + sum exp(l))", torch.allclose(got, want))
    adds_to_logits = (logits + s).softmax(-1)
    check("adding s to every logit instead does nothing (softmax is shift-invariant)",
          torch.allclose(adds_to_logits, logits.softmax(-1)))

    print("\nSliding window")
    win = Attention(window=3)
    xp = x.clone()
    xp[:, 0] += 5.0                                                       # perturb token 0
    out, outp = win(x), win(xp)
    check("token 0 cannot affect position 5 with window 3", torch.allclose(out[:, 5], outp[:, 5]))
    check("token 0 still affects position 2 (inside the window)",
          not torch.allclose(out[:, 2], outp[:, 2]))
    d = (sliding_window_decode(win, x) - out).abs().max().item()
    check("rolling cache of 3 entries == full forward with the window mask", d < 1e-5,
          f"max diff {d:.1e}")
    check("each query sees at most W keys", int(causal_mask(T, T, 3).sum(-1).max()) == 3)

    print("\nMixture of experts")
    torch.manual_seed(1)
    moe = MoE()
    xt = torch.randn(64, D)
    d = (moe(xt) - moe.forward_dense(xt)).abs().max().item()
    check("sparse dispatch == dense masked oracle", d < 1e-5, f"max diff {d:.1e}")
    _, _, idx, g = moe.route(xt)
    check("selected weights sum to 1 when norm_topk", torch.allclose(g.sum(-1), torch.ones(64)))
    logits = moe.router(xt)
    top = logits.topk(2, -1)
    a = top.values.softmax(-1)
    full = logits.softmax(-1).gather(-1, top.indices)
    check("softmax over the top-k logits == renormalized top-k of the full softmax",
          torch.allclose(a, full / full.sum(-1, keepdim=True), atol=1e-6))
    with torch.no_grad():
        moe.bias.copy_(torch.linspace(-0.5, 0.5, 8))
    _, s, idx_b, g_b = moe.route(xt)
    _, _, _, g_bug = moe.route(xt, bias_in_weights=True)
    check("bias changes which experts are chosen", not torch.equal(idx_b, idx))
    want = s.gather(-1, idx_b)
    want = want / want.sum(-1, keepdim=True)
    check("gate weights come from unbiased scores", torch.allclose(g_b, want))
    check("putting the bias in the weights changes the output (detector)",
          not torch.allclose(g_bug, g_b))
    with torch.no_grad():
        moe.bias.zero_()
        moe.router.weight[0] += 0.8 * moe.router.weight.std() * torch.sign(xt.mean(0))
        moe.router.weight[0] += 2.0                                       # expert 0 hogs tokens
    start = moe.load(xt).max().item()
    for _ in range(400):
        moe.update_bias(xt)
    end = moe.load(xt).max().item()
    check("bias updates rebalance a skewed router (no aux loss)", end < start and end < 0.25 + 0.1,
          f"max expert load {start:.2f} -> {end:.2f} (uniform = {1 / 8:.3f})")
    uniform_idx = torch.arange(64 * 2).remainder(8).view(64, 2)
    flat = torch.zeros(64, 8)
    lb = switch_balance_loss(flat, uniform_idx, 8).item()
    check("Switch balance loss is 1 at perfect balance", abs(lb - 1.0) < 1e-6, f"{lb:.4f}")
    hog = torch.zeros(64, 8)
    hog[:, 0] = 10.0                                                     # router prefers expert 0
    skew = switch_balance_loss(hog, torch.zeros(64, 2, dtype=torch.long), 8).item()
    check("and near n_experts when the router sends everything to one expert", skew > 7.5,
          f"{skew:.3f} (max {8})")
    z = router_z_loss(torch.zeros(5, 8)).item()
    check("router z-loss of zero logits = log(n_experts)^2", abs(z - math.log(8) ** 2) < 1e-5,
          f"{z:.4f} vs {math.log(8) ** 2:.4f}")

    print("\nMulti-head latent attention")
    torch.manual_seed(2)
    mla = MLA()
    d = (mla.absorbed_scores(x) - mla.explicit_scores(x)).abs().max().item()
    check("absorbed scores (from cache only) == explicit per-head scores", d < 1e-4, f"max diff {d:.1e}")
    d = (mla.decode(x) - mla(x)).abs().max().item()
    check("latent-cache decode == full forward", d < 1e-4, f"max diff {d:.1e}")
    c, kr = mla.cache_entries(x, torch.arange(T))
    per_tok = c.shape[-1] + kr.shape[-1]
    expanded = H * (NOPE + ROPE_D) + H * V_D
    check(f"cache holds {KV_LORA}+{ROPE_D} = {per_tok} numbers per token, not {expanded}",
          per_tok == KV_LORA + ROPE_D)
    try:
        import importlib.util
        import pathlib
        spec = importlib.util.spec_from_file_location(
            "config_budget", pathlib.Path(__file__).with_name("config_budget.py"))
        cb = importlib.util.module_from_spec(spec)
        sys.modules["config_budget"] = cb
        spec.loader.exec_module(cb)
        cfg = cb.Config(d_model=D, n_layers=1, n_heads=H, head_dim=V_D, vocab=128, d_ff=128,
                        kv_lora_rank=KV_LORA, qk_nope_head_dim=NOPE, qk_rope_head_dim=ROPE_D,
                        v_head_dim=V_D)
        budget_attn = cb.attention_params(cfg)["total"] - KV_LORA          # minus the latent norm
        n = sum(p.numel() for p in mla.parameters())
        check("parameter count matches config_budget's MLA formula (latent norm aside)",
              n == budget_attn, f"{n:,} vs {budget_attn:,}")
    except Exception as e:                                                # noqa: BLE001
        check("parameter count matches config_budget's MLA formula", False, repr(e))

    print("\nMuon update scale (torch.optim.Muon)")
    for shape in ((256, 64), (64, 256)):
        u = muon_update(shape, "match_rms_adamw")
        rms = u.pow(2).mean().sqrt().item()
        check(f"match_rms_adamw scales the update to RMS ~0.2 regardless of shape {shape}",
              0.15 < rms < 0.22, f"RMS {rms:.3f}; below 0.2 because Newton-Schulz singular "
              "values are not exactly 1")
    u = muon_update((256, 64), None)
    sv = torch.linalg.svdvals(u / math.sqrt(256 / 64))
    check("Newton-Schulz output is near-orthogonal (singular values ~1)",
          bool(((sv > 0.5) & (sv < 1.5)).all()), f"singular values {sv.min():.2f}..{sv.max():.2f}")

    print("\nBug detectors (each must catch its bug)")
    for name, fn in BUG_DETECTORS.items():
        caught, detail = fn()
        check(f"{name}: caught", caught, detail)

    failed = RESULTS.count(False)
    print(f"\n{'all checks passed' if not failed else f'{failed} checks FAILED'}")
    return int(failed > 0)


def _detect_moe_bias_in_weights():
    torch.manual_seed(3)
    m = MoE()
    with torch.no_grad():
        m.bias.copy_(torch.linspace(-0.3, 0.3, 8))
    xt = torch.randn(32, D)
    ok, bug = m.route(xt)[3], m.route(xt, bias_in_weights=True)[3]
    d = (ok - bug).abs().max().item()
    return d > 1e-3, f"gate weights differ by up to {d:.3f}"


def _detect_mla_rope_on_latent():
    torch.manual_seed(4)
    m = MLA()
    x = torch.randn(B, T, D)
    d = (m.absorbed_scores(x) - m.explicit_scores(x, rope_on_latent=True)).abs().max().item()
    return d > 1e-3, f"absorbed and explicit scores differ by {d:.3f}"


def _detect_topk_renorm_mismatch():
    torch.manual_seed(5)
    a, b = MoE(MoECfg(scoring="softmax", norm_topk=True)), MoE(MoECfg(scoring="softmax", norm_topk=False))
    b.load_state_dict(a.state_dict())
    x = torch.randn(32, D)
    ratio = (a(x) - sum(e(x) for e in a.shared)).norm() / (b(x) - sum(e(x) for e in b.shared)).norm()
    return ratio.item() > 1.5, f"routed output norm ratio {ratio.item():.2f}x"


def _detect_sink_added_to_logits():
    logits = torch.randn(4, 6)
    proper = torch.cat([logits, torch.full((4, 1), 1.5)], -1).softmax(-1)[:, :-1]
    wrong = (logits + 1.5).softmax(-1)
    d = (proper - wrong).abs().max().item()
    return d > 1e-3, f"probabilities differ by up to {d:.3f}"


BUG_DETECTORS = {
    "moe-bias-in-weights": _detect_moe_bias_in_weights,
    "mla-rope-on-latent": _detect_mla_rope_on_latent,
    "moe-topk-renorm-mismatch": _detect_topk_renorm_mismatch,
    "sink-added-to-logits": _detect_sink_added_to_logits,
}

BUGS = {
    "moe-bias-in-weights": """
DeepSeek-V3's balancing bias b is added to the scores only to pick the top-k experts. The
weights that scale each expert's output come from the unbiased scores. Adding b to the weights
as well couples load balancing to the model's output: every bias update now changes what the
layer computes, and the bias starts acting like an untrained parameter in the forward pass.
Training still runs, so nothing crashes. Check: route() with and without the bias in the
weights must give the same weights for the same selection.""",
    "mla-rope-on-latent": """
MLA caches only the latent c_KV and recomputes keys as W_uk c_KV. At decode, the absorbed path
moves W_uk onto the query side: q . (W_uk c) = (W_uk^T q) . c. RoPE is a position-dependent
rotation R_t, and q . (R_t W_uk c) cannot be rewritten that way, so applying RoPE to the
latent-derived keys breaks absorption and forces you to cache full per-head keys (losing the
memory win). DeepSeek's fix is a separate small RoPE key shared across heads. Symptom: the
absorbed decode disagrees with the explicit forward.""",
    "moe-topk-renorm-mismatch": """
With softmax scoring, top-k weights either stay as raw probabilities (sum < 1; DeepSeek-V2,
OLMoE: norm_topk_prob false) or are renormalized to sum to 1 (Mixtral, Qwen3: true). Both are
legitimate, but a checkpoint trained one way and run the other way scales every routed output
by a different factor per token. From scratch it only shifts the learned scale; on a pretrained
checkpoint it silently degrades quality. Read norm_topk_prob from the config.""",
    "sink-added-to-logits": """
A learned attention sink is an extra column in the softmax denominator: exp(s) / (exp(s) +
sum exp(l)) of the mass goes nowhere. Adding s to every logit instead does nothing at all,
because softmax is invariant to a constant shift. The model trains, the parameter gets
gradients of zero, and the rows still sum to 1. Check: with a large s the attention output
should shrink toward zero.""",
}


def shapes() -> int:
    torch.manual_seed(0)
    x = torch.randn(B, T, D)
    print(f"x {tuple(x.shape)}")
    a = Attention(gate="elementwise", sinks=True)
    print(f"gated attention: gate sigmoid(x W_g) {tuple(a.g(x).shape)} -> view (B, T, H, HD) "
          f"{(B, T, H, HD)}; sinks {tuple(a.sinks.shape)}; probs {tuple(a.probs(x).shape)}; out {tuple(a(x).shape)}")
    m = MoE()
    logits, s, idx, g = m.route(x.reshape(-1, D))
    print(f"MoE: tokens {(B * T, D)}, router logits {tuple(logits.shape)}, chosen {tuple(idx.shape)}, "
          f"weights {tuple(g.shape)}, bias buffer {tuple(m.bias.shape)}, out {tuple(m(x).shape)}")
    mla = MLA()
    c, kr = mla.cache_entries(x, torch.arange(T))
    print(f"MLA: c_kv {tuple(c.shape)}, k_rope {tuple(kr.shape)} (cached); "
          f"absorbed scores {tuple(mla.absorbed_scores(x).shape)}; out {tuple(mla(x).shape)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--shapes", action="store_true")
    ap.add_argument("--bug", choices=sorted(BUGS))
    a = ap.parse_args()
    if a.bug:
        print(BUGS[a.bug].strip())
        return 0
    if a.shapes:
        return shapes()
    return verify()


if __name__ == "__main__":
    sys.exit(main())
