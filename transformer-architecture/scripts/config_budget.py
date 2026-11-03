#!/usr/bin/env python3
"""
Parameter, FLOP, KV-cache and memory budget for a decoder-only transformer,
plus a validator that flags configs that deviate from shipping practice.

Counts are exact for the modern stack: RMSNorm + RoPE + GQA + gated MLP, plus
mixture-of-experts layers (routed and shared experts, router, leading dense
layers), multi-head latent attention (MLA), per-head or full-width QK-norm,
learned attention sinks, and sliding-window layers (for the KV cache). The
self-test checks every preset against the exact total in the published
checkpoint (safetensors index, cross-checked by instantiating the config on
PyTorch's meta device with transformers):

    python config_budget.py --self-test

Usage:
    python config_budget.py --preset llama3-8b
    python config_budget.py --d-model 1024 --n-layers 24 --n-heads 16 \
        --n-kv-heads 4 --d-ff 2752 --vocab 32000 --seq-len 2048 --tie
    python config_budget.py --from-hf-config path/to/config.json
    python config_budget.py --preset qwen3-30b-a3b        # MoE: total and active
    python config_budget.py --preset llama3-8b --json     # machine-readable

Not modeled (refused by --from-hf-config with a message): linear-attention or
state-space hybrids (Gated DeltaNet, Mamba), Qwen2-MoE's gated shared expert,
MoE layers on a stride other than every layer, and multi-token-prediction heads
(DeepSeek-V3's MTP module is excluded, as Hugging Face's implementation does).

Every number is derived, not looked up. `--show-work` prints the arithmetic
for the parameter count so you can check it by hand.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import dataclasses
from dataclasses import dataclass, asdict

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class Config:
    d_model: int
    n_layers: int
    n_heads: int
    vocab: int
    d_ff: int
    n_kv_heads: int | None = None      # None -> MHA (n_kv_heads = n_heads)
    head_dim: int | None = None        # None -> d_model // n_heads
    seq_len: int = 2048                # context length used for FLOP/KV math
    tie_embeddings: bool = False
    gated_mlp: bool = True             # SwiGLU/GeGLU (3 matrices) vs. plain (2)
    norm: str = "rmsnorm"              # "rmsnorm" | "layernorm"
    qk_norm: bool = False
    qk_norm_scope: str = "head"        # "head": one head_dim gain each for Q and K (Qwen3, Gemma 3)
                                       # "full": gains over the whole Q and K projections (OLMo 2, OLMoE)
    attn_bias: str = "none"            # "none" | "qkv" (Qwen2 style) | "all" (GPT-2, gpt-oss)
    mlp_bias: bool = False
    rope_theta: float | None = 10000.0  # None -> learned absolute positions
    norms_per_layer: int = 2           # 2 for pre-norm or OLMo 2 post-norm; 4 for Gemma 2/3 sandwich norms
    attn_sinks: bool = False           # learned per-head sink logit (gpt-oss): n_heads params per layer
    # Mixture of experts. n_experts = 0 means every layer has a dense MLP of width d_ff.
    n_experts: int = 0                 # routed experts per MoE layer
    top_k: int = 0                     # routed experts each token uses
    n_shared_experts: int = 0          # always-on experts (DeepSeekMoE), each moe_d_ff wide
    moe_d_ff: int | None = None        # hidden width of one expert; None -> d_ff
    first_k_dense: int = 0             # leading layers that keep a dense MLP of width d_ff
    router_bias: bool = False
    expert_bias: bool = False          # biases inside each expert (gpt-oss)
    # Multi-head latent attention (DeepSeek-V2/V3). kv_lora_rank = 0 means standard attention.
    kv_lora_rank: int = 0              # width of the cached latent c_KV
    q_lora_rank: int | None = None     # None -> queries projected directly from d_model
    qk_nope_head_dim: int = 0          # per-head query/key width without RoPE
    qk_rope_head_dim: int = 0          # per-head width that carries RoPE (key part shared by all heads)
    v_head_dim: int = 0
    # Sliding-window attention: n_sliding_layers layers only ever cache sliding_window tokens.
    sliding_window: int | None = None
    n_sliding_layers: int = 0
    name: str = "custom"

    @property
    def is_moe(self) -> bool:
        return self.n_experts > 0

    @property
    def is_mla(self) -> bool:
        return self.kv_lora_rank > 0

    @property
    def n_moe_layers(self) -> int:
        return self.n_layers - self.first_k_dense if self.is_moe else 0

    def __post_init__(self):
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        if self.head_dim is None:
            if self.d_model % self.n_heads:
                raise ValueError(
                    f"d_model={self.d_model} is not divisible by n_heads={self.n_heads}; "
                    "pass --head-dim explicitly if you meant to decouple them"
                )
            self.head_dim = self.d_model // self.n_heads
        if self.is_moe:
            if not 1 <= self.top_k <= self.n_experts:
                raise ValueError(f"top_k={self.top_k} must be between 1 and n_experts={self.n_experts}")
            if self.moe_d_ff is None:
                self.moe_d_ff = self.d_ff
            if not 0 <= self.first_k_dense < self.n_layers:
                raise ValueError(f"first_k_dense={self.first_k_dense} must leave at least one MoE layer")
        if self.is_mla and not (self.qk_nope_head_dim and self.qk_rope_head_dim and self.v_head_dim):
            raise ValueError("MLA needs qk_nope_head_dim, qk_rope_head_dim and v_head_dim")
        if self.n_sliding_layers and not self.sliding_window:
            raise ValueError("n_sliding_layers > 0 needs sliding_window")
        if self.qk_norm_scope not in ("head", "full"):
            raise ValueError("qk_norm_scope must be 'head' or 'full'")


# Dense presets verified against Hugging Face config.json, Aug 2026; MoE presets
# and the olmo2-7b correction verified Nov 2, 2026 against config.json, the
# safetensors index total_size, and a meta-device instantiation (transformers 5.22.1). against Hugging Face config.json, Aug 2026. See references/reference-configs.md.
PRESETS: dict[str, Config] = {
    "gpt2-124m": Config(
        name="GPT-2 124M", d_model=768, n_layers=12, n_heads=12, vocab=50257,
        d_ff=3072, seq_len=1024, tie_embeddings=True, gated_mlp=False,
        norm="layernorm", attn_bias="all", mlp_bias=True, rope_theta=None,
    ),
    "llama3-8b": Config(
        name="Llama-3.1-8B", d_model=4096, n_layers=32, n_heads=32, n_kv_heads=8,
        vocab=128256, d_ff=14336, seq_len=8192, tie_embeddings=False,
        rope_theta=500000.0,
    ),
    "llama3.2-1b": Config(
        name="Llama-3.2-1B", d_model=2048, n_layers=16, n_heads=32, n_kv_heads=8,
        vocab=128256, d_ff=8192, seq_len=8192, tie_embeddings=True,
        rope_theta=500000.0,
    ),
    "qwen2.5-0.5b": Config(
        name="Qwen2.5-0.5B", d_model=896, n_layers=24, n_heads=14, n_kv_heads=2,
        vocab=151936, d_ff=4864, seq_len=32768, tie_embeddings=True,
        attn_bias="qkv", rope_theta=1000000.0,
    ),
    "qwen3-8b": Config(
        name="Qwen3-8B", d_model=4096, n_layers=36, n_heads=32, n_kv_heads=8,
        head_dim=128, vocab=151936, d_ff=12288, seq_len=40960,
        tie_embeddings=False, qk_norm=True, rope_theta=1000000.0,
    ),
    "mistral-7b": Config(
        name="Mistral-7B-v0.1", d_model=4096, n_layers=32, n_heads=32, n_kv_heads=8,
        vocab=32000, d_ff=14336, seq_len=8192, tie_embeddings=False,
        rope_theta=10000.0,
    ),
    "olmo2-7b": Config(
        name="OLMo-2-7B", d_model=4096, n_layers=32, n_heads=32, n_kv_heads=32,
        vocab=100352, d_ff=11008, seq_len=4096, tie_embeddings=False,
        qk_norm=True, qk_norm_scope="full", rope_theta=500000.0,
    ),
    "smollm2-135m": Config(
        name="SmolLM2-135M", d_model=576, n_layers=30, n_heads=9, n_kv_heads=3,
        vocab=49152, d_ff=1536, seq_len=8192, tie_embeddings=True,
        rope_theta=100000.0,
    ),
    "gemma3-1b": Config(
        name="Gemma-3-1B", d_model=1152, n_layers=26, n_heads=4, n_kv_heads=1,
        head_dim=256, vocab=262144, d_ff=6912, seq_len=32768, tie_embeddings=True,
        qk_norm=True, rope_theta=1000000.0, norms_per_layer=4,
        sliding_window=512, n_sliding_layers=22,
    ),
    "mixtral-8x7b": Config(
        name="Mixtral-8x7B-v0.1", d_model=4096, n_layers=32, n_heads=32, n_kv_heads=8,
        vocab=32000, d_ff=14336, seq_len=32768, rope_theta=1000000.0,
        n_experts=8, top_k=2,
    ),
    "olmoe-1b-7b": Config(
        name="OLMoE-1B-7B-0924", d_model=2048, n_layers=16, n_heads=16, n_kv_heads=16,
        vocab=50304, d_ff=1024, seq_len=4096, qk_norm=True, qk_norm_scope="full",
        n_experts=64, top_k=8,
    ),
    "qwen3-30b-a3b": Config(
        name="Qwen3-30B-A3B", d_model=2048, n_layers=48, n_heads=32, n_kv_heads=4,
        head_dim=128, vocab=151936, d_ff=6144, seq_len=40960, qk_norm=True,
        rope_theta=1000000.0, n_experts=128, top_k=8, moe_d_ff=768,
    ),
    "gpt-oss-20b": Config(
        name="gpt-oss-20b", d_model=2880, n_layers=24, n_heads=64, n_kv_heads=8,
        head_dim=64, vocab=201088, d_ff=2880, seq_len=131072, attn_bias="all",
        rope_theta=150000.0, attn_sinks=True, n_experts=32, top_k=4,
        router_bias=True, expert_bias=True, sliding_window=128, n_sliding_layers=12,
    ),
    "deepseek-v2-lite": Config(
        name="DeepSeek-V2-Lite", d_model=2048, n_layers=27, n_heads=16, head_dim=128,
        vocab=102400, d_ff=10944, seq_len=32768, n_experts=64, top_k=6,
        n_shared_experts=2, moe_d_ff=1408, first_k_dense=1, kv_lora_rank=512,
        qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128,
    ),
    "deepseek-v3": Config(
        name="DeepSeek-V3 (without the MTP module)", d_model=7168, n_layers=61,
        n_heads=128, head_dim=128, vocab=129280, d_ff=18432, seq_len=163840,
        n_experts=256, top_k=8, n_shared_experts=1, moe_d_ff=2048, first_k_dense=3,
        kv_lora_rank=512, q_lora_rank=1536, qk_nope_head_dim=128,
        qk_rope_head_dim=64, v_head_dim=128,
    ),
}

# Exact totals: safetensors index total_size / bytes per parameter, and the same
# number from instantiating config.json on the meta device (transformers 5.22.1).
# deepseek-v3 is FP8 on the Hub, so its total is from the meta device only.
PUBLISHED_TOTALS: dict[str, int] = {
    "gpt2-124m": 124_439_808,
    "llama3-8b": 8_030_261_248,
    "qwen2.5-0.5b": 494_032_768,
    "smollm2-135m": 134_515_008,
    "qwen3-8b": 8_190_735_360,
    "olmo2-7b": 7_298_617_344,
    "gemma3-1b": 999_885_952,
    "mixtral-8x7b": 46_702_792_704,
    "olmoe-1b-7b": 6_919_161_856,
    "qwen3-30b-a3b": 30_532_122_624,
    "gpt-oss-20b": 20_914_757_184,
    "deepseek-v2-lite": 15_706_484_224,
    "deepseek-v3": 671_026_404_352,
}

# Active-parameter claims as published (rounded). Labs count differently: Qwen
# counts both embedding tables; gpt-oss counts the unembedding but not the input
# embedding ("Unembedding parameters are counted towards active, but not
# embeddings", gpt-oss model card Table 1). The self-test reports all three
# conventions and which one each claim matches.
PUBLISHED_ACTIVE: dict[str, float] = {
    "qwen3-30b-a3b": 3.3e9,      # model card: "30.5B in total and 3.3B activated"
    "deepseek-v2-lite": 2.4e9,   # model card: "16B total params, 2.4B active params"
    "gpt-oss-20b": 3.61e9,       # model card (arXiv 2508.10925) Table 1: "Active Parameters 3.61B"
    "deepseek-v3": 37e9,         # tech report: "671B total parameters with 37B activated"
}


# ---------------------------------------------------------------------------
# Parameter count
# ---------------------------------------------------------------------------


def attention_params(c: Config) -> dict:
    """Per-layer attention parameters, split by matrix so --show-work can print them."""
    d, hd, nh, nkv = c.d_model, c.head_dim, c.n_heads, c.n_kv_heads
    if c.is_mla:
        qk = c.qk_nope_head_dim + c.qk_rope_head_dim
        if c.q_lora_rank:
            w_q = d * c.q_lora_rank + c.q_lora_rank + c.q_lora_rank * nh * qk   # down, norm, up
        else:
            w_q = d * nh * qk
        w_kv_a = d * (c.kv_lora_rank + c.qk_rope_head_dim) + c.kv_lora_rank    # latent + shared rope key, norm
        w_kv_b = c.kv_lora_rank * nh * (c.qk_nope_head_dim + c.v_head_dim)      # latent -> per-head K_nope, V
        w_o = nh * c.v_head_dim * d
        parts = {"w_q": w_q, "w_kv_a": w_kv_a, "w_kv_b": w_kv_b, "w_o": w_o}
    else:
        q_out, kv_out = nh * hd, nkv * hd
        parts = {"w_q": d * q_out, "w_k": d * kv_out, "w_v": d * kv_out, "w_o": q_out * d}
        if c.attn_bias in ("qkv", "all"):
            parts["qkv_bias"] = q_out + 2 * kv_out
        if c.attn_bias == "all":
            parts["o_bias"] = d
        if c.qk_norm:
            parts["qk_norm"] = 2 * hd if c.qk_norm_scope == "head" else q_out + kv_out
    if c.attn_sinks:
        parts["sinks"] = nh
    parts["total"] = sum(parts.values())
    return parts


def mlp_params(c: Config, width: int, bias: bool) -> int:
    n_mats = 3 if c.gated_mlp else 2
    out = n_mats * c.d_model * width
    if bias:
        out += width * (2 if c.gated_mlp else 1) + c.d_model
    return out


def param_breakdown(c: Config) -> dict:
    d, L = c.d_model, c.n_layers
    attn_parts = attention_params(c)
    attn = attn_parts["total"]
    dense_mlp = mlp_params(c, c.d_ff, c.mlp_bias)

    per_norm = 2 * d if c.norm == "layernorm" else d   # LayerNorm has gain+bias
    norms = c.norms_per_layer * per_norm

    n_moe = c.n_moe_layers
    n_dense = L - n_moe
    expert = mlp_params(c, c.moe_d_ff, c.expert_bias) if c.is_moe else 0
    router = (d * c.n_experts + (c.n_experts if c.router_bias else 0)) if c.is_moe else 0
    shared = c.n_shared_experts * mlp_params(c, c.moe_d_ff, False) if c.is_moe else 0
    moe_mlp = c.n_experts * expert + shared + router

    dense_layer = attn + dense_mlp + norms
    moe_layer = attn + moe_mlp + norms
    all_layers = n_dense * dense_layer + n_moe * moe_layer
    embed = c.vocab * d
    pos_embed = c.seq_len * d if c.rope_theta is None else 0
    lm_head = 0 if c.tie_embeddings else c.vocab * d
    final_norm = per_norm

    total = embed + pos_embed + all_layers + final_norm + lm_head
    inactive = n_moe * (c.n_experts - c.top_k) * expert
    active = total - inactive
    embeddings = embed + pos_embed + lm_head
    mlp_per_layer = dense_mlp if not c.is_moe else moe_mlp
    return {
        "embed": embed,
        "pos_embed": pos_embed,
        "attn_per_layer": attn,
        "attn_parts": attn_parts,
        "mlp_per_layer": mlp_per_layer,
        "dense_mlp": dense_mlp,
        "expert": expert,
        "router": router,
        "shared_experts": shared,
        "moe_mlp": moe_mlp,
        "n_dense_layers": n_dense,
        "n_moe_layers": n_moe,
        "norms_per_layer": norms,
        "per_layer": dense_layer if not c.is_moe else moe_layer,
        "all_layers": all_layers,
        "attn_all_layers": L * attn,
        "mlp_all_layers": n_dense * dense_mlp + n_moe * moe_mlp,
        "final_norm": final_norm,
        "lm_head": lm_head,
        "total": total,
        "non_embedding": total - embeddings,
        "active": active,
        "active_non_embedding": active - embeddings,
        "active_with_unembedding": active - embed - pos_embed + (0 if lm_head else embed),
        "w_q": attn_parts.get("w_q"), "w_k": attn_parts.get("w_k"),
        "w_v": attn_parts.get("w_v"), "w_o": attn_parts.get("w_o"),
    }


# ---------------------------------------------------------------------------
# FLOPs, KV cache, memory
# ---------------------------------------------------------------------------


def attention_span(c: Config, context: int) -> list[int]:
    """Keys each layer attends to (and caches) at a given context length."""
    full = [context] * (c.n_layers - c.n_sliding_layers)
    sliding = [min(c.sliding_window or context, context)] * c.n_sliding_layers
    return full + sliding


def flops_per_token(c: Config, p: dict) -> dict:
    """
    Training FLOPs per token, PaLM-style accounting plus the output projection:
        C = 6 * N_active_non_embedding  +  6 * vocab * d_model
            + 6 * sum_over_layers( span_l * n_heads * (qk_head_dim + v_head_dim) )
    The first term is 2 FLOPs per MAC x (1 forward + 2 backward) over every
    weight a token actually multiplies (for MoE, only its top_k experts plus
    shared experts). The second is the output projection (logits), a real
    matmul whether or not it shares weights with the embedding; leaving it out
    understates small large-vocab models by 27-38%. The third is the attention
    score and score-times-value matmuls; with qk = v = head_dim and span =
    seq_len it is PaLM's 12 * L * n_heads * head_dim * seq_len. It counts the
    full (unmasked) matrix as PaLM does; causal kernels skip about half of it.
    Sliding-window layers attend to at most sliding_window keys.
    """
    dense = 6 * p["active_non_embedding"]
    head = 6 * c.vocab * c.d_model
    if c.is_mla:
        qk, v = c.qk_nope_head_dim + c.qk_rope_head_dim, c.v_head_dim
    else:
        qk = v = c.head_dim
    attn = 6 * sum(attention_span(c, c.seq_len)) * c.n_heads * (qk + v)
    total = dense + head + attn
    return {"dense": dense, "head": head, "attention": attn, "total": total,
            "attention_share": attn / total}


def kv_elements_per_token_per_layer(c: Config) -> int:
    """Cached numbers per token per layer. MLA caches the latent plus the shared RoPE key."""
    if c.is_mla:
        return c.kv_lora_rank + c.qk_rope_head_dim
    return 2 * c.n_kv_heads * c.head_dim


def kv_cache_bytes(c: Config, context: int, dtype_bytes: int = 2) -> int:
    """Whole KV cache for one sequence of `context` tokens."""
    return sum(attention_span(c, context)) * kv_elements_per_token_per_layer(c) * dtype_bytes


def kv_cache_bytes_per_token(c: Config, dtype_bytes: int = 2) -> int:
    """Growth per extra token at long context: only full-attention layers keep growing."""
    return (c.n_layers - c.n_sliding_layers) * kv_elements_per_token_per_layer(c) * dtype_bytes


def training_memory(p: dict, bytes_per_param_state: int = 16) -> dict:
    """
    Mixed-precision AdamW steady state, per parameter:
        2  bf16 weights
      + 2  bf16 gradients
      + 4  fp32 master weights
      + 4  Adam first moment (fp32)
      + 4  Adam second moment (fp32)
      = 16 bytes
    Activations are excluded, they depend on batch, sequence length and
    whether you checkpoint, and are estimated separately.
    """
    n = p["total"]
    return {
        "weights_bf16": n * 2,
        "grads_bf16": n * 2,
        "master_fp32": n * 4,
        "adam_m": n * 4,
        "adam_v": n * 4,
        "total": n * bytes_per_param_state,
    }


def activation_memory(c: Config, batch: int, checkpointed: bool) -> int:
    """
    Coarse estimate, bf16, FlashAttention (no materialized n x n matrix).
    Checkpointed: only the block boundary is stored -> ~2 bytes * b * s * d * L.
    Not checkpointed: the dominant terms are the MLP intermediates
    (~2 * d_ff for a gated MLP) plus attention projections (~4 * head budget).
    """
    per_tok_per_layer = 2 * c.d_model  # residual stream in/out
    if not checkpointed:
        width = c.d_ff
        if c.is_moe:                     # a token only materializes its own experts' intermediates
            moe_width = (c.top_k + c.n_shared_experts) * c.moe_d_ff
            width = (c.first_k_dense * c.d_ff + c.n_moe_layers * moe_width) / c.n_layers
        per_tok_per_layer += 2 * (2 * width) + 2 * (2 * c.n_heads * c.head_dim
                                                    + 2 * c.n_kv_heads * c.head_dim)
    return int(batch * c.seq_len * c.n_layers * per_tok_per_layer)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate(c: Config, p: dict, tokens: float | None) -> list[tuple[str, str]]:
    """Returns (severity, message). Severity is ERROR / WARN / NOTE."""
    out: list[tuple[str, str]] = []
    d, L = c.d_model, c.n_layers

    if c.is_mla:
        out.append(("NOTE", f"MLA: the cache holds a {c.kv_lora_rank}-wide latent plus a "
                            f"{c.qk_rope_head_dim}-wide shared RoPE key per layer, not per-head K "
                            "and V, so n_kv_heads and head_dim checks do not apply"))
    elif c.n_heads % c.n_kv_heads:
        out.append(("ERROR", f"n_heads={c.n_heads} is not divisible by "
                             f"n_kv_heads={c.n_kv_heads}; GQA groups must be equal-sized"))
    if c.is_mla:
        pass
    elif c.head_dim % 8:
        out.append(("WARN", f"head_dim={c.head_dim} is not a multiple of 8; "
                            "FlashAttention kernels are tuned for 64/128 and will fall back"))
    elif c.head_dim not in (64, 96, 128, 256):
        out.append(("NOTE", f"head_dim={c.head_dim} is off the shipping consensus "
                            "(64 for <1B, 128 almost everywhere else, 256 for Gemma)"))
    if not c.is_mla and c.n_heads * c.head_dim != d:
        ratio = c.n_heads * c.head_dim / d
        out.append(("NOTE", f"n_heads*head_dim = {c.n_heads*c.head_dim} != d_model = {d} "
                            f"({ratio:.2f}x). Legal and deliberate in Qwen3/Gemma, but it "
                            "means attention capacity is decoupled from backbone width, "
                            "make sure that was intentional"))
    has_dense_mlp = not c.is_moe or c.first_k_dense > 0
    if has_dense_mlp and c.d_ff % 128:
        out.append(("WARN", f"d_ff={c.d_ff} is not a multiple of 128; round it up so the "
                            "GEMM tiles cleanly on tensor cores"))
    ff_ratio = c.d_ff / d
    if has_dense_mlp and not 2.4 <= ff_ratio <= 6.2:
        out.append(("WARN", f"d_ff/d_model = {ff_ratio:.2f} is outside the observed "
                            "dense shipping range 2.67-6.0"))
    if c.is_moe:
        gran = c.moe_d_ff / d
        out.append(("NOTE", f"MoE: {c.top_k} of {c.n_experts} routed experts per token"
                            f"{f' plus {c.n_shared_experts} shared' if c.n_shared_experts else ''}; "
                            f"expert width {c.moe_d_ff} = {gran:.2f} x d_model. Active "
                            f"{h(p['active'])} of {h(p['total'])} total ({p['active']/p['total']:.1%}). "
                            "Memory and checkpoint size follow the total; FLOPs follow the active count"))
        if c.top_k > c.n_experts // 2:
            out.append(("WARN", f"top_k={c.top_k} of n_experts={c.n_experts} is barely sparse; "
                                "shipping MoE models activate 3-25% of routed experts"))
    aspect = d / L
    if aspect < 19:
        out.append(("WARN", f"d_model/n_layers = {aspect:.1f}, deeper and narrower than every "
                            "model in the reference table (minimum: SmolLM2-135M at 19.2)"))
    elif aspect > 160:
        out.append(("WARN", f"d_model/n_layers = {aspect:.1f}, very wide and shallow. "
                            "Cheap to run but limited serial computation depth"))
    if d % 128:
        out.append(("NOTE", f"d_model={d} is not a multiple of 128; tensor parallelism and "
                            "GEMM tiling both prefer it"))
    if c.vocab % 64:
        out.append(("WARN", f"vocab={c.vocab} is not a multiple of 64; pad it, the "
                            "unembedding GEMM is the widest matmul in the model"))

    share = p["embed"] / p["total"]
    if share > 0.21 and not c.tie_embeddings:
        out.append(("WARN", f"embeddings are {share:.0%} of parameters and are untied. "
                            "Every model above 21% share in the reference table ties"))
    elif share > 0.35:
        out.append(("NOTE", f"embeddings are {share:.0%} of parameters even tied, the "
                            "vocabulary is large relative to the backbone. Consider a "
                            "smaller vocab unless multilingual/code coverage needs it"))
    elif share < 0.05 and c.tie_embeddings:
        out.append(("NOTE", f"embeddings are only {share:.1%} of parameters; tying saves "
                            "little here, and models this embedding-light in the reference "
                            "table are mostly untied"))

    if c.rope_theta is not None:
        if c.seq_len > 8192 and c.rope_theta <= 10000:
            out.append(("WARN", f"rope_theta={c.rope_theta:g} with seq_len={c.seq_len}. "
                                "Models training natively past 8k use 5e5-1e6, or keep a "
                                "small theta and add explicit scaling (YaRN/NTK)"))
        if c.seq_len <= 4096 and c.rope_theta >= 500000:
            out.append(("NOTE", f"rope_theta={c.rope_theta:g} is large for seq_len="
                                f"{c.seq_len}; it costs nothing but buys nothing either"))

    if tokens is not None:
        tpp = tokens / p["active"]
        if tpp < 20:
            out.append(("WARN", f"{tpp:.0f} tokens/param is below Chinchilla's ~20. "
                                "The model is undertrained for its size, shrink it or "
                                "train longer"
                                + (" (per active parameter; Chinchilla was fit on dense models)"
                                   if c.is_moe else "")))
        elif tpp > 2000:
            out.append(("NOTE", f"{tpp:.0f} tokens/param. Far past compute-optimal, which "
                                "is the right call when inference cost dominates "
                                "(Llama-3.2-1B ran ~7300, Qwen3-0.6B ~60000)"))
    if not out:
        out.append(("NOTE", "no issues found"))
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def h(n: float) -> str:
    for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= div:
            return f"{n/div:.3g}{unit}"
    return f"{n:.0f}"


def hb(n: float) -> str:
    for unit, div in (("TiB", 2**40), ("GiB", 2**30), ("MiB", 2**20), ("KiB", 2**10)):
        if abs(n) >= div:
            return f"{n/div:.2f} {unit}"
    return f"{n:.0f} B"


def report(c: Config, tokens: float | None, batch: int, show_work: bool) -> str:
    p = param_breakdown(c)
    f = flops_per_token(c, p)
    kv = kv_cache_bytes_per_token(c)
    kv_ctx = kv_cache_bytes(c, c.seq_len)
    mem = training_memory(p)
    L = []
    add = L.append

    title = f"# {c.name}: {h(p['total'])} parameters"
    if c.is_moe:
        title += f", {h(p['active'])} active"
    add(title)
    add("")
    add("## Config")
    add("")
    add("| field | value | | field | value |")
    add("|---|---|---|---|---|")
    add(f"| d_model | {c.d_model} | | vocab | {c.vocab} |")
    add(f"| n_layers | {c.n_layers} | | d_ff | {c.d_ff} |")
    add(f"| n_heads | {c.n_heads} | | seq_len | {c.seq_len} |")
    add(f"| n_kv_heads | {c.n_kv_heads} | | rope_theta | "
        f"{'learned abs' if c.rope_theta is None else f'{c.rope_theta:g}'} |")
    add(f"| head_dim | {c.head_dim} | | tied embeddings | {c.tie_embeddings} |")
    if c.is_moe:
        add(f"| experts (routed / active / shared) | {c.n_experts} / {c.top_k} / {c.n_shared_experts} | "
            f"| expert d_ff | {c.moe_d_ff} |")
        add(f"| MoE layers | {c.n_moe_layers} | | leading dense layers | {c.first_k_dense} |")
    if c.is_mla:
        add(f"| kv_lora_rank | {c.kv_lora_rank} | | q_lora_rank | {c.q_lora_rank} |")
        add(f"| qk_nope / qk_rope / v head dims | {c.qk_nope_head_dim} / {c.qk_rope_head_dim} / "
            f"{c.v_head_dim} | | | |")
    if c.n_sliding_layers:
        add(f"| sliding layers | {c.n_sliding_layers} of {c.n_layers} | | window | {c.sliding_window} |")
    add("")
    add("## Derived ratios")
    add("")
    add("| ratio | value | consensus |")
    add("|---|---|---|")
    if not c.is_moe or c.first_k_dense:
        add(f"| d_ff / d_model | {c.d_ff/c.d_model:.2f} | 2.67-6.0 for dense MLPs, no single standard |")
    if c.is_moe:
        add(f"| expert d_ff / d_model | {c.moe_d_ff/c.d_model:.2f} | fine-grained MoE uses narrow experts (0.27-1.0 in the presets) |")
        add(f"| active / total | {p['active']/p['total']:.1%} | 3-28% in the presets |")
    add(f"| d_model / n_layers | {c.d_model/c.n_layers:.1f} | 19-128 in the reference table; 33-128 above 1B |")
    if not c.is_mla:
        add(f"| n_heads / n_kv_heads | {c.n_heads//c.n_kv_heads}x | 1x (none) to 8x; 4x most common |")
    add(f"| embedding share | {p['embed']/p['total']:.1%} | >21% -> every model ties; below that, size and family decide |")
    add(f"| attention share of FLOPs | {f['attention_share']:.1%} | at seq_len={c.seq_len} |")
    add("")
    add("## Parameter budget")
    add("")
    add("| component | count | share |")
    add("|---|---|---|")
    rows = [("token embedding", p["embed"]),
            ("learned positions", p["pos_embed"]),
            ("attention x n_layers", p["attn_all_layers"]),
            ("MLP (dense layers)", p["n_dense_layers"] * p["dense_mlp"]),
            ("routed experts", p["n_moe_layers"] * c.n_experts * p["expert"]),
            ("shared experts", p["n_moe_layers"] * p["shared_experts"]),
            ("routers", p["n_moe_layers"] * p["router"]),
            ("norms", p["norms_per_layer"] * c.n_layers + p["final_norm"]),
            ("lm_head (untied)", p["lm_head"])]
    for label, v in rows:
        if v:
            add(f"| {label} | {h(v)} | {v/p['total']:.1%} |")
    add(f"| **total** | **{h(p['total'])}** | |")
    add(f"| non-embedding | {h(p['non_embedding'])} | {p['non_embedding']/p['total']:.1%} |")
    if c.is_moe:
        add(f"| **active per token** | **{h(p['active'])}** | {p['active']/p['total']:.1%} |")
        add(f"| active, non-embedding | {h(p['active_non_embedding'])} | {p['active_non_embedding']/p['total']:.1%} |")
    add("")

    if show_work:
        a = p["attn_parts"]
        add("<details><summary>arithmetic</summary>")
        add("")
        add("```")
        if c.is_mla:
            qk = c.qk_nope_head_dim + c.qk_rope_head_dim
            if c.q_lora_rank:
                add(f"W_q   = d*q_lora + q_lora(norm) + q_lora*n_heads*(nope+rope) = {c.d_model}*{c.q_lora_rank} + "
                    f"{c.q_lora_rank} + {c.q_lora_rank}*{c.n_heads}*{qk} = {a['w_q']:,}")
            else:
                add(f"W_q   = d*n_heads*(nope+rope) = {c.d_model}*{c.n_heads}*{qk} = {a['w_q']:,}")
            add(f"W_kva = d*(kv_lora+rope) + kv_lora(norm) = {c.d_model}*({c.kv_lora_rank}+{c.qk_rope_head_dim}) "
                f"+ {c.kv_lora_rank} = {a['w_kv_a']:,}")
            add(f"W_kvb = kv_lora*n_heads*(nope+v) = {c.kv_lora_rank}*{c.n_heads}*({c.qk_nope_head_dim}+{c.v_head_dim}) "
                f"= {a['w_kv_b']:,}")
            add(f"W_o   = n_heads*v*d = {c.n_heads}*{c.v_head_dim}*{c.d_model} = {a['w_o']:,}")
        else:
            add(f"W_q  = d_model * n_heads * head_dim    = {c.d_model} * {c.n_heads} * {c.head_dim} = {a['w_q']:,}")
            add(f"W_k  = d_model * n_kv_heads * head_dim = {c.d_model} * {c.n_kv_heads} * {c.head_dim} = {a['w_k']:,}")
            add(f"W_v  = same as W_k                      = {a['w_v']:,}")
            add(f"W_o  = n_heads * head_dim * d_model    = {a['w_o']:,}")
        for extra in ("qkv_bias", "o_bias", "qk_norm", "sinks"):
            if extra in a:
                add(f"+ {extra:<36}= {a[extra]:,}")
        add(f"attn per layer                          = {p['attn_per_layer']:,}")
        n_mats = 3 if c.gated_mlp else 2
        if p["n_dense_layers"]:
            add(f"dense mlp = {n_mats} * d_model * d_ff{'  (gated: gate, up, down)' if c.gated_mlp else '  (up, down)'} "
                f"= {n_mats} * {c.d_model} * {c.d_ff} = {p['dense_mlp']:,}  x {p['n_dense_layers']} layers")
        if c.is_moe:
            add(f"one expert = {n_mats} * d_model * moe_d_ff = {n_mats} * {c.d_model} * {c.moe_d_ff} = {p['expert']:,}")
            add(f"moe mlp = n_experts * expert + shared + router = {c.n_experts} * {p['expert']:,} + "
                f"{p['shared_experts']:,} + {p['router']:,} = {p['moe_mlp']:,}  x {p['n_moe_layers']} layers")
        add(f"norms per layer = {c.norms_per_layer} x {p['norms_per_layer'] // c.norms_per_layer} = {p['norms_per_layer']:,}")
        add(f"all layers                              = {p['all_layers']:,}")
        add(f"+ embedding {c.vocab} * {c.d_model}          = {p['embed']:,}")
        if p["lm_head"]:
            add(f"+ lm_head (untied)                      = {p['lm_head']:,}")
        add(f"= total                                 = {p['total']:,}")
        if c.is_moe:
            add(f"active = total - moe_layers*(n_experts-top_k)*expert = {p['total']:,} - "
                f"{p['n_moe_layers']}*{c.n_experts - c.top_k}*{p['expert']:,} = {p['active']:,}")
        add("```")
        add("</details>")
        add("")

    add("## Compute and memory")
    add("")
    add("| quantity | value | note |")
    add("|---|---|---|")
    add(f"| training FLOPs/token | {h(f['total'])} | 6N{'_active' if c.is_moe else ''} = {h(f['dense'])}, "
        f"output projection = {h(f['head'])}, attention = {h(f['attention'])} |")
    kv_note = ""
    if c.is_mla:
        kv_note = (f"; MLA latent cache. An implementation that caches expanded per-head K and V "
                   f"stores {2*c.n_heads*(c.qk_nope_head_dim+c.qk_rope_head_dim+c.v_head_dim)//2:,} "
                   f"numbers per token per layer instead of {kv_elements_per_token_per_layer(c):,}")
    if c.n_sliding_layers:
        kv_note += f"; {c.n_sliding_layers} sliding layers stop growing at {c.sliding_window} tokens"
    add(f"| KV cache | {hb(kv)}/token | {hb(kv_ctx)} at seq_len={c.seq_len}, bf16{kv_note} |")
    add(f"| KV cache, batch 32 | {hb(kv_ctx*32)} | full context |")
    add(f"| optimizer state | {hb(mem['total'])} | bf16 weights+grads, fp32 master+Adam = 16 B/param |")
    add(f"| activations (checkpointed) | {hb(activation_memory(c, batch, True))} | batch {batch} x seq {c.seq_len} |")
    add(f"| activations (no checkpointing) | {hb(activation_memory(c, batch, False))} | batch {batch} x seq {c.seq_len} |")
    add(f"| inference weights, bf16 | {hb(p['total']*2)} | |")
    add(f"| inference weights, int4 | {hb(p['total']*0.5)} | |")
    add("")

    if tokens is not None:
        total_flops = f["total"] * tokens
        per = "active param" if c.is_moe else "param"
        add(f"Training on **{h(tokens)} tokens** = {tokens/p['active']:.0f} tokens/{per} "
            f"= **{total_flops:.2e} FLOPs**.")
        # Calibration: Llama 3.1 8B is 8.69e23 FLOPs by this accounting and Meta
        # reported 1.46M H100-hours, i.e. ~165 TFLOP/s sustained (~17% MFU on the
        # 989 TFLOP/s bf16 dense peak). Well-tuned small-cluster runs reach ~400.
        lo, hi = total_flops / (400e12 * 3600), total_flops / (150e12 * 3600)
        add(f"At 150-400 TFLOP/s sustained per H100 (Llama 3's published 8B run "
            f"implies ~165; a well-tuned single-node run reaches ~400): "
            f"**{lo:,.0f}-{hi:,.0f} GPU-hours**, "
            f"or {lo/8:,.0f}-{hi/8:,.0f} wall-clock hours on 8 GPUs.")
        add("")
    chin = 20 * p["active"]
    add(f"Chinchilla-optimal token count for {h(p['active'])} {'active ' if c.is_moe else ''}params: **{h(chin)}** "
        f"(20 tokens/param). Small models intended for cheap inference are routinely "
        f"trained far past this.")
    add("")

    add("## Validation")
    add("")
    for sev, msg in validate(c, p, tokens):
        mark = {"ERROR": "[ERROR]", "WARN": "[WARN] ", "NOTE": "[note] "}[sev]
        add(f"- `{mark}` {msg}")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# HF config.json ingestion
# ---------------------------------------------------------------------------


QK_NORM_SCOPE = {"qwen3": "head", "qwen3_moe": "head", "gemma3_text": "head", "gemma3": "head",
                 "olmo2": "full", "olmoe": "full", "olmo3": "full"}


def _unsupported(j: dict) -> list[str]:
    reasons = []
    types = j.get("layer_types") or []
    if any(t not in ("full_attention", "sliding_attention") for t in types):
        reasons.append(f"layer_types contains {sorted(set(types) - {'full_attention', 'sliding_attention'})} "
                       "(linear attention or state-space layers)")
    if j.get("shared_expert_intermediate_size"):
        reasons.append("a gated shared expert (Qwen2-MoE / Qwen3-Next style)")
    if j.get("moe_layer_freq", 1) != 1 or j.get("decoder_sparse_step", 1) != 1 or j.get("mlp_only_layers"):
        reasons.append("MoE layers on a stride other than every layer")
    if j.get("interleave_moe_layer_step", 1) != 1:
        reasons.append("interleaved MoE layers (Llama 4 style)")
    return reasons


def from_hf(path: str) -> Config:
    with open(path) as fh:
        j = json.load(fh)
    j = j.get("text_config", j)                       # multimodal wrappers nest the LM config
    bad = _unsupported(j)
    if bad:
        raise ValueError(f"{path}: not modeled by this script: {'; '.join(bad)}. "
                         "The total would be wrong; count those blocks by hand.")
    if "n_embd" in j:                                   # GPT-2-style key names
        d = j["n_embd"]
        return Config(
            name=j.get("_name_or_path", path), d_model=d, n_layers=j["n_layer"],
            n_heads=j["n_head"], vocab=j["vocab_size"], d_ff=j.get("n_inner") or 4 * d,
            seq_len=j.get("n_positions", 1024),
            tie_embeddings=j.get("tie_word_embeddings", True), gated_mlp=False,
            norm="layernorm", attn_bias="all", mlp_bias=True, rope_theta=None,
        )
    mt = j.get("model_type", "")
    act = j.get("hidden_act") or j.get("hidden_activation") or "silu"
    n_experts = j.get("num_local_experts") or j.get("n_routed_experts") or j.get("num_experts") or 0
    top_k = j.get("num_experts_per_tok") or j.get("experts_per_token") or 0
    kv_lora = j.get("kv_lora_rank") or 0
    L = j["num_hidden_layers"]
    types = j.get("layer_types") or []
    n_sliding = sum(t == "sliding_attention" for t in types)
    window = j.get("sliding_window")
    if not types and window and j.get("sliding_window_pattern"):      # Gemma 3: every Nth layer global
        n_sliding = L - L // j["sliding_window_pattern"]
    elif not types and window and j.get("use_sliding_window", True) and mt in ("mistral", "mixtral"):
        n_sliding = L
    gpt_oss = mt == "gpt_oss"
    gemma = mt.startswith("gemma")
    return Config(
        name=j.get("_name_or_path", path),
        d_model=j["hidden_size"],
        n_layers=L,
        n_heads=j["num_attention_heads"],
        n_kv_heads=j.get("num_key_value_heads"),
        head_dim=j.get("head_dim") or (j.get("v_head_dim") if kv_lora else None),
        vocab=j["vocab_size"],
        d_ff=j["intermediate_size"],
        seq_len=min(j.get("max_position_embeddings", 2048), 32768),
        tie_embeddings=j.get("tie_word_embeddings", gemma),
        gated_mlp=any(k in act for k in ("silu", "swish", "gelu_pytorch_tanh", "geglu")),
        norm="layernorm" if "layer_norm_eps" in j else "rmsnorm",
        qk_norm=mt in QK_NORM_SCOPE,
        qk_norm_scope=QK_NORM_SCOPE.get(mt, "head"),
        attn_bias=("all" if j.get("attention_bias")
                   else ("qkv" if mt.startswith("qwen2") else "none")),
        mlp_bias=j.get("mlp_bias", False),
        rope_theta=j.get("rope_theta", 10000.0),
        norms_per_layer=4 if gemma else 2,
        attn_sinks=gpt_oss,
        n_experts=n_experts,
        top_k=top_k if n_experts else 0,
        n_shared_experts=j.get("n_shared_experts") or 0,
        moe_d_ff=j.get("moe_intermediate_size") if n_experts else None,
        first_k_dense=j.get("first_k_dense_replace", 0) if n_experts else 0,
        router_bias=gpt_oss,
        expert_bias=gpt_oss,
        kv_lora_rank=kv_lora,
        q_lora_rank=j.get("q_lora_rank"),
        qk_nope_head_dim=j.get("qk_nope_head_dim", 0) if kv_lora else 0,
        qk_rope_head_dim=j.get("qk_rope_head_dim", 0) if kv_lora else 0,
        v_head_dim=j.get("v_head_dim", 0) if kv_lora else 0,
        sliding_window=window if n_sliding else None,
        n_sliding_layers=n_sliding,
    )


# ---------------------------------------------------------------------------
# Self-test: the counts must reproduce published parameter totals
# ---------------------------------------------------------------------------


def self_test() -> int:
    fails = 0
    print(f"{'preset':<17} {'computed':>18} {'published':>18}  {'delta':>8}")
    for k, want in PUBLISHED_TOTALS.items():
        got = param_breakdown(PRESETS[k])["total"]
        ok = got == want
        fails += not ok
        print(f"{k:<17} {got:>18,} {want:>18,}  {abs(got - want) / want:>7.3%} {'ok' if ok else 'FAIL'}")
    print()
    print("Active parameters vs the published claim. Conventions: A = both embedding tables")
    print("counted, B = neither, C = unembedding only (gpt-oss's rule). A claim passes when its")
    print("closest convention is within 6% (claims are rounded to 2-3 significant figures):")
    print(f"{'preset':<17} {'A':>8} {'B':>8} {'C':>8} {'claimed':>8}  closest")
    for k, claim in PUBLISHED_ACTIVE.items():
        p = param_breakdown(PRESETS[k])
        conv = {"A": p["active"], "B": p["active_non_embedding"], "C": p["active_with_unembedding"]}
        best = min(conv, key=lambda n: abs(conv[n] - claim))
        err = abs(conv[best] - claim) / claim
        ok = err < 0.06
        fails += not ok
        print(f"{k:<17} {h(conv['A']):>8} {h(conv['B']):>8} {h(conv['C']):>8} {h(claim):>8}  "
              f"{best} ({err:.1%}) {'ok' if ok else 'FAIL'}")
    return fails


# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=sorted(PRESETS))
    ap.add_argument("--from-hf-config", metavar="PATH")
    for name, typ in (("d-model", int), ("n-layers", int), ("n-heads", int),
                      ("n-kv-heads", int), ("head-dim", int), ("vocab", int),
                      ("d-ff", int), ("seq-len", int)):
        ap.add_argument(f"--{name}", type=typ)
    ap.add_argument("--rope-theta", type=float, default=10000.0)
    ap.add_argument("--tie", action="store_true", help="tie input and output embeddings")
    ap.add_argument("--no-gated-mlp", action="store_true", help="plain 2-matrix MLP")
    ap.add_argument("--qk-norm", action="store_true")
    ap.add_argument("--qk-norm-scope", choices=["head", "full"], default="head")
    ap.add_argument("--attn-sinks", action="store_true", help="learned per-head sink logits (gpt-oss)")
    moe = ap.add_argument_group("mixture of experts")
    moe.add_argument("--n-experts", type=int, default=0, help="routed experts per MoE layer")
    moe.add_argument("--top-k", type=int, default=0, help="routed experts per token")
    moe.add_argument("--n-shared-experts", type=int, default=0)
    moe.add_argument("--moe-d-ff", type=int, help="hidden width of one expert (default: --d-ff)")
    moe.add_argument("--first-k-dense", type=int, default=0, help="leading layers with a dense MLP")
    mla = ap.add_argument_group("multi-head latent attention")
    mla.add_argument("--kv-lora-rank", type=int, default=0)
    mla.add_argument("--q-lora-rank", type=int)
    mla.add_argument("--qk-nope-head-dim", type=int, default=0)
    mla.add_argument("--qk-rope-head-dim", type=int, default=0)
    mla.add_argument("--v-head-dim", type=int, default=0)
    sw = ap.add_argument_group("sliding window")
    sw.add_argument("--sliding-window", type=int)
    sw.add_argument("--n-sliding-layers", type=int, default=0)
    ap.add_argument("--attn-bias", choices=["none", "qkv", "all"], default="none")
    ap.add_argument("--tokens", type=float, help="planned training tokens, e.g. 10e9")
    ap.add_argument("--batch", type=int, default=8, help="per-device batch for activation estimate")
    ap.add_argument("--show-work", action="store_true", help="print the parameter arithmetic")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of markdown")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        return min(self_test(), 1)

    if a.from_hf_config:
        c = from_hf(a.from_hf_config)
        if a.seq_len:
            c.seq_len = a.seq_len
    elif a.preset:
        c = dataclasses.replace(PRESETS[a.preset])
        if a.seq_len:
            c.seq_len = a.seq_len
        if a.tokens is None and a.preset == "llama3-8b":
            a.tokens = 15e12
    else:
        need = ["d_model", "n_layers", "n_heads", "vocab", "d_ff"]
        missing = [n for n in need if getattr(a, n) is None]
        if missing:
            ap.error("need --preset, --from-hf-config, or all of: "
                     + ", ".join("--" + m.replace("_", "-") for m in need))
        c = Config(
            d_model=a.d_model, n_layers=a.n_layers, n_heads=a.n_heads,
            n_kv_heads=a.n_kv_heads, head_dim=a.head_dim, vocab=a.vocab,
            d_ff=a.d_ff, seq_len=a.seq_len or 2048, tie_embeddings=a.tie,
            gated_mlp=not a.no_gated_mlp, qk_norm=a.qk_norm, qk_norm_scope=a.qk_norm_scope,
            attn_bias=a.attn_bias, rope_theta=a.rope_theta, attn_sinks=a.attn_sinks,
            n_experts=a.n_experts, top_k=a.top_k, n_shared_experts=a.n_shared_experts,
            moe_d_ff=a.moe_d_ff, first_k_dense=a.first_k_dense, kv_lora_rank=a.kv_lora_rank,
            q_lora_rank=a.q_lora_rank, qk_nope_head_dim=a.qk_nope_head_dim,
            qk_rope_head_dim=a.qk_rope_head_dim, v_head_dim=a.v_head_dim,
            sliding_window=a.sliding_window, n_sliding_layers=a.n_sliding_layers,
        )

    if a.json:
        p = param_breakdown(c)
        print(json.dumps({
            "config": asdict(c),
            "params": p,
            "flops_per_token": flops_per_token(c, p),
            "kv_bytes_per_token": kv_cache_bytes_per_token(c),
            "kv_bytes_at_seq_len": kv_cache_bytes(c, c.seq_len),
            "training_memory_bytes": training_memory(p),
            "validation": [{"severity": s, "message": m}
                           for s, m in validate(c, p, a.tokens)],
        }, indent=2))
    else:
        print(report(c, a.tokens, a.batch, a.show_work))
    return 0


if __name__ == "__main__":
    sys.exit(main())
