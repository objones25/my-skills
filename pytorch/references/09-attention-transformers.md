# Attention and transformer components

SDPA, FlexAttention, RoPE, RMSNorm, KV cache, GQA, the modern block.

## Overview

A modern decoder-only transformer block is a small fixed set of parts: a
normalization, multi-head self-attention with rotary position encoding, another
normalization, and a gated MLP, each wrapped in a residual connection. This file
covers the pieces the core `nn` layers do not hand you ready-made, the way
LLaMA-style models use them. One toy shape is threaded throughout: batch `B=2`,
sequence `L=4`, `d_model=8`, `n_heads=2`, so `head_dim=4`.

## Recommended approach

Normalize -> sublayer -> residual add, twice. Norm **before** the sublayer (pre-norm)
is the modern default for training deep stacks without warmup fragility.

```python
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, d_model=8, n_heads=2):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model, eps=1e-6)     # set eps - default is None
        self.attn  = SelfAttention(d_model, n_heads)   # RoPE + SDPA inside
        self.norm2 = nn.RMSNorm(d_model, eps=1e-6)
        self.mlp   = SwiGLU(d_model)
    def forward(self, x):                              # x: (B, L, d_model)
        x = x + self.attn(self.norm1(x))               # pre-norm + residual
        x = x + self.mlp(self.norm2(x))
        return x
```

Inside attention, always use `F.scaled_dot_product_attention` - never a hand-rolled
softmax:

```python
out = F.scaled_dot_product_attention(q, k, v, is_causal=True)   # q,k,v: (B, H, L, head_dim)
```

## API reference

`scaled_dot_product_attention` - signature from the docstring (a C builtin, so
`inspect.signature` cannot read it; the docstring is unchanged on 2.15.0):

```
F.scaled_dot_product_attention(query, key, value, attn_mask=None,
                               dropout_p=0.0, is_causal=False, scale=None,
                               enable_gqa=False) -> Tensor
```

| Param | Meaning |
|-------|---------|
| `query`, `key`, `value` | `(B, n_heads, L, head_dim)`; default `scale = 1/sqrt(head_dim)` |
| `attn_mask` | bool mask where **`True` = participate** (the inverse of `nn.MultiheadAttention`), or a float mask *added* to the scores (`-inf` blocks) |
| `is_causal` | applies a triangular mask; assumes a **square** score matrix |
| `enable_gqa` | share KV heads across query groups (more query heads than KV heads) |

Component APIs, introspected from torch 2.15.0:

| Piece | Form | Notes |
|-------|------|-------|
| Pick a kernel | `with sdpa_kernel(SDPBackend.FLASH_ATTENTION): ...` | backends: `FLASH_ATTENTION`, `EFFICIENT_ATTENTION`, `CUDNN_ATTENTION`, `MATH`, `OVERRIDEABLE`; auto-selected otherwise. `sdpa_kernel(backends, set_priority=False)` |
| Which backend can run | `torch.nn.attention.can_use_flash_attention(params)` | diagnose a silent fallback to `MATH` |
| High-level module | `nn.MultiheadAttention(embed_dim, num_heads, batch_first=False)` | opposite mask convention; note the `batch_first` default |
| RMSNorm | `nn.RMSNorm(normalized_shape, eps=None, elementwise_affine=True)` | no bias; `eps=None` resolves to the dtype epsilon, so set it from the checkpoint config (`1e-6` original LLaMA, `1e-5` Llama 3.x) |
| FlexAttention | `flex_attention(q, k, v, score_mod=None, block_mask=None, scale=None, enable_gqa=False, return_lse=False, kernel_options=None)` | custom masks and score modifications, compiled into one kernel |
| Block mask | `create_block_mask(mask_mod, B, H, Q_LEN, KV_LEN, device=None, BLOCK_SIZE=128, ...)` | precompute once, reuse every step |
| Mask combinators | `and_masks(...)`, `or_masks(...)`, `noop_mask` | compose mask_mods |
| Variable length | `torch.nn.attention.varlen` | packed sequences without padding |
| RoPE | apply to `q` and `k` before SDPA | **no built-in module in core** (checked: nothing in `torch.nn` on 2.15.0) - write it or use `torchtune` |
| KV cache | append the new token's `k, v` to the stored past | drop `is_causal` on cached steps |
| GQA | fewer KV heads plus `enable_gqa=True` | shrinks the cache |
| Gated MLP | `w2(F.silu(w1(x)) * w3(x))` (SwiGLU) | hidden about `8/3 * d`, not `4 * d` |

Self-attention shape thread (`B=2, L=4, d_model=8, n_heads=2`):

| Step | Tensor | Shape |
|------|--------|-------|
| input | `x` | `(2, 4, 8)` |
| qkv projection | `qkv` | `(2, 4, 24)` |
| split per head | `q, k, v` | `(2, 2, 4, 4)` |
| attention out | `o` | `(2, 2, 4, 4)` |
| merge heads | `o` | `(2, 4, 8)` |
| output projection | return | `(2, 4, 8)` |

## Walkthrough

A single attention head over `L=3` tokens with `head_dim=2`, computed by hand and
then checked against SDPA. Output executed on torch 2.15.0.

**1. Query, key, value for 3 tokens.**

```python
import torch, torch.nn.functional as F
q = torch.tensor([[1.,0.],[0.,1.],[1.,1.]])   # (3, 2)
k = torch.tensor([[1.,0.],[1.,1.],[0.,1.]])
v = torch.tensor([[10.,0.],[0.,10.],[5.,5.]])
```

**2. Scores are `Q K^T / sqrt(d_k)`.** With `head_dim=2`, divide by sqrt(2).

```python
scores = q @ k.T / 2 ** 0.5
print(torch.round(scores, decimals=3))
```
```
tensor([[0.7070, 0.7070, 0.0000],
        [0.0000, 0.7070, 0.7070],
        [0.7070, 1.4140, 0.7070]])
```

**3. Causal mask, then softmax.** Future positions go to `-inf`, so each row's
weights are lower-triangular and sum to 1.

```python
mask = torch.triu(torch.ones(3, 3, dtype=torch.bool), 1)
w = torch.softmax(scores.masked_fill(mask, float("-inf")), -1)
print(torch.round(w, decimals=3))
print("rows sum:", w.sum(1).tolist())
```
```
tensor([[1.0000, 0.0000, 0.0000],
        [0.3300, 0.6700, 0.0000],
        [0.2480, 0.5030, 0.2480]])
rows sum: [1.0, 1.0, 1.0]
```

Token 0 attends only to itself; later tokens blend their visible past.

**4. Output is the weighted average of value vectors - and SDPA agrees exactly.**

```python
manual = w @ v
sdpa = F.scaled_dot_product_attention(q[None,None], k[None,None], v[None,None],
                                      is_causal=True)[0, 0]
print(torch.round(manual, decimals=3))
print(torch.round(sdpa,   decimals=3))
print("allclose:", torch.allclose(manual, sdpa, atol=1e-5))
```
```
tensor([[10.0000,  0.0000],
        [ 3.3020,  6.6980],
        [ 3.7240,  6.2760]])
tensor([[10.0000,  0.0000],
        [ 3.3020,  6.6980],
        [ 3.7240,  6.2760]])
allclose: True
```

Why divide by sqrt(d_k): the dot product sums `d_k` products, so with unit-variance
entries its variance grows like `d_k` and scores scale as sqrt(d_k). Large scores
saturate softmax (one weight near 1, the rest near 0) and gradients vanish. Dividing
keeps score variance near 1 for any head dimension.

**5. FlexAttention reproduces the same result from a mask function**, and the block
mask tells you how much work it skips.

```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
B, H, L, D = 1, 2, 256, 32
q2, k2, v2 = (torch.randn(B, H, L, D) for _ in range(3))
causal = lambda b, h, q_idx, kv_idx: q_idx >= kv_idx
bm = create_block_mask(causal, B=None, H=None, Q_LEN=L, KV_LEN=L, device="cpu")
print("allclose to SDPA:", torch.allclose(
    flex_attention(q2, k2, v2, block_mask=bm),
    F.scaled_dot_product_attention(q2, k2, v2, is_causal=True), atol=1e-4))

sliding = lambda b, h, q_idx, kv_idx: (q_idx >= kv_idx) & (q_idx - kv_idx <= 255)
for name, mod in [("causal", causal), ("sliding w=256", sliding)]:
    m = create_block_mask(mod, None, None, 4096, 4096, device="cpu")
    print(f"{name:14s} L=4096 sparsity: {m.sparsity():.1f}%")
```
```
allclose to SDPA: True
causal         L=4096 sparsity: 48.4%
sliding w=256  L=4096 sparsity: 90.9%
```

Sparsity is the fraction of `128 x 128` blocks FlexAttention skips entirely. A
sliding window at 4096 tokens skips 91% of the work - that is the speedup, and it is
why FlexAttention beats SDPA on sparse patterns while dense patterns still favor SDPA.

## Gotchas

- **Use `F.scaled_dot_product_attention`, not a hand-rolled softmax** - it is stable
  and dispatches to FlashAttention.
- **SDPA boolean mask `True` = keep; `nn.MultiheadAttention` masks use `True` =
  block.** Opposite conventions - mixing them inverts your masking.
- **`is_causal=True` only for a square (prefill) score matrix.** A cached
  single-token step has a `1 x L_past` matrix and must **not** set it - the new token
  legitimately attends to all past keys.
- **Call FlexAttention under `torch.compile`.** Eager `flex_attention` warns and is
  slow; `torch.compile(flex_attention)` is the intended path.
- **`score_mod` and `mask_mod` must be pure functions.** Since 2.15 they must not mutate
  their inputs or return an alias of one (release notes); compute a new tensor instead.
- **Recreate `block_mask` when the sequence length changes**, and reuse it across
  steps otherwise - building it every iteration wastes real time.
- **`.transpose` then `.view` usually errors** when the new shape needs contiguous memory. Add `.contiguous()` or use
  `.reshape()`.
- **`nn.RMSNorm` has no bias, and `eps` defaults to `None`** (the dtype epsilon), not
  LayerNorm's `1e-5`. Match the checkpoint: `1e-6` for the original LLaMA, `1e-5`
  for Llama 3.x.
- **RoPE pairing convention varies** (interleaved versus split-half) - match the
  reference model or weights will load but produce garbage. No built-in RoPE in core.
- **KV cache memory grows with context** and dominates long-context inference; GQA
  mitigates it.
- **SwiGLU hidden size is about `8/3 * d`**, not `4 * d`, to match the classic FFN's
  parameter count (three matrices instead of two).
- **A silent fallback to the `MATH` backend** (unsupported dtype, head dim, or mask)
  turns "FlashAttention" into a slow reference implementation. Check with
  `can_use_flash_attention` or force a backend and see whether it raises.

## Alternatives

**Roll your own attention versus `nn.MultiheadAttention`.** The stock module is
batteries-included. Roll your own over SDPA when you need RoPE, a KV cache, GQA, or a
custom QKV layout - which it does not expose cleanly, and whose mask conventions are
the inverse of SDPA's.

```python
class SelfAttention(nn.Module):
    def __init__(self, d_model=8, n_heads=2):
        super().__init__()
        self.n_heads, self.head_dim = n_heads, d_model // n_heads
        self.qkv  = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
    def forward(self, x):                              # (B, L, d_model)
        B, L, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)         # each (B, L, d_model)
        shp = lambda t: t.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        q, k, v = shp(q), shp(k), shp(v)               # (B, n_heads, L, head_dim)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        o = o.transpose(1, 2).contiguous().view(B, L, d)
        return self.proj(o)
```

**Three attention paths, by what you need:**

| Path | Use when |
|------|----------|
| `F.scaled_dot_product_attention` | standard causal or padded attention - the default |
| `flex_attention` + `create_block_mask` | sliding window, document masks, ALiBi, soft-capping, prefix-LM - anything expressible as a `score_mod` or `mask_mod` |
| `nn.MultiheadAttention` | cross-attention in an encoder-decoder where you want the module to own the projections |

**Positional-encoding menu:**

| Method | Where applied | Character |
|--------|---------------|-----------|
| Sinusoidal (fixed) | added to embeddings at input | parameter-free, extrapolates poorly |
| Learned absolute | added embedding table | simple, fixed max length |
| Rotary (RoPE) | rotates Q,K inside each attention | relative, the modern default |
| ALiBi | linear bias on scores | cheap, easy via a FlexAttention `score_mod` |

RoPE gives relative position because rotating `q` by `m*theta` and `k` by `n*theta`
makes the score a function of the offset `n - m` alone - rotations compose by adding
angles. Learned absolute embeddings cannot run past their table at all; RoPE can, but quality
degrades past the training length without explicit scaling such as YaRN (Su et al.,
2021, for the relative-position property). RMSNorm drops the mean subtraction because ablations found the
re-scaling, not the centering, is what stabilizes transformer training - halving the
reductions and parameters at no quality cost. GQA shrinks the KV cache because its
size and per-token memory bandwidth scale with the number of KV heads: sharing 8
query heads over 2 KV heads quarters it. MQA (1 KV head) is the extreme.

```python
out = F.scaled_dot_product_attention(q8, k2, v2, enable_gqa=True)  # 8 q heads, 2 kv heads

class SwiGLU(nn.Module):
    def __init__(self, d_model=8, hidden=None):
        super().__init__()
        hidden = hidden or int(8 / 3 * d_model)        # ~ a 4x ReLU FFN; real models round up
                                                       # to a multiple of 256 (Llama-2-7B: 11008)
        self.w1 = nn.Linear(d_model, hidden, bias=False)   # gate
        self.w3 = nn.Linear(d_model, hidden, bias=False)   # value
        self.w2 = nn.Linear(hidden, d_model, bias=False)   # output
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
```

## Related

- `03-models-layers.md` - every part here is an `nn.Module` assembled by the same rules.
- `01-devices-amp.md` - FlashAttention needs fp16/bf16, so autocast is a prerequisite.
- `11-compile-export.md` - FlexAttention is designed to run under `torch.compile`.
- `04-losses-optimizers.md` - `nn.LinearCrossEntropyLoss` pairs with a large-vocab head.
- `13-performance-profiling.md` - confirming which attention backend actually ran.

## Sources

- `inspect` and docstring against torch 2.15.0 (re-checked on 2.14.1) (`F.scaled_dot_product_attention`,
  `flex_attention`, `create_block_mask`, `sdpa_kernel`, `SDPBackend`, `nn.RMSNorm`);
  all walkthrough output executed on the same build.
- docs.pytorch.org/docs/2.13 - `nn.attention.html`, `nn.attention.flex_attention.html`.
- PyTorch 2.13 release blog - FlexAttention on MPS, deterministic backward.
- PyTorch 2.15 release notes - pure `score_mod`/`mask_mod`, `FlopCounterMode` counts
  `flex_attention`, FA4 with different QK and V head dims, head-dim-256 NaN fix.
- Su et al. (2021), "RoFormer," arXiv:2104.09864 - RoPE.
