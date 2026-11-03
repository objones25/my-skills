# Components: what each one is, why it won, and the bug it invites

The runnable version of everything here is `scripts/reference_block.py`. Read
this file for the reasoning; copy the code from there. `--verify` runs property
tests for each component, and `--bug <name>` explains one failure mode in
detail.

Every shape below uses the same toy config, so you can hold one example in
your head the whole way through:

```
d_model=64   n_layers=4   n_heads=4   n_kv_heads=2   head_dim=16
d_ff=176     vocab=128    batch=2     seq=6
```

## The block

```
x ──┬──────────────────────────────────────────────► + ──┬────────────────────────► + ──►
    │                                                ▲   │                          ▲
    └─► RMSNorm ─► Attention (RoPE, GQA) ─► o_proj ──┘   └─► RMSNorm ─► SwiGLU ─────┘
```

Pre-norm: `x = x + sublayer(norm(x))`. The residual path is clean from
embedding to final norm, with nothing on it. That is why pre-norm trains
without a warmup where post-norm needs one, and it is also why the residual
stream's norm grows with depth. See `stability.md`.

## Contents

| Component | Section |
|---|---|
| RMSNorm | [below](#rmsnorm) |
| RoPE | [below](#rope-rotary-position-embeddings) |
| GQA | [below](#gqa-grouped-query-attention) |
| SwiGLU | [below](#swiglu) |
| KV cache | [below](#kv-cache) |
| Weight tying | [below](#weight-tying) |
| Attention scaling | [below](#the-sqrtd_k-in-the-attention-score) |

---

## RMSNorm

**What it is.** Divide each activation vector by the root mean square of its
own components, then multiply by a learned per-dimension gain.

```
RMSNorm(x) = x / sqrt( (1/d) * sum_i x_i^2  +  eps )  *  g
```

`d` is `d_model`, `x` is one token's activation vector, `g` is a learned vector
of length `d` initialized to ones, `eps` is a small constant (1e-5 or 1e-6 in
practice) that keeps the division finite.

**Why it replaced LayerNorm.** LayerNorm subtracts the mean and adds a learned
shift `beta`:

```
LayerNorm(x) = g * (x - mean(x)) / std(x) + beta
```

RMSNorm drops the mean subtraction and the shift. Zhang and Sennrich (2019)
showed convergence is comparable while training runs 7% to 64% faster
depending on the model, because you skip one reduction pass and store fewer
parameters. Their Transformer machine-translation result: RMSNorm 26.8 BLEU
versus LayerNorm 26.6, at a 7% to 9% speedup.

The property RMSNorm gives up is invariance to shifting the input, which turns
out not to matter for transformers. It keeps the property that does: scale
invariance, so a layer whose inputs double produces the same normalized output.

**Bugs it invites.**

The epsilon goes inside the square root, added to the mean square. Two wrong
placements both type-check and train. `x * rsqrt(mean(x**2)) + eps` makes eps an
output offset and gives NaN on an all-zero row in the forward pass.
`x / (sqrt(mean(x**2)) + eps)` is subtler: the forward pass is finite (0/eps = 0)
and within about eps of the right answer elsewhere, but the gradient on an all-zero
row is NaN, because the derivative of sqrt(u) is infinite at u = 0. All-zero rows
are ordinary (padding, a zero-initialized branch at step 0). `reference_block.py
--verify` checks the gradient for this reason.

Cast to fp32 before squaring. The usual explanation, that a bf16 reduction
loses accumulation precision, is not quite right: PyTorch already accumulates
bf16 reductions in fp32. The loss is in the squaring and the intermediate
cast. Over 10,000 vectors of width 4096, the bf16 path's relative error on
`mean(x^2)` averages 1.5e-3 and peaks near 4e-3; an fp32 upcast adds only ~1e-7
beyond the input's own bf16 rounding. The bf16 error is zero-mean noise of up to
~0.2% in the norm scale at every norm site, twice per layer. Small, and free to
avoid.

The gain multiply happens after the cast back to the compute dtype in most
reference implementations. Doing it in fp32 and then casting is also fine, but
they differ numerically, which matters if you are matching another codebase
bit-for-bit.

---

## RoPE (Rotary Position Embeddings)

**What it is.** Instead of adding a position vector to the token embedding,
rotate the query and key vectors by an angle proportional to their position.
Split each head's `head_dim` into `head_dim/2` two-dimensional blocks, and
rotate block `i` at position `m` by angle `m * theta_i`, where

```
theta_i = base ** (-2i / head_dim),   i = 0 .. head_dim/2 - 1,   base = 10000 in RoFormer
```

Low `i` gives a large `theta_i` and a fast-oscillating dimension (short-range
resolution). High `i` gives a slow one (long-range).

**Why rotating works.** A 2D rotation by angle `a` is multiplication by

```
R(a) = [ cos a   -sin a ]
       [ sin a    cos a ]
```

Rotations compose: `R(a)^T R(b) = R(b - a)`. So if you rotate the query at
position `m` and the key at position `n`, the attention score becomes

```
(R(m*theta) q)^T (R(n*theta) k)  =  q^T R((n - m)*theta) k
```

The score depends only on `n - m`, the relative distance. Absolute positions
cancel. This is the whole point: attention cares about how far apart two tokens
are, not where they sit in the buffer, and RoPE delivers that exactly, inside
the dot product, rather than approximately, bolted onto the input.

`reference_block.py --verify` checks this numerically: a query at position 3
against a key at position 5 scores `+0.616412`, and a query at 10 against a key
at 12 scores `+0.616412`, while 3 against 6 scores `+0.329711`.

Rotation matrices are orthogonal, so RoPE preserves vector norms. It adds zero
parameters. RoFormer also proves (Section 3.4.3, via Abel summation) that an
upper bound on the attention weight decays as `|n - m|` grows, so distant
tokens are bounded toward lower similarity, though not monotonically.

**RoPE touches Q and K only. V is never rotated.** The position signal belongs
in the score, not in the content being retrieved.

**The base value is a lever, not a constant.** RoFormer used 10000. Modern
long-context models raise it: Llama 3 uses 500000, Qwen 1000000, SmolLM3
5000000. Raising the base slows every frequency, which spreads the resolvable
range over a longer context. The alternative strategy is to keep the base small
and rescale at the end (YaRN, NTK-aware, LongRoPE). Both ship. See the spread
table in `reference-configs.md`.

**Bugs it invites.**

*Applying RoPE before splitting heads.* RoPE rotates within each head's
`head_dim` slots. Applied to the flat `(batch, seq, n_heads*head_dim)` tensor,
the pairing straddles head boundaries and rotates head 0's dimensions against
head 2's. Shapes are unchanged, nothing errors, and the model trains a few
percent worse, which is small enough to blame on the data. Split heads first.

*Mixing the two pairing conventions.* Two exist:

| Convention | Pairs | Used by |
|---|---|---|
| interleaved | dim `2i` with dim `2i+1` | the RoFormer paper, some from-scratch code |
| halves | dim `i` with dim `i + head_dim/2` | Hugging Face Llama, and therefore most checkpoints |

Both are valid rotations. Training from scratch works either way and nothing
errors. It only bites when you load or export a checkpoint, where the weights
are silently incompatible and the loss is merely bad rather than NaN. If you
plan to interoperate with Hugging Face weights, use the halves convention, and
make sure the `cos`/`sin` table is built for the same pairing (each frequency
appears twice, `cat([freqs, freqs], -1)`).

*Rotating cached keys twice.* Append to the KV cache after rotating, so each
cached key keeps the rotation for its own absolute position. Rotating on the
way out of the cache re-rotates everything by the current position.

---

## GQA (Grouped-Query Attention)

**What it is.** Give every attention head its own query projection, but let
several heads share one key/value projection. With `n_heads` query heads and
`n_kv_heads` KV heads, each KV head serves a group of `n_heads / n_kv_heads`
query heads.

```
MHA          n_kv_heads = n_heads    most expressive, largest KV cache
GQA          1 < n_kv_heads < n_heads
MQA          n_kv_heads = 1          smallest cache, least diversity
```

**Why share K and V rather than Q.** The KV cache stores keys and values for
every past token, at every layer, for the life of a generation. Queries are
consumed immediately and never revisited. So sharing K/V is what shrinks the
thing that dominates inference memory, and keeping Q per-head preserves the
head specialization that makes multi-head attention useful in the first place.

Concretely, from `config_budget.py --preset llama3-8b`: KV cache at 128 KiB per
token, 1 GiB at 8192 context, 32 GiB at batch 32. Going to MHA would multiply
all three by four.

**Where the group count comes from.** Ainslie et al. (2023) converted an
existing MHA checkpoint by mean-pooling the key (and value) heads inside each
group, then "uptraining" for `alpha = 0.05` of the original pretraining steps.
They chose 8 groups as a middle ground: inference time degrades slowly as
groups increase from MQA, then faster as it approaches full MHA. On T5-XXL:

| Variant | inference time / sample | average score |
|---|---|---|
| MHA | 1.51 | 47.2 |
| MQA | 0.24 | 46.6 |
| GQA-8 | 0.28 | 47.1 |

GQA recovers essentially all of MHA's quality at close to MQA's speed. Note the
paper's own caveat: it evaluated encoder-decoder models with ROUGE, which it
calls "a flawed evaluation," and never compared against training the same
architecture from scratch.

**Bugs it invites.**

*The wrong expansion.* Both of these produce
`(batch, n_heads, seq, head_dim)` from `(batch, n_kv_heads, seq, head_dim)`:

```python
x[:, :, None, :, :].expand(b, n_kv, n_rep, s, d).reshape(b, n_kv*n_rep, s, d)  # correct
x.repeat(1, n_rep, 1, 1)                                                        # wrong
```

The correct one gives KV head `j` the contiguous block of query heads
`[j*n_rep, (j+1)*n_rep)`. With `n_kv=4, n_rep=2`, the head index order is
`[0,0,1,1,2,2,3,3]`. `repeat` tiles the whole set instead and gives
`[0,1,2,3,0,1,2,3]`. `repeat_interleave(n_rep, dim=1)` also gives the correct
layout.

Be precise about what this costs, because the obvious description ("every
query head reads the wrong KV head") overstates it. Correct: query head `j`
reads KV head `j // n_rep`. Buggy: query head `i` reads KV head `i % n_kv`.
Those differ by a *permutation of query heads*, and query heads are
independent and symmetric at init. `reference_block.py --verify` proves it:
permute the head blocks of `W_q` and the matching column blocks of `W_o` by
the perfect shuffle, and the buggy path reproduces the correct one to
`1.79e-07`, against `1.20e+00` with identical weights.

So from scratch it trains fine. What it actually breaks is every boundary
where another implementation assumes the contiguous layout: exported
checkpoints, imported checkpoints, `scaled_dot_product_attention(...,
enable_gqa=True)`, and fused GQA kernels. It also materializes the full
expanded K and V, which `expand` does not, so it wastes activation memory
proportional to `n_rep`. Fix it because it is a lie about your own weight
layout, not because it is costing you loss.

*Uneven groups.* `n_heads % n_kv_heads != 0` has no sensible meaning. Assert it.

---

## SwiGLU

**What it is.** A gated MLP. Where the original transformer's feedforward is
`down(relu(up(x)))`, a gated MLP splits the up-projection in two and multiplies
them elementwise:

```
SwiGLU(x) = down( silu(gate(x)) * up(x) )
silu(z)   = z * sigmoid(z)
```

`gate` and `up` are both `d_model -> d_ff`; `down` is `d_ff -> d_model`. No
biases. Three matrices instead of two.

**Why it won.** Shazeer (2020) swapped the activation in T5-base and held
parameters fixed. Heldout log-perplexity after 524k steps:

| FFN variant | log-perplexity | GLUE avg | SuperGLUE avg |
|---|---|---|---|
| ReLU (baseline) | 1.677 | 83.80 | 72.76 |
| GELU | 1.679 | 83.86 | - |
| Swish | 1.683 | 83.60 | - |
| GEGLU | **1.633** | 84.12 | 73.96 |
| **SwiGLU** | **1.636** | 84.36 | **74.56** |
| ReGLU | 1.645 | **84.67** | 73.66 |

The gated variants beat the ungated ones consistently. The paper declines to
explain why: "we attribute their success, as all else, to divine benevolence."

**The 2/3 rule.** Three matrices instead of two means that to hold parameters
fixed you scale `d_ff` by 2/3. Shazeer took T5-base from `d_ff = 3072` to
`2048`, exactly 2/3. Applied to the classic 4x expansion:

```
d_ff = round_to_multiple( (2/3) * 4 * d_model, 256 )
```

For `d_model = 4096` this gives 11008, which is Llama-2-7B's actual `d_ff`.
`reference_block.py --verify` confirms the parameter counts match to 0.8%.

Round to a multiple of 256 so the GEMM tiles cleanly on tensor cores. Shipping
models range from 2.67x to 6.0x `d_model`, so the 2/3-of-4x result is a
starting point, not a law.

**Bugs it invites.** Forgetting the 2/3 and using `d_ff = 4*d_model` with three
matrices, which quietly inflates the MLP by 50% and shifts the model's
parameter distribution. The MLP is already 70% of a Llama-3-8B's parameters, so
this is not a rounding error.

---

## KV cache

**What it is.** During generation, each new token attends to every previous
token, which needs their keys and values. Cache them instead of recomputing.

This is an exact optimization: same output, less work. It is inference-only.
Training uses teacher forcing and processes the whole sequence at once, so the
question never arises.

**Cache K and V, never Q.** A past token's query is never used again. The
current token's query goes against all cached keys and values, and that is all
attention needs.

**Size.** `2 * n_layers * n_kv_heads * head_dim * bytes` per token. Run
`config_budget.py` rather than doing this by hand, because the number is
usually larger than people expect: for Llama-3-8B at batch 32 and 8k context it
is 32 GiB, which is more than the weights.

**Bugs it invites.**

*Leaving `is_causal=True` during incremental decode.* With a length-1 query
against a length-`t` key, a causal mask on that 1-by-`t` rectangle allows only
the first key position. The model attends to token 0 and nothing else. Prefill
looks perfect, because there query and key have equal length, and generation
degrades into nonsense after the first token. Guard on
`k.shape[2] == q.shape[2]`.

That guard is only correct for the two common cases, full prefill and
single-token decode. Chunked prefill, where a length-`m` query runs against a
length-`t` key with `1 < m < t`, needs an explicit mask offset by `t - m`
rather than either branch.

*Position offset.* The cached tokens occupy positions `0..t-1`, so the new
token's RoPE angle must use position `t`, not 0. Pass the offset explicitly.

*Rotating twice.* Covered under RoPE above: append to the cache after rotating.

---

## Weight tying

**What it is.** Use one matrix for both the input embedding (`vocab x d_model`
lookup) and the output projection (`d_model -> vocab` logits).

**What it buys.** `vocab * d_model` parameters, which for a small model is most
of the model. Gemma3-270M is 63% embedding; Qwen2.5-0.5B is 28%. Press and Wolf
(2016) also found it reduces overfitting: on PTB, tying cut test perplexity
from 78.4 to 74.3 while cutting the model from 66M to 51M parameters, with
training perplexity going *up*, which is what less overfitting looks like.

**When models stop.** The better predictor is embedding share, not parameter
count. Above roughly 10% share, models tie. Below, most untie. Gemma keeps
tying to 12B because its 262k vocabulary keeps the share above the threshold at
sizes where its peers have stopped. OLMo2-1B unties at 13.8% share and
Phi-4-mini ties at 16%, so it is a soft rule.

**What to watch.** Press and Wolf's analysis of the update dynamics found that
in a tied model, every row of the shared matrix gets updated each step in a way
that resembles the *output* embedding's update, not the input's. So the tied
matrix behaves like an output embedding, so initialize it at output-projection
scale (std 0.02, as `reference_block.py` does) rather than a unit-variance
embedding scale, which makes the initial logits large.

**The initialization-order trap.** If you tie in `__init__` and then call
`self.apply(init_fn)`, the shared tensor is drawn twice, and whichever module
`apply` visits last wins. Module order follows registration order, so which
draw survives depends on the order you wrote the attribute assignments in.
Either tie *after* `apply`, or make the init deterministic enough that it does
not matter, and assert `lm_head.weight.data_ptr() == embed.weight.data_ptr()`
either way.

Tying is not universally good. It was actively harmful for word2vec-style
models, where driving down a token's output-embedding norm also destroys its
input embedding. The decoupling that makes it safe in a transformer comes from
the many layers in between.

---

## The sqrt(d_k) in the attention score

```
Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) V
```

`d_k` is `head_dim`, not `d_model`. The original paper's reason, from the
footnote in Section 3.2.1: for large `d_k` the dot products grow large in
magnitude, pushing softmax into a region where its gradients are extremely
small. Dividing by `sqrt(d_k)` counteracts that.

The mechanism is that a dot product of two `d_k`-dimensional vectors with
independent unit-variance components has variance `d_k`, so its standard
deviation is `sqrt(d_k)`. Dividing by that returns the pre-softmax logits to
unit scale regardless of head dimension.

Attention logit growth during training is a real failure mode that this
scaling alone does not prevent. QK-norm and logit softcapping exist for it. See
`stability.md`.

---

## What each component costs

From `config_budget.py --preset llama3-8b --show-work`, an 8.03B model:

| Component | Parameters | Share |
|---|---|---|
| MLP, all 32 layers | 5.64B | 70.2% |
| Attention, all 32 layers | 1.34B | 16.7% |
| Token embedding | 525M | 6.5% |
| Untied lm_head | 525M | 6.5% |
| Norms | 266K | 0.003% |

The MLP is where the parameters are, which is why mixture-of-experts replaces
the MLP and not the attention, and why LoRA on the MLP matters more than LoRA
on attention (LoRA Without Regret, a vendor blog; see `sources.md`).
