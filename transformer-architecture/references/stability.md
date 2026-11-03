# Training stability: what goes wrong, why, and what fixes it

A pretraining run that diverges at 30% is expensive. Most of the interventions
here cost nothing at inference and little at training, so the default posture
is to include them rather than to add them after the first failure.

Where the literature disagrees, this file says so instead of picking one. Those
disagreements are collected at the end and they are real, not noise.

## Contents

- [Symptom to cause](#symptom-to-cause)
- [Norm placement](#norm-placement)
- [Initialization](#initialization)
- [Attention logit growth](#attention-logit-growth-qk-norm-and-softcapping)
- [Output logit growth: z-loss](#output-logit-growth-z-loss)
- [Optimizer settings](#optimizer-settings)
- [Learning rate schedules](#learning-rate-schedules)
- [Weight decay](#weight-decay)
- [Muon](#muon)
- [Gating as a stabilizer](#gating-as-a-stabilizer)
- [Stress testing a recipe](#stress-testing-a-recipe)
- [What to log](#what-to-log)
- [Where sources disagree](#where-sources-disagree)

## Symptom to cause

| Symptom | Most likely cause | Where to look |
|---|---|---|
| Loss spikes that recover | Data batch interacting with parameter state; repeated n-grams | [Data](#data-as-a-spike-cause) |
| Loss spikes that do not recover | Attention logit growth; LR too high for scale | [QK-norm](#attention-logit-growth-qk-norm-and-softcapping) |
| NaN in fp16, fine in bf16 | Gradient underflow, loss scaler crashed to 0 | [Optimizer](#optimizer-settings) |
| NaN in bf16 | Activation magnitude runaway, not a range problem | [Norm placement](#norm-placement) |
| Grad norm collapses toward 0 | AdamW eps too large relative to grad RMS | [Optimizer](#optimizer-settings) |
| Loss flat at ln(vocab) | Nothing is learning: check gradient flow, tying, LR | run `reference_block.py --verify` |
| Val loss diverges from train early | Overfitting, or too few tokens per parameter | Chinchilla check in `config_budget.py` |
| Later layers near-identity | Too deep for the width at this budget | run `scripts/residual_probe.py` |
| Attention collapses to one-hot | Entropy collapse; entropy is bounded below by the logits' spectral norm | [QK-norm](#attention-logit-growth-qk-norm-and-softcapping) |
| Needs warmup or immediately diverges | Post-norm placement | [Norm placement](#norm-placement) |

## Norm placement

**Post-norm** (original 2017): `x = LN(x + Sublayer(x))`. The norm sits on the
residual path.

**Pre-norm** (modern default): `x = x + Sublayer(LN(x))`. The residual path is
clean.

Xiong et al. (2020) proved with a mean-field analysis that at initialization
post-norm has large expected gradients near the output layer, which is why it
needs a learning-rate warmup and why removing warmup diverges. Pre-norm's
gradients are well-behaved at init, so it trains without warmup.

The catch is that "well-behaved at initialization" is an init-time result, and
the problem shows up later. Because nothing renormalizes the residual stream
between blocks, its variance grows with depth, and Kim et al. (2025, "Peri-LN")
show this growth can be exponential in pre-norm and persists through training,
producing activations that in their runs exceeded fp16 range. In their 1.5B
runs, pre-norm showed early instability across seeds that lowering the learning
rate did not fix.

Three responses in shipping models:

| Scheme | Form | Used by |
|---|---|---|
| pre-norm | `x + f(N(x))` | Llama, Mistral, Qwen |
| peri-LN | `x + N(f(N(x)))` | Gemma 2 and 3 ("we use RMSNorm to normalize the input and output of each sub-layer") |
| reordered norm | `x + N(f(x))` | OLMo 2, Chameleon, following Swin |

OLMo 2 reports that reordered norm alone and QK-norm alone each helped little,
but combined they cut their gradient-norm spike score from 0.108 to 0.069.

Practically: pre-norm is the safe default at small scale. If you are past a
billion parameters, or seeing activation growth in `residual_probe.py`, adding
a norm on the sublayer output is the cheap intervention with two independent
shipping precedents.

## Initialization

**The residual scaling rule.** GPT-2: "We scale the weights of residual layers
at initialization by a factor of `1/sqrt(N)` where N is the number of residual
layers." N is `2 * n_layers`, since each block writes twice (attention output
projection, MLP down projection). nanoGPT applies it to exactly those tensors:

```python
nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * n_layer))   # o_proj, down_proj only
nn.init.normal_(p, mean=0.0, std=0.02)                          # everything else
```

Without it, all `2*n_layers` writes into the stream have equal variance, so the
stream's variance grows linearly with depth. OPT independently arrived at the
same `1/sqrt(2L)` scaling on its output layer.

**But depth-position-dependent scaling is not obviously better.** OLMo-0424
scaled output projections by `1/sqrt(2 * d_model * layer_idx)`, making later
layers smaller. OLMo 2 found this *worse*: it produced faster spike growth, and
switching to a flat `N(0, 0.02^2)` with no depth scaling at all dropped their
spike score from 0.40 to 0.03. This contradicts the general intuition that
deeper layers need smaller init. Treat depth-scaled init as a knob to test, not
a law.

**muP** (Yang et al., "Tensor Programs V") is the principled version. It picks
init variance and per-tensor learning rate as functions of width so that every
layer keeps updating at a non-vanishing, non-exploding rate as width grows. The
Adam rules:

| | Input weights and biases | Output weights | Hidden weights |
|---|---|---|---|
| init variance | `1/fan_in` | `1/fan_in^2` | `1/fan_in` |
| Adam LR | `eta` | `eta/fan_in` | `eta/fan_in` |

Plus, for transformers: use `1/d` rather than `1/sqrt(d)` in the attention
score, and divide the readout logits by the width.

What it buys is zero-shot learning-rate transfer: tune the LR on a narrow proxy
model and it transfers to a much wider one. The paper demonstrates 40M proxy to
6.7B target. MiniCPM confirms it empirically and reports the optimal base LR
settles near 0.01 across 0.036B to 2.1B once muP scaling is applied, an order
of magnitude above the unscaled numbers below.

Worth it if you are running a scaling ladder. Overhead if you are training one
model once.

**For very deep stacks** (100+ layers), see DeepNet (`alpha = (2M)^(1/4)` on the
residual, `beta = (8M)^(-1/4)` on the init of ffn/v/out projections), Admin, and
T-Fixup. These are out of scope for anything under about 50 layers.

## Attention logit growth (QK-norm and softcapping)

The failure mode: attention logits grow during training until softmax saturates
into near-one-hot weights. Zhai et al. (2023) observed low attention entropy
alongside instability and proved a tight lower bound on attention entropy that
decays exponentially in the spectral norm of the attention logits. The bound
explains why "attention logit growth" and "entropy collapse" travel together; it
is not by itself a proof that the growth causes divergence.

Wortsman et al. (2023) reproduced it at small scale by pushing the learning
rate, which is the useful finding: the LR at which a model diverges shrinks as
scale grows, so what looks like a large-model-only problem can be studied on a
small one (their paper studies slow divergence, not fast loss spikes). Their
concrete threshold: every run with attention logits above 1e4
diverged.

Two fixes, and they are complementary rather than competing:

**QK-norm.** Apply RMSNorm to the query and key projections before the dot
product. OLMo 2: "we normalize the key and query projections with RMSNorm before
calculating attention." Two scopes ship, with different costs: per head (Qwen3,
Gemma 3), one gain vector of size `head_dim` for Q and one for K, `2 * head_dim`
parameters per layer; or over the whole projection (OLMo 2, OLMoE), gains of size
`n_heads*head_dim` and `n_kv_heads*head_dim`, 8,192 per layer in OLMo-2-7B. Both
counts were checked against the released checkpoints (`assets/hf-configs/README.md`). First proposed by Henry et al. (2020); ViT-22B brought it to large-scale
training. Used by Chameleon, OLMo 2, Gemma 3, Qwen3. `reference_block.py` has an
implementation (`Cfg(qk_norm=True)`), plus `z_loss` and `softcap`.

**Logit softcapping.** `logit <- cap * tanh(logit / cap)`. Gemma 2 used 50.0 on
attention logits and 30.0 on final logits.

Gemma 2 chose softcapping; Gemma 3 replaced it with QK-norm one generation
later, citing the same papers other labs used to justify QK-norm from the
start. NVIDIA's ablation on an 830M model (Rybakov et al., 2024) found the baseline
converges at 6e-3 and diverges at 8e-3, QK-norm alone survives 40e-3, and
QK-norm plus softcapping (or one norm after the QKV projection instead of before
it) survives 60e-3, the 1.5x. Adding norms after the Proj and FC2 layers did not
beat QK-norm. They intervene at
different points: softcapping bounds the logit going into softmax, QK-norm
bounds the inputs to the dot product that produces it.

**Kimi K2's QK-clip** is a third approach for the Muon optimizer: after each
step, for each head, compute `gamma_h = min(1, tau / max_logit_h)` and scale
that head's Q and K weights by `gamma_h ** 0.5` each (alpha = 0.5), so its max
logit shrinks by `gamma_h`. With `tau = 100` they trained on 15.5T
tokens with zero loss spikes.

## Output logit growth: z-loss

A separate growth phenomenon, at the other end of the model. The softmax
partition function `Z = sum_i exp(x_i)` over the output logits drifts far from
what cross-entropy assumes. The fix, from PaLM:

```
z_loss = coefficient * (log Z)^2      added to the training loss
```

PaLM: "We additionally use an auxiliary loss of `z_loss = 1e-4 * log^2 Z` to
encourage the softmax normalizer `log(Z)` to be close to 0, which we found
increases the stability of training." Wortsman et al. use the same 1e-4;
Chameleon uses 1e-5. The coefficient is not settled.

MoE routers get the same treatment: ST-MoE's router z-loss is the mean squared
log-sum-exp of the router logits with coefficient 1e-3, which stabilized 3 of 3
seeds against 4 of 6 for the baseline with a slight quality gain [ST-MoE §3.3,
Eq. 5]. ST-MoE also keeps the router in fp32.

## Optimizer settings

Near-universal across shipping models:

| Setting | Value | Source |
|---|---|---|
| Optimizer | AdamW | most shipping models. Exceptions: PaLM (Adafactor); Muon for 2D weights in Kimi K2, Moonlight, Kimi Linear and Qwen3.8-Next [Moonlight §3.3; Kimi Linear §5.4.1; Qwen3.8-Next §3.1]. See the Muon subsection below |
| beta_1, beta_2 | 0.9, 0.95 | OPT, Chameleon, OLMo 2 |
| Gradient clip (global norm) | 1.0 | PaLM, OPT, Chameleon, Wortsman. Qwen3.8-Next uses 0.5 and reports clipping never engaging at its optimum [§3.2-3.3] |
| Weight decay | 0.1 in `torch.optim.AdamW`, which already multiplies the decay by the current LR | OLMo 2 describes exactly this (each step multiplies weights by 1 - 0.1 x lr). Llama 3's paper gives no value. Wortsman et al. use independent decay (1e-4, not scaled by LR), which lowered LR sensitivity but is not what PyTorch's AdamW does |
| Precision | bf16 compute, fp32 master and optimizer state | |

**AdamW epsilon is the contested one.** Both Wortsman et al. and OLMo 2 say
decrease it, and they disagree on the target by seven orders of magnitude.
Wortsman found grad RMS can collapse toward the 1e-8 default, at which point
epsilon distorts the update rather than just preventing division by zero, and
recommend 1e-15. OLMo 2 moved from a codebase default of 1e-5 down to 1e-8 and
reports "the gradient norm settles much more quickly and remains permanently
lower." The direction is agreed, the value is not. Reported working values span
1e-8 to 1e-15.

Molybog et al. give the theory: Adam can enter a state where the update vector
has a large norm and is essentially uncorrelated with the descent direction,
and this is more likely in deep models with large batches, which is exactly the
LLM pretraining regime.

**fp16 versus bf16.** bf16 has fp32's exponent range with fp16's mantissa. It
removes the gradient-underflow failure mode that forces fp16 runs to use
dynamic loss scaling (OPT's divergences correlated with "the dynamic loss
scalar crashing to 0"). It does not remove the activation-magnitude-runaway
failure mode, which is what QK-norm and z-loss target. "Just use bf16 and
stability goes away" is not supported by any of these papers.

**Data as a spike cause.** OLMo 2 found batches containing long repeated
n-grams (a span repeated 32 or more times) were often associated with gradient
spikes, and that filtering those documents plus masking the loss on such spans
"results in a clear mitigation, though not complete elimination." OPT removed
some Pile subsets for the same reason. PaLM ruled out data as a *sufficient*
explanation: replaying the same spike-adjacent batches from an earlier
checkpoint did not reproduce the spike, so it is an interaction between the
batch and the parameter state, not either alone. PaLM's mitigation was to
restart 100 steps before the spike and skip 200 to 500 batches.

## Learning rate schedules

**Peak LR by scale**, from published runs:

| Model | Params | Peak LR | Warmup |
|---|---|---|---|
| GPT-3 | 125M | 6.0e-4 | |
| OPT | 125M | 6.0e-4 | 375M tokens |
| OLMo 2 | 7B | 3.0e-4 | 2000 steps |
| OLMo 2 | 13B | 9.0e-4 | 2000 steps |
| OLMo 2 | 32B | 6.0e-4 | 2000 steps |
| OPT | 6.7B | 1.2e-4 | 375M tokens |
| GPT-3 | 175B | 6.0e-5 | |
| Llama 3 | 405B | 8.0e-5 | 8000 steps |
| MiniCPM (muP) | 0.036B to 2.1B | ~1.0e-2 | |

GPT-3 and OPT show peak LR falling with model size. OLMo 2 is non-monotonic on
purpose: they found "higher learning rates universally perform better early on
(as long as they avoid loss spikes)," tested 6e-4 through 3e-3 at 7B (3e-3
spiked unrecoverably), and picked an aggressive 9e-4 for 13B. So the
"bigger model, smaller LR" heuristic is a tendency, not a rule.

**Cosine versus WSD.** Warmup-stable-decay holds the LR constant for most of
training and decays only at the end:

```
lr(t) = (t/W) * eta         t < W          linear warmup
        eta                 W <= t < T     stable
        f(t - T) * eta      T <= t < S     decay
```

MiniCPM found a decay phase of about 10% of total steps is enough. Hägele et al.
(arXiv:2405.18392) found a cooldown of under 20% of steps matches cosine, that it
passes cosine between 10% and 20% at 124M, and that the fraction needed shrinks for
longer runs (5% with a 1-sqrt shape nearly matched cosine at 20B tokens) [§3.2,
Takeaway 3, Fig. 5-6]. Dremov, Hägele, Kosson and Jaggi (arXiv:2508.01483) study the
cooldown's shape as a bias-variance trade-off and report gains from a higher AdamW
beta2 during cooldown. Treat the fraction as a knob to test. The River Valley paper (Wen et al.) explains why
the decay phase drops loss so sharply.

Two reasons to prefer it. You do not have to commit to a total token budget in
advance, since the stable-phase checkpoint is a valid restart point, whereas
cosine has to know its endpoint to shape the decay. And fitting a scaling law
gets much cheaper: branch a short decay off one shared stable run at each
budget of interest, rather than training a separate full run per budget.
Hägele et al. report constant-plus-cooldown "scales predictably and reliably,
similar to cosine."

## Weight decay

**Exclude embeddings.** OLMo 2's mechanism, following Takase et al.: the
Jacobian of a normalization layer is inversely proportional to the norm of its
input, and in early layers the residual stream norm essentially *is* the
embedding norm. Decay embeddings too hard and you shrink that norm, which
inflates the gradients through the first norm layer. Their ablation: spike
score 0.16 with weight decay on embeddings, 0.092 without. OLMo 1 and OLMo-0424
decayed embeddings; OLMo 2 reversed it.

**Exclude norm gains and biases.** Universal in practice (nanoGPT applies decay
only to Linear weights) but no source in this research isolates and tests it
the way OLMo 2 does for embeddings. Inherited convention.

## Muon

`torch.optim.Muon` (present in torch 2.15, where it was checked; on older versions test `hasattr(torch.optim, "Muon")`) orthogonalizes each 2D weight's
momentum with five Newton-Schulz steps. Embeddings, the output head, norms,
biases, routers and gates stay on AdamW [Moonlight §2; Qwen3.8-Next §3.1]. Two
findings matter when switching:

- **Match the update RMS to reuse AdamW hyperparameters.** Moonlight scales each
  matrix's update by `0.2 * sqrt(max(A, B))` and adds AdamW-style decoupled weight
  decay, then reuses AdamW's LR and weight decay [Moonlight Eq. 4]. In PyTorch that
  is `adjust_lr_fn="match_rms_adamw"`; the default (`"original"`) does not transfer
  AdamW LRs. Measured update RMS with that setting: 0.17 (`modern_blocks.py`).
- **The optimum moves anyway.** Qwen3.8-Next refit at 156B-A7B and found optimal
  LR 1.76e-3 and batch 8.4M tokens against 6.8e-4 and 4.2M under its previous
  recipe [§3.2, Fig. 9, Table 10]; batch-size warmup gave no gain and cost 18.8% more
  steps at 10.8B-A0.89B [Fig. 8]. Keep the optimizer through post-training: a Muon
  base reached MMLU 55.7 with Muon SFT and 50.2 with AdamW SFT [Moonlight Table 6].
- **Logit growth.** Muon runs grow attention logits; Kimi K2's QK-clip (above) was
  built for this.

Split fused QKV or gate-up matrices before orthogonalizing; orthogonalizing the
fused matrix couples their scales [Qwen3.8-Next §3.1].

## Gating as a stabilizer

Gates that can push a branch's output toward zero also bound activations.
A per-head sigmoid gate after attention let a 48-layer 1.7B dense model train at
LR 8e-3 where the baseline diverged [Gated Attention Table 2], and in the 15B-A2B MoE
setting cut the maximum activation from 1053 to 94 [Table 4]. Qwen3.8-Next's GatedNorm cut spikes from 32.0 to
3.2 per 10k steps at 3x LR [§3.3]. Both are Qwen self-reports; a small-scale
replication did not see the sink reduction (`references/papers/gated-attn-sinks-replication.md`).

## Stress testing a recipe

Qwen3.8-Next's protocol, cheap enough to run on proxies: train at constant LR at
2x and 4x the optimum, count spikes (loss more than 0.1 above a 201-step rolling
median) and how often gradient clipping engages [§3.3]. A recipe that survives 4x
has margin. Caveat: sink and gate effects may not show in short proxy runs
[Rizwan et al. §VI.C]. For the OLMo 2 spike score on any log, see
`scripts/spike_monitor.py`.

## What to log

Ordered from most lagging to most leading. The first two are what everyone
logs and both are late.

| Signal | Lead time | Source |
|---|---|---|
| Loss | none, it is the failure | |
| Grad norm | near-zero lead | universal |
| Final-layer output L2 norm | early; not quantified | Chameleon: output-norm growth strongly correlated with later divergence, in runs that diverged as late as 20-30% into training |
| Per-linear-layer output L2 norm | same | NVIDIA survey found QKV, output projection and the second FFN layer show >2x growth in a diverging run versus a converging one |
| Max attention logit per head | before saturation | Wortsman: everything above 1e4 diverged |
| Attention entropy | before collapse | Zhai et al. |
| Update/param ratio per tensor | early | WeSaR ties elevated `\|dW\|/\|W\|` on small-norm tensors directly to early-training spikes |
| Residual norm growth exponent | continuous | OLMo 2's `lambda = (1/L) * log(\|x_L\| / \|x_1\|)`, target ~0. OLMo 2 used it to compare initializations; no trained-model threshold is established. `scripts/residual_probe.py` computes it. |

The 1e-3 update-ratio rule of thumb is folklore with no primary source, though
WeSaR gives the general diagnostic a real paper.

Nothing in this literature ties dead or collapsed attention heads to catching
instability early. That one is folklore too; the head-pruning literature is
about post-hoc analysis of trained models, not training-time monitoring.

## Where sources disagree

Do not average these away. If a decision depends on one of them, say which
source you followed.

1. **AdamW epsilon.** Wortsman: 1e-15. OLMo 2: 1e-8. Direction agreed, value
   spans seven orders of magnitude.
2. **z-loss coefficient.** PaLM and Wortsman: 1e-4. Chameleon: 1e-5. OLMo 2's
   body text says 1e-4 and its own summary table says 1e-5.
3. **QK-norm versus softcapping.** Gemma 2 chose softcapping, Gemma 3 reversed
   to QK-norm, and NVIDIA's ablation says they are complementary.
4. **Depth-scaled init.** GPT-2, OPT, DeepNet and T-Fixup all scale init by
   depth. OLMo 2 tested a depth-position-scaled variant, found it worse than a
   flat 0.02, and shipped flat.
5. **Peak LR versus model size.** GPT-3 and OPT decrease it monotonically.
   OLMo 2 deliberately does not.
6. **bf16 and stability.** bf16 removes fp16's underflow failure mode. It does
   not remove the activation-runaway one, which is a different problem.
7. **NoPE in global layers.** Kimi Linear reports better long context with NoPE
   in its MLA layers [Table 5]; Qwen3.8-Next saw endless generation after
   post-training with NoPE global layers [§2.1.1].
8. **Widening the residual stream.** Hyper-connections were stable at 7B [HC §4.3]
   and unstable at 27B without mHC's doubly stochastic constraint [mHC §3.1].
9. **QK-norm and long context.** QK-norm stabilizes training; Cohere reports it
   hurt long-context retrieval at 8B [Rope to Nope §2.2.1; numbers in an image,
   not read].
10. **Init std.** 0.02 in most models; DeepSeekMoE, V2 and V3 use 0.006
    [DeepSeekMoE §4.1.3; DeepSeek-V3 §4.2].
11. **Batch-size ramps.** Llama 3 and DeepSeek-V2/V3 ramp batch size during the run;
    Qwen3.8-Next found a ramp useless under Muon. The optimizers differ, so this is
    not a contradiction yet.
