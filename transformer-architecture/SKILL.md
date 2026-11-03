---
name: transformer-architecture
description: Design, size, review, implement, debug, and write up decoder-only transformer LLMs, dense or mixture-of-experts. Use whenever the work is how a transformer is built or trained rather than prompted, including choosing a config (width, depth, heads, GQA or MLA, experts, d_ff, vocab, context, RoPE), computing parameter, active-parameter, FLOP, KV-cache or VRAM budgets, writing or fixing RoPE, RMSNorm, GQA, KV-cache, MoE router, MLA, gated-attention or attention-sink code, diagnosing loss spikes, divergence, NaNs or a model that will not learn, choosing AdamW or Muon, planning a pretraining run, setting up SFT, DPO or GRPO-style RL, or defending architecture choices. Trigger on questions like "how big should I make this", "should this be an MoE", "why is my loss spiking at step 4000", "how much VRAM to train this", "garbage after the first token", "what beta for DPO", or "why do my GRPO answers keep getting longer", even when the user never says architecture.
compatibility: Scripts need Python 3.10+ and torch (tested on 2.15.0, CPU, machine not recorded; 182 tests passed on 2026-11-03 on macOS arm64; torch.optim.Muon checked on 2.15). residual_probe.py --model also needs transformers. Run pytest from the skill root to re-check scripts, presets and the claims tied to the reference table.
metadata:
  version: 1.2.0
  verified-against: "torch 2.15.0, transformers 5.22.1"
---

# Transformer architecture

Two jobs. **Sizing**: given constraints, pick a config and defend it with
arithmetic. **Implementation**: write or repair the blocks, and catch the bugs
that pass every shape check.

Both have the same failure mode, which is confident hand-waving. Parameter
counts estimated in prose are wrong often enough to matter, and RoPE bugs
survive review because the tensors are the right shape. So: compute the numbers
with the scripts, and verify the code with the property tests. The reasoning
matters, but it goes on top of real output, not instead of it.

## Rules

1. Every number comes from a script run, a config file, or a located source
   (`[DeepSeek-V2 App. D.1]`, `[OLMo 2 §3.2]`). Parameter counts are never estimated
   in prose. `[inference]` marks synthesis no source states.
2. Library facts (torch, transformers, PyTorch's Muon) come from the installed
   package (`inspect.signature`, `inspect.getsource`) or current docs, not memory.
   This version was checked against torch 2.15.0 and transformers 5.22.1.
3. When sources disagree, say so and say which one you follow; the known
   disagreements are listed at the end of `references/stability.md`. Lab papers
   benchmarking their own models are self-reports; prefer single-variable
   ablations inside a paper over cross-lab headline comparisons.

## What is here

```
scripts/config_budget.py        params (total and active), FLOPs, KV cache, VRAM, validation;
                                dense, MoE, MLA, sliding window; 13 presets exact to the checkpoint
scripts/reference_block.py      canonical RMSNorm/RoPE/GQA/SwiGLU/cache + property tests, --impl
scripts/modern_blocks.py        gated attention, sinks, sliding window, MoE routing, MLA, Muon
scripts/post_training_losses.py DPO/IPO/SimPO, GRPO/Dr. GRPO/DAPO/GSPO, with property tests
scripts/spike_monitor.py        spike score, grad-norm growth, attention-logit flags from a log
scripts/residual_probe.py       per-block residual-stream analysis (wider vs deeper)
```

| Read | When |
|---|---|
| `references/components.md` | the dense block: what each piece is, why it won, the bug it invites |
| `references/moe-and-attention-variants.md` | MoE routing and balancing, MLA, gated attention, sinks, sliding/hybrid layers, Muon |
| `references/reference-configs.md` | ~35 shipping configs (dense and MoE), ratios, consensus vs spread |
| `references/stability.md` | symptom-to-cause table, init, norms, LR, optimizers, what to log |
| `references/post-training.md` | SFT, DPO and variants, GRPO-family RL: recipes, failure modes, evaluation |
| `references/sources.md` | which paper supports which claim, and how it was checked |
| `references/papers/README.md` | 16 digests of 2024-2026 papers with located claims (MoE, MLA, hybrids, gating, Muon, RL) |

Read only the file for the part you are working on. `examples/` has six worked
answers with real script output (`examples/README.md`); `assets/` has an
architecture write-up template, a run plan, and real `config.json` files;
`evals/evals.json` is the regression set.

## Sizing a model

### Get the four constraints first

Almost every config question is underdetermined until you have these. Ask if
they are missing, or state your assumption and move.

| Constraint | Why it binds |
|---|---|
| Training compute or wall-clock budget | Sets total parameters times tokens. Everything downstream is a split of this. |
| Inference target: batch, context, latency, VRAM | Sets the KV cache budget, which is what actually decides GQA groups and often decides depth. |
| Token budget available | With compute, fixes the params/tokens split. A fixed dataset caps useful model size. |
| Vocabulary and languages | Sets vocab size, which for a small model is most of the parameters. |

If the person is training on one GPU for a weekend, that is the binding
constraint and it should drive the answer, not a scaled-down copy of Llama.

### Then run the budget

Do this before writing any prose about the config. Hand-estimating parameters
is where the wrong answers come from.

```bash
python scripts/config_budget.py --d-model 1024 --n-layers 24 --n-heads 16 \
    --n-kv-heads 4 --d-ff 2752 --vocab 32000 --seq-len 2048 --tie \
    --tokens 10e9 --show-work
```

It prints the parameter breakdown by component, derived ratios against the
shipping consensus, FLOPs per token, KV cache per token and at batch, optimizer
and activation memory, GPU-hours, the Chinchilla comparison, and a validator
that flags anything outside the observed range. `--show-work` prints the
arithmetic so it can be checked by hand. `--preset <name>` loads a real model
for comparison; `--from-hf-config` reads a `config.json`.

The self-test reproduces every preset's checkpoint total exactly (dense, MoE and
MLA), and checks each lab's published active count against three counting
conventions. Excerpt:

```
$ python scripts/config_budget.py --self-test
preset                      computed          published     delta
llama3-8b              8,030,261,248      8,030,261,248   0.000% ok
olmo2-7b               7,298,617,344      7,298,617,344   0.000% ok
qwen3-30b-a3b         30,532,122,624     30,532,122,624   0.000% ok
gpt-oss-20b           20,914,757,184     20,914,757,184   0.000% ok
```

### Dense or MoE first

Total parameters set memory, checkpoint size and optimizer state; active
parameters set FLOPs per token. An MoE is cheaper per token, not per byte: Qwen3-30B-A3B
needs 63% of Qwen3-8B's training FLOPs per token and 3.7x its memory
(`examples/03-moe-or-dense.md`). Choose MoE when FLOPs bind and memory is pooled
across devices; then the routing and balancing choices are in
`references/moe-and-attention-variants.md`. Everything below applies to the
attention and dense parts of either.

### Decide in this order

Each choice constrains the next, so the order is not arbitrary.

```
budget ──► vocab ──► d_model, n_layers ──► head_dim, n_heads ──► n_kv_heads
                          │                                          │
                          └──► d_ff ◄────────────────────────────────┘
                                 │
       context length, rope_theta ┴──► tying ──► norm, activation, init
```

**1. Vocabulary.** 32k for English-only, 100k to 130k for multilingual or code,
up to 262k if following Gemma. Round to a multiple of 64. Check the embedding
share the budget script reports: above 25% you are building a lookup table with
a transformer attached, and either the vocab is too big or the model is too
small.

**2. `d_model` and `n_layers`.** These trade off at fixed parameters. The
observed aspect ratio `d_model / n_layers` runs 19 to 128 in the reference table,
and above a billion parameters it still spans 33 to 128: Llama and OLMo sit at
100-128, Qwen, Gemma and SmolLM at 55-90. Width is cheaper per FLOP (bigger GEMMs,
better tensor-core utilization, parallelizes across devices) and depth buys
serial computation. Both `d_model` and `d_ff` should be multiples of 128.

If a trained checkpoint exists, do not guess: run
`scripts/residual_probe.py --model <path>`. It reports, per block, the cosine
similarity between what the block received and what it returned, the relative
size of its write to the residual stream, and the stream's norm across depth.
Blocks in the last third with cosine above 0.995 and a relative write below
0.05 are close to identity functions, which is the concrete argument for moving
parameters from depth into width. Confirm with a matched-parameter val-loss run
before committing, because the probe is evidence, not proof.

**3. `head_dim` and `n_heads`.** `head_dim` is 128 almost everywhere above a
billion parameters, 64 below, 256 in the Gemma family. Pick `head_dim` first,
then `n_heads = d_model / head_dim`. Decoupling them is legal and deliberate
in Qwen3 and Gemma, but it means attention capacity no longer tracks backbone
width, so do it on purpose or not at all.

**4. `n_kv_heads`.** Mostly an inference decision, but not free in quality.
Compute the KV cache at your real serving batch and context, then pick the
group size that fits. The "~0.1 points" folklore comes from uptraining T5 and
scoring with ROUGE; DeepSeek-V2's from-scratch 7B ablation (1.33T tokens, layers
adjusted to match parameters) scored MMLU 45.2 for MHA, 41.2 for GQA-8 and 37.9
for MQA [DeepSeek-V2 App. D.1, Table 8], and MLA beat MHA with 4-14% of its cache
[Table 9]. If the cache budget is tight, MLA or sliding-window / hybrid layers are
alternatives to more aggressive grouping. 4x is modal; the range is 1x (OLMo 2 skips GQA entirely,
even at 13B) to 8x (Llama-3-70B). `n_heads` must be divisible by `n_kv_heads`.

**5. `d_ff`.** Start from `round_to_256( (2/3) * 4 * d_model )`, which holds
parameters fixed against a plain 4x MLP given SwiGLU's three matrices. At
`d_model = 4096` that gives 11008, which is Llama-2-7B's actual value. Shipping
models span 2.67x to 6.0x `d_model`, so this is a starting point.

**6. Context length and `rope_theta`.** Two strategies, and reading theta alone
misleads. Either train natively long with a large base (Llama 5e5, Qwen 1e6),
or keep 1e4 and add explicit scaling later (Mistral plus sliding window,
DeepSeek plus YaRN x40). A third pattern drops RoPE in the global layers and
gets local position from RoPE'd sliding-window or linear-attention layers (Cohere,
Kimi Linear); Qwen3.8-Next reports endless generation after post-training with
NoPE global layers, so the sources disagree (`references/papers/rope-to-nope.md`).
Do not use 1e4 with an 8k+ native context and no scaling plan.

**7. Tying.** Every model in the reference table with an embedding share above
21% ties. Below that, a plain size rule (tie under ~5B parameters) predicts 28 of
32 configs, better than a 10% share threshold (25 of 32). The exceptions are
family choices: Gemma ties at 9B and 12B (shares 9.9% and 8.5%), while OLMo2-1B
and Phi-3-mini untie. Follow the convention of the family you are copying.

**8. Everything else** has a default (next section). Deviate only with a reason.

### Defaults that need a reason to change

| Choice | Default | Why |
|---|---|---|
| Normalization | RMSNorm, eps 1e-5 or 1e-6 | Same convergence, 7-64% faster, fewer parameters |
| Placement | pre-norm | Trains without warmup; post-norm needs it |
| Position | RoPE | Relative distance exactly inside the QK dot product, zero parameters |
| MLP | SwiGLU, no bias | With GEGLU, best log-perplexity at matched parameters, and best on most (not all) fine-tuning tasks in Shazeer 2020 |
| Attention | GQA, FlashAttention | Cuts the cache 4x; costs quality from scratch (step 4). Candidate at 1B+: a per-head sigmoid output gate (`moe-and-attention-variants.md`) |
| Linear biases | none | Universal since Llama, outside Qwen2's QKV bias |
| Init | `N(0, 0.02)`, output projections scaled `1/sqrt(2*n_layers)` | Keeps residual variance flat with depth |
| Precision | bf16 compute, fp32 master and Adam state | 16 bytes per parameter |
| Optimizer | AdamW, betas (0.9, 0.95), grad clip 1.0, wd 0.1 excluding embeddings | Excluding embeddings cut OLMo 2's spike score from 0.16 to 0.092. Muon for 2D weights (AdamW for the rest) is a shipping alternative; refit the LR |
| Schedule | WSD, ~10% of steps in decay | Same quality as cosine without pre-committing a token budget |

Above roughly a billion parameters, add QK-norm. Per head (Qwen3, Gemma 3) it
costs `2*head_dim` parameters per layer; over the full projection (OLMo 2, OLMoE)
it costs `n_heads*head_dim + n_kv_heads*head_dim`. Either is cheap insurance against the
attention logit growth that ends runs, but not the only one: DeepSeek-V3 reports
no irrecoverable spikes without mentioning QK-norm, and Qwen3.8-Next credits gated
residuals plus Muon. Pick one bound and say which. See `references/stability.md`.

### Then check the answer

Compare the config against `references/reference-configs.md`. A config inside
the observed spread needs no defense. One outside it needs a reason you can say
out loud, and "it seemed reasonable" is not one.

## Implementing or fixing a component

`scripts/reference_block.py` is the canonical implementation and the test
suite. After porting anything into your own code, run the property tests
against your version:

```bash
python scripts/reference_block.py --verify   # RoPE relative-distance, GQA layout,
                                             # cache == full forward, causality, init,
                                             # QK-norm, z-loss, and 6 bug detectors
python scripts/reference_block.py --verify --impl my_impl.py   # same tests on YOUR
                                             # apply_rope / repeat_kv / RMSNorm
python scripts/reference_block.py --shapes   # every tensor shape on one toy example
python scripts/reference_block.py --bug gqa-repeat   # explain one failure mode
```

The bugs worth knowing before you write the code, because every one of them
produces correctly-shaped tensors and a model that trains:

| Bug | Symptom |
|---|---|
| RoPE applied before splitting heads | Rotation straddles head boundaries. Loss a few percent worse, blamed on data. |
| Interleaved vs halves RoPE pairing mismatch | Trains fine from scratch, silently incompatible with any external checkpoint. |
| `x.repeat()` instead of expand-reshape in `repeat_kv` | Exactly a query-head permutation, so from-scratch training is unharmed. Breaks checkpoint interop, `enable_gqa=True`, and fused kernels, and wastes activation memory. Say this precisely; calling it a quality bug is wrong. |
| `is_causal=True` during incremental decode | Length-1 query against length-t keys attends only to token 0. Prefill perfect, generation garbage. |
| RMSNorm eps outside the sqrt, or squaring in bf16 | On an all-zero row, `x*rsqrt(ms)+eps` is NaN in the forward pass; `x/(sqrt(ms)+eps)` is finite forward but NaN in the backward pass, so forward-only tests miss it. bf16 squaring adds zero-mean error of up to ~0.2% to the norm per site; upcast first. |
| `d_ff = 4*d_model` with a gated MLP | Silently 50% more MLP parameters than intended. |
| Rotating keys on the way out of the cache, or `cos[:T]` with a cache | Positions restart at 0 every decode step. |
| Tying, then calling `self.apply(init_fn)` | The shared tensor is drawn twice; registration order decides which draw survives. |
| `pad_token == eos_token` with padding masked | Model never learns to stop. |

Two of these are the reason the table says "silently": the GQA layout and the
RoPE pairing both train perfectly well and only fail at a boundary. When
reporting a bug, say which kind it is. A reviewer who calls a relabelling a
quality regression has cost the reader a day.

For MoE layers, MLA, gated attention, sinks and sliding windows, the same job is
done by `scripts/modern_blocks.py --verify` (sparse dispatch against a dense
oracle, absorbed MLA decode against the full forward, bias-based balancing, and
four bug detectors: `--bug moe-bias-in-weights`, `mla-rope-on-latent`,
`moe-topk-renorm-mismatch`, `sink-added-to-logits`). Test decode against forward
for every attention variant you port; training never exercises the decode path.

`references/components.md` has the reasoning behind each component, the
equations with every symbol defined, and the ablation numbers that justify the
choice.

## When training goes wrong

Go to the symptom table at the top of `references/stability.md` rather than
guessing. The common ones:

- **Loss flat at `ln(vocab)`.** Nothing is learning. Check gradient flow, tying,
  and LR before anything else. `reference_block.py --verify` includes a
  gradient-flow check you can adapt.
- **Spikes that recover.** Usually a batch interacting with parameter state,
  not bad data alone. PaLM's mitigation is to restart 100 steps back and skip
  200 to 500 batches.
- **Spikes that do not recover.** Attention logit growth. Wortsman et al. found
  every run above 1e4 attention logits diverged. QK-norm, logit softcapping, or
  a lower LR.
- **NaN in bf16.** Not a range problem, since bf16 has fp32's exponent range.
  Activation magnitude runaway, which is a norm-placement and init question.
- **Garbage after the first generated token.** The KV cache causal mask.

Log the leading indicators, not just loss and grad norm, which are both late.
Chameleon found final-layer output-norm growth strongly correlated with later
divergence, in runs whose divergences appeared as late as 20-30% into training.
`scripts/spike_monitor.py <log>` reads a CSV or JSONL log and reports OLMo 2's
spike score (values 7 sigma from the rolling mean of the previous 1,000), slow
grad-norm growth, attention logits past 1e4, and output-norm growth. Early in a
run use `--window 200`: the default window still contains the loss decay and
misses real spikes (`examples/04-loss-spike-triage.md`).

## SFT, DPO and RL

`references/post-training.md` has the recipes; `scripts/post_training_losses.py`
implements each loss next to the property that distinguishes it. The things that
come up every time:

- **DPO learning rates are 5 to 20 times lower than SFT** at the same scale.
  OLMo 2 7B: SFT 1e-5, DPO 1e-6. Tulu 3 8B: 5e-6 and 5e-7. This is the most
  common thing to get wrong.
- **`beta = 0.1`** is the near-universal starting point.
- **Do not over-train SFT first.** Zephyr found 1 SFT epoch then 3 DPO epochs
  optimal, and that more SFT epochs caused a DPO regression. Skipping SFT
  entirely gave MT-Bench 4.76 versus 7.00.
- **Log both raw logprobs, not just the margin.** The chosen response's
  log-probability routinely falls during DPO while the margin looks healthy.
  This is likelihood displacement and it has driven a safety-tuned model's
  refusal rate from 74% to 33%.

- **GRPO's loss normalization is a design choice, not a detail.** Averaging each
  response's tokens by its own length makes long wrong answers cheap (length grows,
  accuracy does not); dividing advantages by the group std over-weights near-solved
  prompts. Dr. GRPO and DAPO fix these; GSPO moves the ratio to the sequence level
  for MoE stability. Check what your library does (`examples/05-grpo-answers-getting-longer.md`).

Preference tuning does not add knowledge. If the model is factually wrong, the
SFT data is wrong.

## Presenting the answer

Lead with the config table, then the budget table with real numbers from the
script, then the reasoning. Not the reverse.

Name the two or three choices that are actually load-bearing and say what would
change them. Most of a config is convention; a few numbers carry the design.
Being explicit about which is which is the difference between a recommendation
and a list.

Show the arithmetic for anything a reader might want to check. `--show-work`
exists for this.

Keep the numbers consistent across the whole answer. If tokens-per-parameter
appears in two places, both must be for the same config, and both must match
what the script printed. Two different values for the same quantity, or a
number that contradicts the config you just recommended, undoes the credibility
that computing it bought.

Say when something is genuinely unsettled. `d_ff/d_model` has no consensus
value, AdamW epsilon spans seven orders of magnitude in the literature, and
Gemma reversed its own attention-stability choice between generations. Stating
a contested number as settled is worse than saying it is contested.

Cite the paper for the claim, not the field. `references/sources.md` maps
claims to sources. When a number came from a config file rather than a paper,
say so.

## What not to do

Do not estimate parameter counts in prose. Run the script.

Do not recommend a config without the inference constraint. GQA group count and
often depth are decided by the KV cache at the real serving batch and context,
and a config chosen without that number is a guess.

Do not compare MoE "active parameters" across labs without converting: Qwen
counts both embedding tables, gpt-oss counts only the unembedding.
`config_budget.py --self-test` prints all three conventions.

Do not copy Llama's ratios down to 100M parameters. Small models diverge
sharply from large ones: aspect ratios drop to 19-45, embedding share rises
past 25%, tying becomes mandatory, and tokens-per-parameter runs into the tens
of thousands rather than 20.

Do not present Chinchilla's 20 tokens per parameter as the target. It optimizes
training cost. Everything small ships deliberately overtrained because
inference cost dominates: Qwen3-0.6B ran 60,000 tokens per parameter.

Do not assert the folklore. Three things that sound authoritative have no
primary source: the 1e-3 update-ratio rule, dead attention heads as an early
instability signal, and the empirical case for excluding norms from weight
decay. `references/sources.md` lists what is and is not supported.

## Extend this skill

1. A new paper: digest it in `references/papers/` with located claims, a notation
   table and a "disagreements" section; add a row to `references/papers/README.md`
   and to `references/sources.md` (with how it was checked); link it from the
   reference file whose advice it changes.
2. A new shipping model: add its `config.json` to `assets/hf-configs/`, check its
   total three ways (safetensors `total_size`, meta-device instantiation,
   `config_budget.py --from-hf-config`), and add a preset plus a
   `PUBLISHED_TOTALS` entry only when all three agree.
3. A new component: implement it in `scripts/modern_blocks.py` with a property
   test and, if it has a silent failure mode, a bug detector and `--bug` text.
4. Log the change in `CHANGELOG.md` and add an eval to `evals/evals.json` when a
   new kind of question shows up. `tests/test_skill_structure.py` fails on
   broken paths, unregistered arXiv ids, untested scripts and em dashes.
