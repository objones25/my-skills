# Attention sinks: StreamingLLM, gpt-oss learned sinks, and the abstention/filtering account

Three sources, digested together because they answer one question: softmax attention must put its weight somewhere, so what should a head do when nothing is relevant?

1. Efficient Streaming Language Models with Attention Sinks. Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, Mike Lewis (MIT, Meta AI, CMU, NVIDIA). arXiv:2309.17453 (v4, 7 Apr 2024; v1 29 Sep 2023). ICLR 2024 (arXiv comment). Checked: full text of §3-4 from the arXiv HTML.
2. gpt-oss-120b & gpt-oss-20b Model Card. OpenAI. arXiv:2508.10925 (v1, 8 Aug 2025). Checked: §2 (architecture and pretraining) only. Vendor model card; it describes the design without ablations.
3. Abstention and Noise Filtering: Two Missing Primitives of Softmax Attention. Richard Zhe Wang (St. John Fisher University). arXiv:2609.22005 (v2, 29 Sep 2026; v1 18 Sep 2026). Preprint, single author, posted after June 2026: treat with extra care. Checked: abstract, §1-2, §4 results text and Tables 1-2 from the PDF; appendices not read.

## The idea

- StreamingLLM: trained LLMs dump a lot of attention on the first few tokens regardless of content, because those positions are visible to every later query. Evicting them from a sliding-window cache makes perplexity explode (Llama-2-13B: 5158 with a 1024 window vs 5.40 when the first 4 tokens are kept). Keep 4 "sink" tokens plus a rolling window and streaming works to 4M tokens without fine-tuning [SLLM Table 1, §3.2, Fig. 5]. A dedicated learnable sink token prepended in pretraining makes one token enough [§3.3, Table 3].
- gpt-oss: "Each attention head has a learned bias in the denominator of the softmax, similar to off-by-one attention and attention sinks, which enables the attention mechanism to pay no attention to any tokens" [gpt-oss §2.2]. Equivalently, a per-head phantom key with a learned logit and zero value.
- Wang (2026): softmax lacks two things, abstention (outputting nothing) and noise filtering (suppressing interference carried in value vectors). A sink logit supplies abstention; a gate computed from each value supplies filtering; a query-computed output gate (Qiu et al.'s gated attention) supplies abstention but cannot filter individual values. In 10M-350M models the gain from gating was mostly abstention at 10M and about half filtering at 350M; the two add with small overlap [Wang Abstract, §4.4, Table 2].

## Notation

| Symbol | Meaning |
|---|---|
| x+y cache | StreamingLLM cache with x initial (sink) tokens and y most recent tokens |
| SoftMax₁ | Off-by-one softmax, e^{x_i} / (1 + Σ_j e^{x_j}); equivalent to a zero key and zero value prepended [SLLM Eq. 2] |
| Sink token | Learnable placeholder token prepended to every pretraining sample [SLLM §3.3] |
| Sink logit (phantom key) | Per-head learned scalar s_h added as an extra softmax term with value 0, so weights over real tokens can sum to less than 1 [gpt-oss §2.2; Wang §1] |
| Norm ratio | Wang's ratio of the value-read norm at position 0 to elsewhere; well below 1 indicates a crushed sink value [Wang §4.3] |

## The methods

### StreamingLLM at inference

1. Keep the KV of the first 4 tokens permanently and a rolling window of recent tokens [§3.2].
2. Assign positions by place in the cache, not in the original text; cache keys before RoPE and re-rotate each step [§3.2].

### Training-time fixes

- SLLM: prepend a learnable sink token to every sample (tested at 160M) [§3.3, §4.2].
- gpt-oss: per-head learned softmax-denominator bias [§2.2].
- Gated attention: per-head sigmoid output gate (`gated-attention.md`).

## Hyperparameters

| Setting | Value | Source |
|---|---|---|
| Sink tokens at inference | 4 (1-2 not enough; 8 adds little) | SLLM Table 2, §4.4 |
| Sink-token pretraining test | Pythia-160M recipe, batch 256, 143,000 steps, deduplicated Pile | SLLM §4.2 |
| gpt-oss attention | alternating 128-token banded window and dense layers, 64 query heads of dim 64, 8 KV heads, RoPE, YaRN to 131,072 on dense layers | gpt-oss §2.2 |
| gpt-oss residual / MoE | width 2880, pre-norm RMSNorm; 128 (120b) or 32 (20b) experts, top-4, softmax over selected experts | gpt-oss §2.2 |
| Sink-logit init / LR | not given in the model card | |

## Key results

- Llama-2-7B streaming PPL on PG19: 0+4096 window 3359.95; 1+4095 11.88; 4+4092 9.59; 8+4088 9.54 [SLLM Table 2].
- Replacing the first 4 tokens with "\n" tokens restores PPL nearly as well (5.60 vs 5.40 for Llama-2-13B), so position, not content, makes the sink [SLLM Table 1].
- 160M pretraining, streaming PPL with 0/1/2/4 sink tokens: vanilla 27.87 / 18.49 / 18.05 / 18.05; SoftMax₁ ("zero sink") 29214 / 19.90 / 18.27 / 18.01; learnable sink token 1235 / 18.01 / 18.01 / 18.02 [SLLM Table 3]. Zero-shot accuracy unchanged or slightly better with the sink token [Table 4].
- Up to 22.2x per-token decode speedup over sliding window with recomputation [SLLM §4.5].
- Wang, paired validation-loss gains (nats): sink logit over baseline 0.0185 at 10M, 0.0052 at 124M; filtering increment of a projection gate over the sink logit rises from about 0 at 10M to 0.0089 at 350M [Wang §4.2, Table 1]. At 124M the baseline puts 6.0% of attention on position 0 (norm ratio 0.42); with a sink logit 0.5% (norm ratio 0.99) [§4.3]. Wang reports gpt-oss-20b putting 34-48% of attention mass on its phantom [§4.3, App. E].

## Gotchas when reproducing

- StreamingLLM does not extend context: it lets a model run indefinitely on its recent window. Information outside the window is gone [§4.4 Table 6 shows larger caches do not always help].
- Off-by-one softmax alone did not remove the need for real sink tokens in the 160M test; a learnable sink token did [SLLM Table 3].
- Wang's effects are hundredths of a nat on models up to 350M, three seeds; one 350M abstention estimate ranges 0.0013-0.0271 across seeds [§4.2].

## Connections

- Components: three ways to give a head a "no-op": a learnable sink token (input-level), a per-head sink logit (gpt-oss), or a query-dependent output gate (Qwen gated attention). Only the gate also adds the nonlinearity Qiu et al. credit for most of the PPL gain.
- Stability: sinks travel with massive activations; the skill's residual probe skips position 0 for that reason. Gated attention reports both shrinking (`gated-attention.md`); a later small-scale test did not reproduce the sink reduction (`gated-attn-sinks-replication.md`).
- Long context: sink-dependent models degrade more under YaRN [GatedAttn Table 5]; sliding-window layers need the sink tokens kept in the window (gpt-oss uses a 128-token band with learned sinks).
