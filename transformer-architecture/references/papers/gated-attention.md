# Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free

Zihan Qiu, Zekun Wang, Bo Zheng, Zeyu Huang, Kaiyue Wen, Songlin Yang, Rui Men, Le Yu, Fei Huang, Suozhi Huang, Dayiheng Liu, Jingren Zhou, Junyang Lin (Qwen Team, Alibaba; Edinburgh, Stanford, MIT, Tsinghua). arXiv:2505.06708 (v1, 10 May 2025, the only version). The arXiv page names no venue. Checked: full text (HF paper.md rendering of the arXiv source), all tables and appendix A.

## The idea

Multiply the output of scaled dot-product attention (SDPA), per head, by a sigmoid gate computed from the current token's pre-norm hidden state, before the output projection W_o. Across about 30 variants on a 15B-total / 2.54B-active MoE and a 1.7B dense model, this position (called G1) gave the lowest perplexity and best benchmarks, at under 2% wall-time cost [GatedAttn §3.1, Table 1]. The authors give two reasons: the gate puts a nonlinearity between W_v and W_o, which otherwise collapse into one low-rank linear map; and the gate scores are sparse and depend on the query token, which lets a head output nearly nothing instead of dumping attention on the first token. With the gate, the first-token share of attention fell from 0.467 to 0.048 and the largest hidden activation from 1053 to 94 [GatedAttn Table 4]. Gated dense models also tolerated a higher peak learning rate and showed fewer loss spikes [GatedAttn Table 2, Fig. 1 right].

## Notation

| Symbol | Meaning |
|---|---|
| X | Attention-layer input after pre-normalization, shape (n, d_model); the gate reads this [GatedAttn §2.2 footnote 1] |
| n | Sequence length |
| d_model | Residual width |
| d_k | Head dimension (128 in all experiments) [Table 1 caption] |
| q, k | Number of query heads and key-value heads (baseline q = 32, k = 4, i.e. GQA) [Table 1] |
| W_Q, W_K, W_V, W_o | Query, key, value and output projections; W_o has shape (q·d_k, d_model) |
| Y | The tensor being gated; at G1 it is the per-head SDPA output, shape (n, q, d_k) |
| W_θ | Gate projection. Elementwise G1: shape (d_model, q·d_k). Headwise G1: (d_model, q) |
| σ | Gate activation; sigmoid unless stated |
| Y' | Gated output, Y' = Y ⊙ σ(X W_θ) [Eq. 5] |
| S^k_{ij} | Attention probability from token i to token j in head k [Eq. 6] |
| G1 to G5 | Gate positions: G1 after SDPA, G2 after V projection, G3 after K, G4 after Q, G5 after W_o [§2.2, Fig. 1] |
| F-Attn | Mean attention probability on the first token, averaged over heads and layers [Table 4 caption] |
| M-Act | Mean over layers of the rounded maximum hidden-state activation per layer [Table 4 caption] |

## The method

### 1. Pick the gate position and form

Default: head-specific, multiplicative, sigmoid gate at G1 [§2.2]. The score has shape (n, q, d_k) for elementwise gating or (n, q) for headwise gating, one scalar per head per token. For the 15A2B MoE the elementwise gate adds 201M parameters and the headwise gate 1.6M [Table 1 rows 5, 10].

### 2. Compute

For each layer: compute Q, K, V and run SDPA as usual, giving per-head outputs Y. Compute g = sigmoid(X W_θ) from the same pre-norm input X. Multiply Y by g (broadcast over d_k for headwise), concatenate heads, apply W_o [Eq. 5, §2.2]. Gate initialization is not given in the paper.

### 3. Keep the parameter count fair

In the dense experiments the FFN width is reduced so gated and baseline models have equal parameters [§3.2.2, Table 2 caption]. In the MoE table the comparison is against baselines that add as many or more parameters (k = 8 KV heads, q = 48 query heads, 4 more experts) [Table 1 rows 2-4].

### Why it works (the paper's analysis)

- Low rank: per head, output = Σ_j S_ij X_j (W_V W_O), and W_V W_O has rank at most d_k < d_model. A nonlinearity at G1 or G2 breaks this; one at G5 (after W_o) does not, and G5 gave no gain [§4.1, Eq. 6-8, Table 1 row 9]. Per-head RMSNorm at G1, which adds almost no parameters, also cut PPL from 6.026 to 5.847 [Table 3 row 5].
- Sparsity: mean G1 elementwise gate score is 0.116, and most scores sit near 0 [Table 4, Fig. 3]. Forcing the gate into [0.5, 1] with NS-sigmoid(x) = 0.5 + 0.5·sigmoid(x) kept the nonlinearity but lost most of the gain (PPL 5.900 vs 5.761) and brought the sink back (F-Attn 0.451) [§4.2, Table 4 row 7].
- Query dependence: gating V (G2) uses each past token's state, not the query's, and left the sink in place (F-Attn 0.297) [§4.2-4.3, Table 4 row 5].
- Massive activations are not required for sinks: G2 gating cut M-Act to 125 but F-Attn stayed 0.297 [§4.3].

## Hyperparameters

| Setting | Value | Source |
|---|---|---|
| MoE model | 15B total, 2.54B active ("15A2B"), 128 experts, top-8 softmax routing, fine-grained experts, global-batch load-balance loss, z-loss | §3.1 |
| Attention | GQA, q = 32, k = 4, d_k = 128 | §3.1, Table 1 |
| Sequence length | 4096 | §3.1 |
| MoE schedule | warmup 1k steps to 2e-3, cosine to 3e-5, batch 1024 sequences, 100k steps (400B tokens) | §3.2.1, Table 1 caption |
| Dense 1.7B, 400B tokens | 28 layers, peak LR 4e-3, batch 1024 | §3.2.2, Table 2 |
| Dense 1.7B, 3.5T tokens | 28 layers, peak LR 4.5e-3, batch 2048 | §3.2.2, Table 2 |
| Dense 1.7B, 48 layers | 400B tokens at batch 1024 (LR 4e-3 and 8e-3); 1T tokens at batch 4096 (LR 5.3e-3 and 8e-3) | Table 2 |
| Optimizer | AdamW, "other hyperparameters follow the default values" | §3.1 |
| Long-context stage | RoPE base 10k to 1M, 80B tokens at 32k, then YaRN to 128k | §4.4 |
| Gate init, weight decay on W_θ, dense model widths, warmup for dense runs | not given in the paper | |

## Key results

- MoE 15A2B, 400B tokens: baseline PPL 6.026, MMLU 58.79, GSM8k 52.92. G1 elementwise: PPL 5.761, MMLU 60.82, GSM8k 55.27. G1 headwise (1.6M params): PPL 5.792, MMLU 60.05. Adding 4 experts (400M params): PPL 5.964 [Table 1 rows 1, 5, 10, 4].
- Head-shared gating (score averaged over heads) is weaker: PPL 5.801 and F-Attn 0.301 vs 0.048 for head-specific [Table 1 row 12, Table 4 row 4].
- Multiplicative beats additive (5.761 vs 5.821); sigmoid beats SiLU (5.761 vs 5.822) [Table 1 rows 5, 14, 15].
- Dense 1.7B, 3.5T tokens: PPL 6.180 to 6.130, HumanEval 34.15 to 37.80, MMLU 59.10 to 59.61 [Table 2 rows 3-4]. Benchmark gains at this scale are small apart from HumanEval.
- Stability: 48-layer baseline at LR 8e-3 degrades (PPL 9.195) at batch 1024 and diverges at batch 4096; gated models train at 8e-3 and improve (7.325 and 7.078) [Table 2 rows 6, 10, 12, 14]. Sandwich norm also restores convergence at 8e-3 (PPL 7.407) but gives little over the 4e-3 baseline (7.421) [Table 2 row 7]. Loss-spike reduction is shown only as a plot [Fig. 1 right].
- Clipping attention and FFN outputs at 100 or 300 before the residual add did not prevent divergence at LR 8e-3 [App. A.5].
- Long context (RULER): within 32k, gated is slightly better (79.77 vs 79.50 at 32k). After YaRN to 128k, gated scores 66.60 at 64k and 58.82 at 128k vs baseline 37.51 and 31.65, and loses less inside 32k (-6.89 vs -41.56 at 32k) [Table 5].
- Switch-Heads-style routing gains mostly come from the gate itself: one value expert with a sigmoid gate (Switch v 1top1) beat 8- and 4-expert versions [App. A.1, Table 6].

## Gotchas when reproducing

- Gate must read the current token's state and be per head. Head-shared or V-position gates improve PPL but keep the attention sink [Table 4].
- Keep the sigmoid's range down to 0. Clamping the minimum (NS-sigmoid) loses the sparsity and the sink fix [Table 4 row 7, App. A.4].
- If you match parameters by shrinking the FFN, as the dense runs did, the comparison changes from Table 1's (where gates add parameters). Elementwise G1 adds q·d_k·d_model per layer, which is the size of W_Q.
- All runs are Qwen-internal data and benchmarks; the paper is a Qwen-team self-report with no external replication inside it. See `gated-attn-sinks-replication.md` for a later small-scale test that did not reproduce the sink reduction.
- The LR-tolerance claim was shown on 1.7B dense at 48 layers; the MoE runs used one LR.

## Connections

- Components: a drop-in change to the attention block, after SDPA and before W_o. Works with GQA, RoPE and FlashAttention unchanged since the gate is outside the kernel. Qwen3.8-Next keeps "the attention output gate (Qiu et al., 2025b)" [Q38 §3.1] (see `qwen38-next-design.md`); Qwen3-Next's use of it is stated only in a third-party abstract (arXiv 2605.16640), not checked against a Qwen source.
- Stability: a fourth lever next to QK-norm, z-loss and softcapping. QK-norm bounds attention logits; the gate bounds what attention writes to the residual stream and cut massive activations about 11x [Table 4]. The skill's probe skips position 0 because of massive activations; gated models may not need that.
- Long context: relevant to the skill's rope_theta / YaRN step. Models that rely on sinks lost more under YaRN [Table 5].
- Sinks: alternative to gpt-oss's learned sink logit and to StreamingLLM's kept sink tokens (see `attention-sinks.md`).

## Disagreements and errors

- GSM8k for the same "v Elementwise G2" model is 53.97 in Table 1 row 6 and Table 3 row 3, but 51.33 in Table 4 row 5.
- Added parameters for "v headwise" are 0.2M in Table 1 row 11, but Table 6 says Switch v 1top1 "is equivalent to v Headwise Gate in Tab. 1 row (11)" and lists 3M. The PPL and benchmarks match (5.808, 59.32), so one parameter count is wrong.
- §4.3 says head-shared gating "decreases massive activations, but does not reduce attention scores to the first token"; App. A.3 says for head-shared gating "both massive activations and attention sinks become comparable to those observed in the baseline". Table 4 (M-Act 286 vs 1053) supports §4.3.
- §5.2 calls the dense model "1B-parameter"; everywhere else it is 1.7B.
- Small-scale replication (arXiv 2609.08574) reports the sink reduction did not appear in its setting; see that digest for scope.
