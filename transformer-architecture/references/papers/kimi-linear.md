# Kimi Linear: An Expressive, Efficient Attention Architecture

Kimi Team (Yu Zhang, Zongyu Lin, Xingcheng Yao, Jiaxi Hu, ..., Songlin Yang, ...; full list on arXiv). arXiv:2510.26692 (v2, 1 Nov 2025; v1 30 Oct 2025). Technical report. Checked: full text of §1-5 (HF rendering); §6 (DPLR relation, complexity) and appendices skimmed only.

Qwen3-Next, the other model named in the brief, was announced by blog post; no Qwen3-Next paper turned up in a search of the Firecrawl index (searched 2 Nov 2026). Its design is described secondhand in the Qwen3.8-Next paper (`qwen38-next-design.md`).

## The idea

A layerwise hybrid: three Kimi Delta Attention (KDA) layers, then one full-attention MLA layer, repeated [KL §4]. KDA is Gated DeltaNet with a per-channel forget gate instead of one scalar per head, implemented with a restricted diagonal-plus-low-rank transition that the authors say runs about twice as fast as general DPLR kernels [§3, §3.2]. The full-attention layers use no positional encoding (NoPE), leaving position to KDA [§4]. Only one layer in four keeps a growing KV cache, hence "up to 75%" less KV cache, and the paper reports 6.3x faster time per output token than full MLA at 1M context (1.84 ms vs 11.48 ms) [Abstract, Fig. 1b]. In a matched 1.4T-token comparison at 48B total / 3B active, Kimi Linear beat full MLA on most short-context, long-context and RL measures [Tables 3-5, Fig. 6].

## Notation

| Symbol | Meaning |
|---|---|
| x_t | Token input, shape (d,) |
| S_t | KDA state per head, shape (d_k, d_v) (note: transposed vs the Gated DeltaNet paper) |
| q_t, k_t, v_t | Query, key, value per head; d_k = d_v = 128 [§4] |
| α_t | Per-channel forget gate, shape [0,1]^{d_k}, from a rank-d_k low-rank projection and a decay function f [§4] |
| Diag(α_t) | Diagonal matrix of α_t |
| β_t | Write strength, sigmoid(W_β x_t), scalar per head |
| W_g↑, W_g↓ | Low-rank output-gate projections |
| C | Chunk length |
| N:1 | Hybrid ratio, N KDA layers per full MLA layer |

## The method

### 1. KDA recurrence

S_t = (I − β_t k_t k_tᵀ) Diag(α_t) S_{t-1} + β_t k_t v_tᵀ; o_t = S_tᵀ q_t [KL Eq. 1]. Compared with Gated DeltaNet, the scalar α_t becomes a vector, so each key channel forgets at its own rate.

### 2. Projections and output

q, k = L2Norm(Swish(ShortConv(W x))); v = Swish(ShortConv(W_v x)); α from a low-rank projection; β = sigmoid(W_β x). Output: W_o (sigmoid(W_g↑ W_g↓ x) ⊙ RMSNorm(KDA(...))), a head-wise RMSNorm and sigmoid gate before the output projection [Eq. 10]. The authors cite gated attention [79] for the gate and say it alleviates the attention sink [§4].

### 3. Hybrid layout

Layerwise (whole layers), not headwise mixing, "for its superior infrastructure simplicity and training stability"; 3 KDA : 1 MLA [§4].

### 4. NoPE in MLA layers

No RoPE in the global layers. Stated benefits: MLA can be run as plain MQA at inference, and long-context extension needs no RoPE base change or YaRN [§4].

### 5. Backbone

Follows Moonlight (DeepSeek-V3-style MoE). For the 1.4T comparisons: 8 of 256 experts active including 1 shared, 48B total / 3B active, first layer dense "ensuring stable training" [§5.4].

## Hyperparameters

| Setting | Value | Source |
|---|---|---|
| Scale (main comparison) | 48B total, 3B active, MoE 8/256 incl. 1 shared, first layer dense | §5.4 |
| Optimizer | MuonClip | §5.4.1 |
| Schedule | WSD; annealing and long-context phase as in Kimi K2 | §5.4.1 |
| Peak LR | 1.1e-3 | §5.4.1 |
| Batch | 32M tokens | §5.4.1 |
| Context in pretraining | 4,096 | §5.4.1 |
| Tokens | 1.4T for comparisons; 5.7T for the released checkpoint (1M context) | §5.4.1 |
| Head dims | d_k = d_v = 128 | §4 |
| Short conv kernel | small (e.g. 4); exact value not given in the sections read | §5.2 |
| Scaling-law runs | 653M to 1.7B active, 8 of 64 experts, Muon, LR 2.006e-3 down to 1.371e-3, batch 336 to 640, context 4096 | Table 2 |
| Layer count, width for 48B | not given in the sections read | |

## Key results

- Hybrid ratio ablation (16 heads, 16 layers, equal FLOPs), training / validation PPL: 3:1 9.23 / 5.65; 1:1 9.29 / 5.66; 7:1 9.23 / 5.70; 15:1 9.34 / 5.82; all-MLA 9.45 / 5.77 [Table 1].
- Component ablations at 3:1: no output gate 9.25 / 5.67; swish instead of sigmoid gate 9.43 / 5.81; no short convolution 9.29 / 5.70 [Table 1].
- Pretrained at 1.4T, MLA / GDN-H / Kimi Linear: MMLU 71.6 / 72.2 / 73.8; MMLU-Pro 47.2 / 47.9 / 51.0; EvalPlus 59.5 / 63.1 / 60.2 [Table 3].
- Long context at 128K, average of 8 tasks: MLA 52.2, GDN-H 51.2, Kimi Linear with RoPE 51.8, Kimi Linear (NoPE) 54.5; RULER 81.3 / 80.5 / 78.8 / 84.3 [Table 5].
- Scaling-law fit: about 1.16x compute efficiency over MLA, with hyperparameters tuned for MLA and reused for Kimi Linear [§5.3, Fig. 5].
- Speed: comparable to MLA at 4K-16K, faster from 128K on; TPOT 1.84 ms vs 11.48 ms at 1M tokens [§5.6, Fig. 1b].

## Gotchas when reproducing

- Sigmoid output gate, not swish: the swish gate used by Gated DeltaNet was clearly worse here (validation 5.81 vs 5.65) [Table 1].
- Hybrid ratio is a trade-off on validation (out-of-distribution) loss, not training loss: 7:1 matched 3:1 on training PPL but lost on validation [§5.2].
- NoPE in the global layers helped long context but Kimi Linear (RoPE) was similar on short-context tasks [§5.2, Table 5].
- The "75%" is the KV cache of attention layers only; KDA states are fixed size but not free (128×128 per head per layer).
- Self-report by the lab that built the model, on its internal evaluation framework, with temperature 1.0 [§5.4]. The baselines share its recipe, which makes the comparison internally fair, but hyperparameters were tuned for MLA.

## Connections

- KV cache sizing: with a 3:1 hybrid, the skill's cache formula applies to one layer in four. Combined with MLA in the global layers, the per-token cache is (d_c + d_h^R) × (n_layers / 4) if RoPE dims are kept, or d_c × n_layers / 4 with NoPE.
- Position: this is published evidence for NoPE in the global layers of a hybrid (see also `rope-to-nope.md`). The skill's rope_theta step assumes every layer uses RoPE.
- Gating: output gate on the token mixer, same as gated attention (`gated-attention.md`).
- Optimizer: trained with MuonClip (`muon-moonlight.md`).
- The Gated DeltaNet paper's hybrids used sliding-window attention; Kimi Linear and Qwen3.8-Next use global attention at 1 in 4 (`gated-deltanet.md`, `qwen38-next-design.md`).

## Disagreements and errors

- The abstract says Kimi Linear "outperforms full MLA with a sizeable margin across all evaluated tasks". Its own tables show losses to MLA on instruct EvalPlus (61.0 vs 62.6), LiveBench (45.2 vs 45.7), LongBench V2 (35.0 vs 36.1) and Frames (58.8 vs 60.5) [Tables 4-5]. The claim holds on averages, not on every task.
