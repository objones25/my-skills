# DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model (MLA focus)

DeepSeek-AI (Aixin Liu, Bei Feng, Bin Wang, ... ; full list on arXiv). arXiv:2405.04434 (v5, 19 Jun 2024; v1 7 May 2024). Technical report, no venue. Checked: full text of §2-3 from the arXiv HTML, Appendix D (MHA/GQA/MQA and MLA/MHA ablations) from the HF rendering; alignment §4 skimmed only.

## The idea

Multi-head Latent Attention (MLA) caches one small latent vector per token per layer instead of full keys and values. Keys and values are reconstructed from the latent by up-projections, and at inference those up-projections can be folded into the query and output projections, so keys and values never need to be materialized [DSV2 §2.1.2]. RoPE breaks that folding, so position goes through a separate small "decoupled" query/key pair whose key is shared across heads and cached alongside the latent [§2.1.3]. For DeepSeek-V2 the cache is (d_c + d_h^R)·l = (512 + 64)·60 = 34,560 elements per token, the same as GQA with 2.25 groups, and the paper reports MLA beating MHA on four hard benchmarks at both 16B and 250B MoE scale [Table 1, App. D.2 Table 9]. DeepSeek-V2 itself is 236B total / 21B active, 60 layers, DeepSeekMoE FFNs, 8.1T tokens [§3.1].

## Notation

| Symbol | Meaning |
|---|---|
| d | Model width (5120 in DeepSeek-V2) |
| n_h | Attention heads (128) |
| d_h | Per-head dimension of the compressed (content) part (128) |
| l | Number of layers (60) |
| h_t | Attention input for token t, shape (d,) |
| c_t^{KV} | Joint KV latent, shape (d_c,); this is what gets cached [Eq. 9] |
| d_c | KV compression dimension (512 = 4 d_h) |
| W^{DKV} | Down-projection, shape (d_c, d) |
| W^{UK}, W^{UV} | Up-projections to keys and values, shape (d_h n_h, d_c) [Eq. 10-11] |
| c_t^Q | Query latent, shape (d'_c,), used to save training activation memory, not cache [Eq. 12] |
| d'_c | Query compression dimension (1536) |
| q_{t,i}^R | Decoupled RoPE query for head i, shape (d_h^R,) [Eq. 14] |
| k_t^R | Decoupled RoPE key, one per token shared by all heads, shape (d_h^R,); cached [Eq. 15] |
| d_h^R | Decoupled RoPE dimension per head (64 = d_h/2) |
| n_g | GQA group count (for the cache comparison) |
| N_s, N_r, K_r | Shared experts, routed experts, active routed experts (2, 160, 6) |
| M, D | Max devices a token's experts may span (3); devices holding one layer's experts (8) |

## The method

### 1. Joint KV compression

c_t^{KV} = W^{DKV} h_t; k_t^C = W^{UK} c_t^{KV}; v_t^C = W^{UV} c_t^{KV} [Eq. 9-11]. Only c_t^{KV} is cached. Because q·k = q^T W^{UK} c, W^{UK} can be absorbed into W^Q, and W^{UV} into W^O, at inference [§2.1.2].

### 2. Decoupled RoPE

Rotating k^C would place a position-dependent matrix between W^Q and W^{UK} and break absorption, forcing recomputation of all prefix keys [§2.1.3]. Instead: q_t^R = RoPE(W^{QR} c_t^Q) per head, k_t^R = RoPE(W^{KR} h_t) shared across heads, and each head attends with q_{t,i} = [q^C_{t,i}; q^R_{t,i}], k_{t,i} = [k^C_{t,i}; k^R_t], scaled by 1/sqrt(d_h + d_h^R) [Eq. 14-18].

### 3. KV cache per token (elements, any precision) [Table 1]

| Mechanism | Cache per token |
|---|---|
| MHA | 2 n_h d_h l |
| GQA | 2 n_g d_h l |
| MQA | 2 d_h l |
| MLA | (d_c + d_h^R) l, about (9/2) d_h l for DeepSeek-V2 |

### 4. Extra norms and scales

Low-rank bottlenecks and fine-grained experts change output scale, so RMSNorm is added after the compressed latents and extra scaling factors are applied at the bottlenecks (latents and routed-expert hidden states) "to ensure stable training" [§3.1.2]. The scale values are not given in the paper.

### 5. MoE routing (DeepSeekMoE plus communication control)

Device-limited routing: pick the M devices holding the highest-affinity experts, then top-K within them; M ≥ 3 performed roughly like unrestricted top-K [§2.2.2]. Three balance losses: expert (α_1), device (α_2), communication (α_3) [Eq. 23-31]. Device-level token dropping at capacity factor 1.0 during training, with about 10% of sequences never dropped; no dropping at evaluation [§2.2.4].

### 6. Long context

YaRN applied only to the decoupled key k^R: scale s = 40, α = 1, β = 32, target 160K; attention temperature changed to sqrt(t) = 0.0707 ln s + 1; 1000 steps at 32K sequence length, batch 576 sequences [§3.1.4].

## Hyperparameters

| Setting | Value | Source |
|---|---|---|
| Layers, width | 60, 5120 | §3.1.2 |
| Heads, d_h, d_c, d'_c, d_h^R | 128, 128, 512, 1536, 64 | §3.1.2 |
| FFN | all MoE except layer 1; 2 shared + 160 routed experts, 6 active, expert hidden 1536 | §3.1.2 |
| Params | 236B total, 21B active | §3.1.2 |
| Init std | 0.006 | §3.1.2 |
| Optimizer | AdamW (0.9, 0.95), wd 0.1, clip 1.0 | §3.1.2 |
| LR | peak 2.4e-4, 2K warmup, ×0.316 at ~60% and ~90% of tokens | §3.1.2 |
| Batch | ramp 2304 to 9216 sequences over the first 225B tokens | §3.1.2 |
| Sequence length / tokens | 4K / 8.1T | §3.1.2 |
| Routing | D = 8, M = 3; α_1 = 0.003, α_2 = 0.05, α_3 = 0.02 | §3.1.2 |
| Vocab | 100K BBPE | §3.1.1 |
| Bottleneck scaling factors, norm eps | not given in the paper | |

## Key results

- 7B dense, 1.33T tokens, parameters equalized by changing depth: MMLU MQA 37.9, GQA-8 41.2, MHA 45.2; C-Eval 30.0, 37.7, 42.9 [App. D.1 Table 8].
- MLA vs MHA in MoE: small (15.7B, 1.33T tokens) MMLU 50.0 vs 48.7 with cache 15.6K vs 110.6K elements; large (247B, 420B tokens) MMLU 59.0 vs 57.5, BBH 50.7 vs 46.6, cache 34.6K vs 860.2K [App. D.2 Table 9]. C-Eval is the one loss (small: 50.9 vs 51.6).
- Versus DeepSeek 67B dense: 42.5% lower training cost per trillion tokens (172.8K vs 300.6K H800 GPU hours), KV cache 93.3% smaller, maximum generation throughput 5.76x (over 50K tokens/s on one 8×H800 node, with FP8 weights and about 6-bit KV cache quantization) [§3.2.3, Abstract].
- Device-limited routing with M ≥ 3 roughly matched unrestricted top-K [§2.2.2] (no numbers given).

## Gotchas when reproducing

- The RoPE part of the key is shared across heads (MQA-style), and only d_h^R = 64 dimensions carry position. Applying YaRN means scaling only these dimensions [§3.1.4].
- The softmax scale uses d_h + d_h^R, not d_h [Eq. 18].
- Absorbing W^{UK} into W^Q changes the attention compute shape (queries now live in the d_c space). It saves memory bandwidth at decode, but prefill implementations often keep the unabsorbed form; the paper only says absorption is possible [§2.1.2].
- The 93.3% cache reduction and 5.76x throughput mix MLA with FP8 weights and KV quantization; they do not isolate MLA [§3.2.3].
- The MHA/GQA/MQA ablation (App. D.1) matches parameters by changing layer count, so GQA and MQA models are deeper; the head counts of the MHA baselines in Table 9 are not stated.
- All ablations use DeepSeek's internal data and evaluation.

## Connections

- Components (KV cache, GQA): MLA is the main alternative to the skill's GQA default when cache size drives the design. At equal cache it reports quality above MHA, where GQA trades quality for cache. Its cost is more projections, a separate RoPE path, and kernel support.
- The skill says GQA costs "~0.1 points" of quality (from the GQA paper's uptraining on T5). DeepSeek's from-scratch 7B ablation reports a 4-point MMLU drop for GQA-8 vs MHA [App. D.1]. Different setups (uptraining vs from scratch, ROUGE vs MMLU), but the skill should not present the small GQA penalty as settled.
- Long context: decoupled RoPE is what lets YaRN be applied to MLA at all (`yarn.md`).
- Stability: extra RMSNorm on latents is a norm-placement choice the skill's stability file does not cover.
- MoE: see `deepseekmoe.md`; DeepSeek-V3 keeps MLA and replaces the balance losses (`deepseek-v3.md`).

## Disagreements and errors

- Table 9's MHA baselines do not match the 2 n_h d_h l formula with DeepSeek-V2's 128 heads × 128 dims × 60 layers (that would be about 1.97M elements, vs 860.2K listed for the large MHA model). The MHA baseline must use fewer heads or layers, which the paper does not state; the comparison is still at roughly equal total parameters (250.8B vs 247.4B).
