# Widening the residual stream: Hyper-Connections (HC) and Manifold-Constrained Hyper-Connections (mHC)

1. Hyper-Connections. Defa Zhu, Hongzhi Huang, Zihao Huang, Yutao Zeng, Yunyao Mao, Banggu Wu, Qiyang Min, Xun Zhou (ByteDance Seed). arXiv:2409.19606 (v3, 18 Mar 2025; v1 29 Sep 2024). No venue on the arXiv page. Checked: full text of §1-4 from the arXiv HTML; appendices (vision, parameter counts, derivations) not read.
2. mHC: Manifold-Constrained Hyper-Connections. Zhenda Xie, Yixuan Wei, Huanqi Cao, Chenggang Zhao, ..., Wenfeng Liang (DeepSeek-AI). arXiv:2512.24880 (v2, 5 Jan 2026; v1 31 Dec 2025). Checked: full text of §1-6 from the arXiv HTML; Appendix A.1 (model configs) not read.

## The idea

Replace the single residual stream x (width C) with n parallel streams (an n × C matrix). Each layer reads a learned mix of the streams as its input, writes its output back to the streams with learned weights, and mixes the streams with a learned n × n matrix [HC §2]. FLOPs barely change because n (e.g. 4) is far below C. HC reported faster convergence on OLMo-1B/7B and OLMoE-1B-7B (1.8x on OLMoE, ARC-C +6.0) [HC Fig. 1, Table 6]. mHC found that at 27B the unconstrained stream-mixing matrices compound across depth (composite gain up to about 3000) and caused a loss surge; projecting the mixing matrix onto doubly stochastic matrices (rows and columns sum to 1, non-negative) with 20 Sinkhorn-Knopp iterations keeps the composite gain near 1 (max about 1.6), keeps HC's gains, and costs 6.7% training time at n = 4 with fused kernels [mHC §3.1, §4, §5.4].

## Notation

| Symbol | Meaning |
|---|---|
| C (mHC) or d (HC) | Width of one residual stream / layer input |
| n | Expansion rate: number of residual streams (4 in both papers' main runs) |
| x_l (mHC), H (HC) | Hidden state as n streams, shape (n, C) |
| F or T | The layer function (attention or FFN block) |
| H_l^pre (mHC), A_m (HC) | Read weights, shape (1, n): mix the streams into the layer's C-dim input |
| H_l^post (mHC), B (HC) | Write weights, shape (1, n): distribute the layer output across streams |
| H_l^res (mHC), A_r (HC) | Stream-mixing matrix, shape (n, n) |
| α (gating factors), θ / φ, b | Learnable small scalars, dynamic projection weights, and static biases making up each mapping [HC Eq. 10-13; mHC Eq. 5, 7] |
| Amax gain | Max absolute row sum (forward) or column sum (backward) of the composite mixing product across layers [mHC §3.1] |
| t_max | Sinkhorn-Knopp iterations (20) |

## The method

### HC (static and dynamic)

1. Copy the embedding into n streams [HC §2.1].
2. Per layer: input h_0 = A_mᵀ H; output Ĥ = Bᵀ T(h_0)ᵀ + A_rᵀ H [Eq. 2-5].
3. Dynamic variant (DHC): each mapping = static matrix + s·tanh(norm(H) W), with W initialized to 0 and s a small learnable scale [Eq. 10-13].
4. Init equals pre-norm residual: B = 1, A_r = identity, A_m = one-hot cycling over streams by layer index [Eq. 14].
5. Sum the streams at the end before the final norm and unembedding. Scale output-projection init std by sqrt(n) so the summed output keeps the baseline's std [§4 Implementation].
6. No weight decay on static components; weight decay on dynamic ones [§4].

### mHC

1. Flatten the n × C state, RMSNorm it, compute H̃^pre, H̃^post, H̃^res from linear projections plus biases [Eq. 7].
2. Constrain: H^pre = sigmoid(H̃^pre); H^post = 2·sigmoid(H̃^post); H^res = Sinkhorn-Knopp(H̃^res), i.e. exp then alternate row and column normalization 20 times [Eq. 8-9].
3. Doubly stochastic matrices have spectral norm ≤ 1 and are closed under multiplication, so the product across any number of layers stays bounded [§4.1].
4. Systems: fused kernels (TileLang), selective recomputation, overlap of the n-fold larger pipeline communication within DualPipe [§4.3].

## Hyperparameters

| Setting | HC | mHC | Source |
|---|---|---|---|
| n | 4 for main runs (1, 2, 4, 8 ablated) | 4 | HC Table 1, §4.3; mHC §5.1 |
| Models | OLMo-1B, OLMo-7B, OLMoE-1B-7B (1.3B active), 500B tokens each | DeepSeek-V3-style MoE at 3B, 9B, 27B with tokens proportional to size; a 3B model on 1T tokens | HC §4; mHC §5.1 |
| Other training settings | baseline's (OLMo / OLMoE) configuration unchanged | in App. A.1, not read | HC §4 |
| Sinkhorn iterations | n/a | 20 | mHC §4.2 |
| Time overhead | "negligible" FLOPs and params (HC App. B, not read) | 6.7% at n = 4 with optimizations | mHC §1 |

## Key results

- OLMo-1B, 500B tokens, V2 eval loss: baseline 2.811; DHC×1 2.819 (no gain); DHC×2 2.802; DHC×4 2.781; DHC×8 2.778 [HC Table 1]. One stream is not enough.
- OLMo-7B: V2 loss 2.581 to 2.559, downstream average 70.1 to 71.0; the baseline showed "frequent spikes" and the DHC model none [HC Table 5, §4.3].
- OLMoE-1B-7B: ARC-C 41.8 to 47.8, MMLU Var 38.5 to 39.7, HellaSwag 69.5 to 70.2 [HC Table 6].
- mHC ablation of HC components: the stream-mixing matrix gives most of the gain (loss gap −0.022 alone vs −0.027 with all three) [mHC Table 1].
- mHC at 27B: final loss 0.021 below baseline; BBH 43.8 / 48.9 / 51.0, MMLU 59.0 / 63.0 / 63.4, GSM8K 46.7 / 53.2 / 53.8 for baseline / HC / mHC; HC slightly ahead on MATH (26.4 vs 26.0) [mHC §5.2, Table 4].
- HC at 27B: loss surge near step 12k tied to gradient-norm instability; composite Amax gain peaks near 3000. mHC: max about 1.6 [mHC §3.1, §5.4, Figs. 2-3, 7].
- Memory traffic per token per layer for the residual path: about 2C reads / C writes for a plain residual vs (5n+1)C + n² + 2n reads and (3n+1)C + n² + 2n writes for HC [mHC Table 2].

## Gotchas when reproducing

- Without fused kernels HC's extra memory traffic is roughly n times the residual path's, and activations of the mappings must be stored for backward [mHC §3.2]. The FLOP count hides this.
- Pipeline-parallel communication grows n-fold [mHC §3.2].
- Initialize so the model starts as a standard pre-norm transformer (HC Eq. 14), and rescale output-projection init by sqrt(n) [HC §4].
- HC dropped "volatile" downstream datasets from its averages [HC §4 Metrics]; read its averages as curated.
- Both papers report gains with the baseline's hyperparameters reused; no LR retuning for the wider stream is reported in the sections read.

## Connections

- Residual stream and norm placement (skill stability file): HC generalizes pre-norm; mHC's doubly stochastic constraint is a norm-preservation argument similar in spirit to DeepNet's residual scaling. HC reports reduced adjacent-layer cosine similarity (less representation collapse) [HC Fig. 3], which interacts with the skill's residual probe: cosine near 1 in late layers may be a symptom HC addresses.
- Sizing: residual width becomes a separate axis from d_model. Qwen3.8-Next's "Gated Residual" widens the stream to four branches read through an elementwise gate (`qwen38-next-design.md`).
- Kimi's "Attention Residuals" (arXiv 2603.15031, cited by `gated-attn-sinks-replication.md`) is another depth-mixing design; not read here.

## Disagreements and errors

- HC reports HC as more stable than the baseline at 7B (no spikes) [HC §4.3]; mHC reports HC as unstable at 27B (loss surge, gradient-norm spikes) [mHC §3.1, Fig. 2]. Different scales, architectures and labs; the mHC paper is DeepSeek benchmarking its own fix against its own reimplementation of HC.
