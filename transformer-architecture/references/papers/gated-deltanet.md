# Gated Delta Networks: Improving Mamba2 with Delta Rule

Songlin Yang (MIT), Jan Kautz, Ali Hatamizadeh (NVIDIA). arXiv:2412.06464 (v3, 6 Mar 2025; v1 9 Dec 2024). ICLR 2025 (camera ready, per the arXiv comment). Checked: full text of §1-4 and Appendix B (HF rendering).

## The idea

Linear attention keeps a fixed-size matrix state S instead of a growing KV cache. Two earlier fixes for its weak recall were a decay gate (Mamba2: shrink the whole state each step) and the delta rule (DeltaNet: overwrite the value stored under the current key). The gated delta rule does both: S_t = α_t S_{t-1}(I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ [GDN Eq. 10]. α_t near 0 wipes memory quickly; α_t near 1 reduces to the pure delta rule. The authors extend DeltaNet's chunkwise WY-representation algorithm to include the gate, so training runs on tensor cores at nearly DeltaNet's throughput [§3.3, Fig. 3]. At 1.3B parameters and 100B tokens it beat Mamba2, DeltaNet and a Transformer++ baseline on perplexity and zero-shot tasks, and hybrids with sliding-window attention (SWA) beat everything tested, including on recall-heavy tasks [Tables 3-5].

## Notation

| Symbol | Meaning |
|---|---|
| S_t | Recurrent state of one head, shape (d_v, d_k) |
| q_t, k_t, v_t | Query, key (shape d_k), value (shape d_v) for token t |
| d_k, d_v | Head dimensions for keys and values (head dim 128 chosen by ablation) [App. B.2] |
| α_t | Data-dependent decay in (0, 1), a scalar per head per token, Mamba2 parameterization [Eq. 10, §3.4 fn. 4] |
| β_t | Data-dependent writing strength in (0, 1) [§2.2] |
| o_t | Output, o_t = S_t q_t, shape (d_v,) |
| C | Chunk length for chunkwise training |
| γ_j | Cumulative decay product within a chunk, Π α_i |
| W, U, T | WY-representation matrices used to batch the Householder products in a chunk [Eq. 4-7] |
| SWA | Sliding-window softmax attention (window 2K here) |

## The method

### 1. Recurrence

S_t = S_{t-1} (α_t (I − β_t k_t k_tᵀ)) + β_t v_t k_tᵀ; o_t = S_t q_t [Eq. 10]. In the online-learning view, this is one SGD step on ||S k_t − v_t||² with learning rate β_t plus weight decay α_t [§3.1, Table 1].

### 2. Block design

Llama-style macro architecture: token mixer then SwiGLU MLP. q, k, v come from linear projection, short convolution and SiLU; q and k are L2-normalized "for training stability"; α and β come from linear projections only; the output passes through a normalization and a SiLU output gate before the output projection [§3.4, Fig. 1].

### 3. Chunkwise training

Within each chunk of length C, the product of gated Householder matrices is written in WY form and computed with a UT transform (a triangular solve), so the inter-chunk state update and intra-chunk outputs are matmuls [§3.3].

### 4. Hybrids

H1: Gated DeltaNet + SWA. H2: Mamba2 + Gated DeltaNet + SWA, in that order (best of four orderings at 500M/15B tokens) [§3.4, App. B.2 Table S.2]. These hybrids use sliding-window attention, not global attention, so the whole model has a bounded state at inference.

## Hyperparameters

| Setting | Value | Source |
|---|---|---|
| Scale | 1.3B params, 100B tokens of FineWeb-Edu (also 400M; ablations at 400M-500M / 15B tokens) | §4, App. B.2 |
| Optimizer | AdamW, peak LR 4e-4, wd 0.1, clip 1.0 | §4 |
| Schedule | cosine, 1B-token warmup | §4 |
| Batch | 0.5M tokens | §4 |
| Tokenizer | Llama 2, 32,000 vocab | §4 |
| Train length | 4K; SWA window 2K in hybrids | §4 |
| Head dim | 128 | App. B.2 |
| Layer ratio in H1/H2, chunk size C, α/β init | not given in the sections read | |

## Key results

- 1.3B / 100B tokens, Wiki ppl and average zero-shot accuracy: Transformer++ 18.53 / 52.25; Mamba2 16.56 / 54.89; DeltaNet 17.71 / 52.14; Gated DeltaNet 16.42 / 55.32; H1 16.07 / 56.40; H2 15.91 / 56.18 [Table 3].
- Recall-heavy tasks (inputs truncated to 2K), average: Transformer++ 37.0, Mamba2 29.8, Gated DeltaNet 30.6, H1 39.0, H2 40.1. Pure recurrent models trail attention badly on FDA (Gated DeltaNet 23.7 vs Transformer++ 52.2) [Table 4].
- LongBench average: Transformer++ 11.0, Gated DeltaNet 16.6, H2 18.4 [Table 5]. The Transformer was trained at 4K and run on longer inputs, which flatters the recurrent models.
- Block ablations (400M, 15B tokens): short convolution and the output gate matter; output normalization helps a little; L2 normalization of q, k is essential; SiLU beat other feature maps [App. B.2].
- Training throughput on one H100: Gated DeltaNet about equal to DeltaNet, slightly below Mamba2 (2-3K tokens/s) [§4, Fig. 3].

## Gotchas when reproducing

- L2-normalize q and k; the delta rule's Householder factor (I − β k kᵀ) assumes ||k|| = 1 to stay a contraction [§3.4, App. B.2].
- Keep the short convolutions and the output gate; dropping them cost the most in ablation [App. B.2].
- The baselines are small (1.3B, 100B tokens). Production use at scale came later (Qwen3-Next, Kimi Linear's KDA variant, Qwen3.8-Next); see `kimi-linear.md` and `qwen38-next-design.md`.
- "Transformer++" here is a 4K-trained 1.3B model without long-context training, so its LongBench score is not a fair long-context baseline.

## Connections

- KV cache: replacing most attention layers with a gated delta layer replaces their O(n) per-token cache with a fixed (d_v × d_k) state per head. The skill's cache formula then applies only to the remaining softmax layers.
- Components: q/k L2 norm here plays the role QK-norm plays in softmax attention. The output gate is the linear-attention analogue of gated attention (`gated-attention.md`).
- Hybrid ratio: this paper uses SWA hybrids and gives no linear:global ratio. Kimi Linear (3:1) and Qwen3.8-Next (3:1) do.

## Disagreements and errors

- §4 says Table 3 covers "models with 400M and 1.3B parameters"; the rendered table shows one set of rows, and the ablation tables label the small model 400M (Table S.1) and 500M (Table S.2). Treat the small-model size as ambiguous.
