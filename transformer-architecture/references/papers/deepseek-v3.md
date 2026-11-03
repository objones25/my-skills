# DeepSeek-V3 Technical Report (architecture, balancing, MTP, FP8)

DeepSeek-AI (Aixin Liu, Bei Feng, Bing Xue, ... ; full list on arXiv). arXiv:2412.19437 (v2, 18 Feb 2025; v1 27 Dec 2024). Technical report, no venue. Checked: full text of §2 (architecture), §3.3 (FP8), §4.2-4.5 (hyperparameters, long context, ablations) and Appendix B-C; §3.1-3.2 and §3.4-3.5 (infrastructure) and §5 (post-training) only skimmed.

## The idea

DeepSeek-V3 keeps DeepSeek-V2's MLA attention and DeepSeekMoE FFNs and changes four things that matter for design [DSV3 §2, §3.3]:

1. Auxiliary-loss-free load balancing: a per-expert bias added to the routing scores only for top-K selection, nudged up or down after each step according to that expert's load. The gate values still come from the unbiased scores. Replacing the balance losses with this improved most benchmarks at 16B and 229B MoE scale [Table 5].
2. Sigmoid affinities normalized over the selected experts, instead of softmax over all experts [Eq. 13-15].
3. Multi-token prediction (MTP): one extra sequential module predicts the token after next, with a 0.3 then 0.1 loss weight; dropped at inference or reused for speculative decoding [§2.2, Table 4].
4. FP8 training of nearly all GEMMs with 1×128 activation tiles and 128×128 weight blocks, and FP32 promotion every 128 accumulation elements; loss stayed within 0.25% relative of BF16 [§3.3, App. B.1].

The model is 671B total / 37B active, 61 layers, 14.8T tokens, 2.788M H800 GPU hours in total, and the report says it had no irrecoverable loss spikes or rollbacks [§4.2, Abstract].

## Notation

| Symbol | Meaning |
|---|---|
| u_t | FFN input for token t, shape (d,) |
| N_s, N_r | Shared and routed experts (1 and 256) |
| K_r | Active routed experts per token (8) |
| e_i | Centroid of routed expert i, shape (d,) |
| s_{i,t} | Affinity, sigmoid(u_t^T e_i) [Eq. 15] |
| b_i | Per-expert routing bias; used only in the top-K test [Eq. 16] |
| g'_{i,t}, g_{i,t} | Selected affinity (s_{i,t} if s_{i,t} + b_i is in the top K_r, else 0), and its normalization over the selected experts [Eq. 13, 16] |
| γ | Bias update speed: b_i moves by ±γ per step depending on whether expert i was over- or under-loaded on the batch [§2.1.2] |
| α | Weight of the complementary sequence-wise balance loss [Eq. 17] |
| T | Tokens in one sequence (for the sequence-wise loss) |
| M | Max nodes per token for routing (4) |
| D | MTP depth, the number of extra tokens predicted (1) |
| h_i^k | Representation of token i at MTP depth k; h_i^0 is the main model's output |
| M_k | MTP projection, shape (d, 2d) [Eq. 21] |
| TRM_k | The Transformer block of MTP module k |
| λ | MTP loss weight [Eq. 25] |
| N_C | Accumulation interval before promoting partial sums to FP32 (128) |
| d_c, d'_c, d_h^R | MLA dimensions as in `deepseek-v2-mla.md` (512, 1536, 64) |

## The method

### 1. Routing with a bias, not a loss

Top-K picks experts by s_{i,t} + b_i; the expert output is weighted by g_{i,t} computed from s_{i,t} without b_i [Eq. 16]. After each step, decrease b_i by γ if expert i was overloaded on the whole batch, increase it by γ if underloaded [§2.1.2]. A tiny sequence-wise loss (α = 1e-4) remains "just to avoid extreme imbalance within any single sequence" [§4.2]. Routing is limited to 4 nodes per token, chosen by the sum of the top K_r/M affinities on each node [§2.1.2]. No token dropping in training or inference [§2.1.2].

### 2. MTP module

For depth k: h'_i^k = M_k [RMSNorm(h_i^{k-1}); RMSNorm(Emb(t_{i+k}))], then h^k = TRM_k(h'^k), then the shared output head predicts token i+k+1 [Eq. 21-23]. Embedding and output head are shared with the main model. Loss: L_MTP = (λ/D) Σ_k CE_k, added to the main loss [Eq. 24-25]. Unlike Gloeckle et al. (independent parallel heads), predictions are sequential and keep the causal chain [§2.2].

### 3. FP8 mixed precision

- FP8 (E4M3 everywhere, not E5M2 for gradients) for Fprop, Dgrad and Wgrad GEMMs [§3.3.1-3.3.2].
- Kept in BF16/FP32: embedding, output head, MoE gating, norms, attention [§3.3.1].
- Scales: per 1×128 activation tile and per 128×128 weight block, computed online from the current max [§3.3.2].
- H800 FP8 tensor-core accumulation keeps about 14 bits; a K = 4096 random GEMM had up to ~2% relative error. Fix: copy partial sums to FP32 on CUDA cores every N_C = 128 elements [§3.3.2].
- AdamW moments in BF16; master weights and gradients in FP32 [§3.3.3].
- Activations entering the linear after attention use a custom E5M6 format; scaling factors for some communicated activations are powers of 2 [§3.3.3].

### 4. Long context

YaRN on the decoupled key only, s = 40, α = 1, β = 32; two 1000-step phases, 32K (batch 1920) then 128K (batch 480), LR 7.3e-6 [§4.3].

## Hyperparameters

| Setting | Value | Source |
|---|---|---|
| Layers, width | 61, 7168 | §4.2 |
| MLA | 128 heads, d_h 128, d_c 512, d'_c 1536, d_h^R 64 | §4.2 |
| FFN | first 3 layers dense; then 1 shared + 256 routed experts, 8 active, expert hidden 2048 | §4.2 |
| Params | 671B total, 37B active | §4.2 |
| Init std | 0.006 | §4.2 |
| Optimizer | AdamW (0.9, 0.95), wd 0.1, clip 1.0 | §4.2 |
| LR schedule | 2K-step warmup to 2.2e-4; constant to 10T tokens; cosine to 2.2e-5 over 4.3T; 2.2e-5 for 333B; 7.3e-6 for the last 167B | §4.2 |
| Batch | 3072 to 15360 sequences over the first 469B tokens | §4.2 |
| Sequence length, tokens | 4K, 14.8T | §4.2 |
| γ (bias speed) | 0.001 for the first 14.3T tokens, 0 for the last 500B | §4.2 |
| α (sequence balance loss) | 0.0001 | §4.2 |
| MTP | D = 1; λ = 0.3 for first 10T tokens, 0.1 after | §4.2 |
| Vocab | 128K byte-level BPE | §4.1 |
| Packing | document packing without cross-document attention masking | §4.1 |
| FIM | PSM format at rate 0.1 | §4.1 |
| Bottleneck scaling factors | not given in the paper | |

## Key results

- Aux-loss-free vs aux-loss balancing, same data and architecture: small (15.7B, 1.33T tokens) Pile BPB 0.727 to 0.724, GSM8K 27.1 to 29.6; large (228.7B, 578B tokens) BPB 0.656 to 0.652, HumanEval 40.2 to 46.3, GSM8K 70.7 to 74.5, but MMLU 68.3 to 67.2 [Table 5].
- The gain comes from balancing per batch rather than per sequence. A batch-wise auxiliary loss matched the bias method: 1B MoE val loss 2.258 (sequence-wise loss), 2.253 (bias), 2.253 (batch-wise loss); 3B: 2.085, 2.080, 2.080 [§4.5.3].
- MTP (D = 1) vs none: small BPB 0.729 both, HumanEval 20.7 to 26.8, GSM8K 25.4 to 31.4; large (540B tokens) BPB 0.658 to 0.657, HumanEval 44.5 to 53.7, MMLU 67.5 to 66.6 [Table 4].
- FP8 vs BF16: relative loss error below 0.25% at ~16B (1.33T tokens) and ~230B (~0.9T tokens) [App. B.1, Fig. 10].
- Block-wise (128×128) quantization of activation gradients made a ~16B MoE diverge at ~300B tokens [App. B.2].
- Cost: 180K H800 GPU hours per trillion tokens; 2664K pre-training + 119K context extension + 5K post-training = 2788K [§1 Table 1].

## Gotchas when reproducing

- γ is set to 0 for the final 500B tokens, freezing the biases during the last LR stage [§4.2].
- Batch-level balancing needs large micro-batches; the authors rely on large expert and data parallelism for this, and on redundant expert copies at inference to handle domain shift [§4.5.3].
- The bias only changes which experts are picked, so gate magnitudes still come from the sigmoid scores; normalizing over the selected experts is part of the method [Eq. 13].
- Keep the attention operator, norms, router and embeddings out of FP8 [§3.3.1]. Use per-token (1×128) scaling for activation gradients, not 128×128 blocks [App. B.2].
- FP8 accumulation precision is a hardware issue on H800 (Hopper); the promotion trick and its N_C = 128 were tuned there [§3.3.2].
- Benchmarks are self-reported on DeepSeek's harness. The ablations change one thing at a time, which makes them more useful than the headline comparisons with other labs' models.

## Connections

- MoE sizing: 37B active of 671B (5.5%). The first 3 layers stay dense (DeepSeekMoE kept only 1).
- Stability: the report attributes no specific stabilizer to the absence of spikes. Neither the V3 nor the V2 report mentions QK-norm or z-loss (text search of both); the only extra norms named are the RMSNorms on MLA's compressed latents [§4.2]. The skill's "add QK-norm above 1B" default is not what this model did.
- Precision: the skill's default (bf16 compute, fp32 master and Adam state, 16 bytes per parameter) is one option; DeepSeek-V3 stores Adam moments in BF16 [§3.3.3].
- Packing: no cross-document masking [§4.1], unlike Llama 3, which the skill cites for masking.
- Load balancing contrasts with ST-MoE's router z-loss (`st-moe.md`) and the gated-attention paper's "global-batch LBL" [GatedAttn §3.1].
- Post-training uses GRPO (`rl-post-training.md`).

## Disagreements and errors

- §4.3 says "The YaRN configuration is consistent with that used in DeepSeek-V2" but sets sqrt(t) = 0.1 ln s + 1, while DeepSeek-V2 used 0.0707 ln s + 1 [DSV2 §3.1.4].
- §3.3 says FP8 was validated by "training for approximately 1 trillion tokens"; App. B.1 says 1.33T (small) and about 0.9T (large).
- The MTP ablation's large model trains on 540B tokens [Table 4], the aux-free ablation's on 578B [Table 5], with the same 228.7B architecture; the two ablations do not share a baseline.
