# On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training Stability

Qwen Team (Zihan Qiu, Zekun Wang, Xiao Li, ... ; full list on arXiv). arXiv:2608.30320 (v1, 31 Aug 2026; PDF dated 2026-09-01). Technical report. Posted after June 2026: newer than this reviewer's background knowledge, so everything below comes only from the PDF, and no independent replication exists yet. Checked: full text from the PDF via pdftotext. Most table numbers survived extraction; a few tables (2, 3, 7-11) lost their column alignment, so only the numbers the text itself states are quoted from those.

## The idea

A lab report on how Qwen picked the architecture of a 125B-total / 6B-active MoE (plus 51B of n-gram embedding tables kept in host memory) that matches or beats its 397B-A17B predecessor on 8 of 14 base benchmarks at about 1/9 the training FLOPs [Q38 Abstract, §4, Table 11]. Each change was judged on three axes: loss and benchmarks, train/prefill/decode cost, and its effect on optimal hyperparameters and stability [§1]. The decisions:

1. Token mixing: 3 Gated DeltaNet (GDN) layers per 1 full-attention layer. In a 25B-A3B ablation the GDN hybrid beat full attention on 8 of 9 benchmarks (average 53.81 vs 49.87) and an SWA hybrid (window 128) on 7 of 9 (51.15) [§2.1.1, Table 1].
2. Keep RoPE in the full-attention layers. NoPE looked the same in pretraining but gave "a substantially higher rate of endless generation after post-training" [§2.1.1].
3. Qwen Sparse Attention (QSA) replaces the full-attention layers during continued pretraining: a DSA-style indexer that scores 4-token blocks instead of tokens [§2.1.2].
4. Gated Residual (GR): 4 residual branches, read through an elementwise sigmoid gate computed from all branches, written with one scalar per branch, no branch-mixing matrix [§2.2].
5. One n-gram embedding layer at layer 2 [§2.3].
6. Muon for matrix weights, which together with GR raised the optimal LR and batch size and made batch-size warmup pointless [§3.1-3.2].
7. A stress test (constant LR at 2-4x optimal) on which the new recipe had zero loss spikes where the previous AdamW recipe spiked 183 times per 10k steps [§3.3].

## Notation

| Symbol | Meaning |
|---|---|
| S_t | GDN state per head, shape (d_k, d_v) (transpose of the original paper's convention) [Eq. 1-5] |
| α_t, β_t | GDN decay exp(−exp(A)·softplus(W_α x + b_α)) and write strength σ(W_β x) [Eq. 9-10] |
| W_z | GDN output-gate projection; o_t = W_o[σ(W_z x) ⊙ RMSNorm(y_t)] [Eq. 11] |
| n_r | Number of residual branches (4) |
| R_i, R | Residual branch i, shape (d,); all branches R, shape (n_r, d) |
| R̂_i | Branch i after its own RMSNorm with gain γ_i [Eq. 30] |
| G | Elementwise read gates, shape (n_r, d), σ(W_u SiLU((1/n_r) W_d vec(R̂))) with bottleneck rank r = d/8 [Eq. 31] |
| x | Block input, (1/n_r) Σ_i G_i ⊙ R̂_i [Eq. 32] |
| s | Write scalars, 2σ((1/n_r) W_w vec(R̂)), shape (n_r,) [Eq. 33] |
| H_mix, H_combine, H_res | HC's read, write and branch-mixing operators [Eq. 23-28] |
| r (QSA) | Key compression ratio (block size), 4 |
| K | QSA token budget per query, 2048 |
| K_B | Block budget, ⌈K/r⌉ = 512 |
| I_ib | Indexer score of query i for block b, Σ_h ReLU(⟨q_i^h, k̄_b⟩), only for complete blocks [Eq. 15] |
| B, η | Batch size in tokens, peak learning rate |
| TPP | Training tokens per active parameter (300 in n-gram experiments) |

## The method

### 1. GDN hybrid (§2.1.1)

q, k = L2Norm(SiLU(ShortConv(W x))); v = SiLU(ShortConv(W_v x)); gated delta update; output through zero-centered RMSNorm and a sigmoid output gate (sigmoid replaced GDN's original SiLU gate, "consistent improvements") [Eq. 6-11]. Zero-centered RMSNorm is used everywhere to limit growth of norm weights. Full-attention layers keep RoPE and the attention output gate (Qiu et al. 2025b) [§2.1.1, §3.1]. Kernel: FlashQLA (TileLang), 2-3x forward and about 2x backward over the FLA Triton kernel.

### 2. QSA (§2.1.2)

- Indexer: MQA with 4 query heads and 1 shared key head; keys average-pooled over blocks of r = 4 tokens, then RMSNorm, then partial RoPE (64 of 128 dims) at the block's start position; block-causal ReLU scores; top-512 blocks plus the current incomplete block [Eq. 12-16, 19].
- Training: during continued pretraining at 256K. Stage 1: indexer only, KL to the summed-over-heads teacher attention max-pooled to blocks, 1000 steps × 8 × 256K (about 2B tokens), LR 1e-3. Stage 2: everything trained sparse, KL on the selected blocks, 8000 steps × 96 × 256K (about 200B tokens), LR 2.5e-5 [Eq. 17-20].
- Indexer cost falls from O(n²) to O(n²/r). MTP reuses QSA indices across speculative steps.

### 3. Gated Residual (§2.2)

Per sublayer (separate GR modules for attention and MLP): normalize each branch, compute elementwise read gates from all branches, average gated branches as the block input (this replaces the block's pre-norm), write the block output to every branch scaled by s_i ∈ (0, 2) [Eq. 30-34]. No static terms or special init needed. Dropping H_res removes one full read of the residual state per block, the main decode cost of a widened stream. The residual state can be stored in FP8 "with almost no loss in quality" because gates bound what is written [§2.2].

### 4. Muon setup (§3.1)

- Nesterov momentum µ = 0.95; update scaled by 0.2·sqrt(max(A, B)) (Moonlight rule); Polar Express Newton-Schulz coefficients, 8 steps (fewer gradient-norm spikes under stress); Frobenius-normalization epsilon 1e-14.
- Muon on: attention q/k/v/o, GDN input/output projections, expert fc1/fc2 (routed and shared), n-gram key/value projections.
- AdamW on: input embeddings, output head, MoE router (Muon destabilized it early), GR low-rank projections, GDN decay/beta projections (vectors), output gates (AdamW on par or better). N-gram table: Adam with no weight decay.
- Split fused matrices (qkv per head, GDN input per head, fc1 into gate and up) before orthogonalization, so singular directions of unrelated sub-blocks do not mix and the shape scale is right.
- Systems: "Canzona" partitions whole matrices across data-parallel ranks by estimated Newton-Schulz FLOPs; optimizer step captured in a CUDA graph.

### 5. Hyperparameter scaling and stress test (§3.2-3.3)

- Refit the LR and batch-size scaling law used for Qwen3.5; the new fit predicts larger B and η and slower LR decay with model size. The fitted formula is not printed in the text read.
- Stress test: 28-layer 25B-A3B MoE, LR held constant at 2x or 4x optimal, clip 0.5; spike = loss more than 0.1 above a 201-step rolling median.

## Hyperparameters

| Setting | Value | Source |
|---|---|---|
| Final model | 125B total, 6B active, plus 51B n-gram embedding params off-accelerator | Abstract, Table 11 |
| Layer pattern | 3 GDN : 1 full attention (QSA after CPT) | §2.1.1, Fig. 1 |
| Residual branches | n_r = 4, GR rank d/8 | §2.2 |
| N-gram layer | single layer at layer 2; vocabulary scaled 20V-200V in ablations (V = 250K base vocab) | §2.3, Tables 8-9 |
| QSA | r = 4, K = 2048, 4 indexer query heads, partial RoPE 64/128 | §2.1.2 |
| Muon | µ 0.95, 8 Polar Express NS steps, scale 0.2·sqrt(max(A,B)), eps 1e-14 | §3.1 |
| Grad clip | 0.5 | §3.2-3.3 |
| Validated LR/batch (48-layer 156B-A7B, 419B tokens) | B = 8.4M tokens, η = 1.76e-3 (previous Qwen3.5 recipe: 4.2M, 6.8e-4) | Fig. 9, Table 10 |
| Validated batch (20-layer 10.8B-A0.89B, 4T tokens) | B = 25.2M (previous 12.6M) | Fig. 8 |
| Layer count, width, expert count, total tokens, schedule of the final model | not given in the text read | |

## Key results

- Architecture ablation (28-layer 25B-A3B, 400B tokens at 4K + 80B at 32K), averages of 9 benchmarks: full attention 49.87, SWA hybrid 51.15, GDN hybrid 53.81; GDN hybrid loses only EvalPlus to full attention (49.71 vs 51.01) [Table 1].
- Residual ablation (25B-A3B, 560B tokens), loss / benchmark average: pre-norm 1.617 / 50.91; mHC static 1.596 / 52.49; mHC dynamic 1.594 / 54.47; GR 1.590 / 54.66 [Table 5]. Static widening alone (one learnable read scalar per branch, round-robin write) cut loss about 0.01 on a 25B-A3B / 400B-token run [§2.2].
- Loss and benchmarks disagree: static to dynamic read/write changed loss by 0.002 but benchmarks by 1.98 points [§2.2].
- GR vs Attention Residuals (28 layers): pre-norm 1.789; Block AttnRes (S = 4) 1.773; Full AttnRes 1.762; GR 1.762; adding GatedNorm lowers each by 0.002-0.005. At 48 layers: Block AttnRes S = 4 1.711 vs GR 1.707 [Table 6].
- QSA vs full attention: short-benchmark average 75.9 to 76.8; RULER beyond 512K 90.08 to 93.00; MRCR (8-needle) 30.66 to 40.53 at 512K and 20.71 to 26.44 at 1M; training loss gap on the order of 1e-4. Kernel speed at 1M: 7.6x prefill, 4.9x decode [§2.1.2, Fig. 4, §1].
- Batch size (10.8B-A0.89B, 4T tokens), mean loss over the last 20B tokens: B = 25.2M 1.5702, 37.7M 1.5707, previous 12.6M 1.5774. Ramping 6.3M to 25.2M was 2.5e-4 to 3.5e-4 worse and took 18.8% more optimizer steps [§3.2, Fig. 8].
- LR (156B-A7B, 419B tokens): predicted optimum and η × or ÷ sqrt(2) and B × 1.25 end within 7e-4 of each other; the previous recipe ends 7.8e-3 higher. Max pre-clip gradient norm 28% of the 0.5 threshold at the optimum vs 51% under the previous recipe [§3.2, Fig. 9].
- Stress test (2x optimal LR): spikes per 10k steps AdamW/Qwen3.5 structure 4.3 vs Muon 0.2. At 4x: AdamW 183 per 10k steps and 213 of 19,932 steps over the clip threshold; both Muon runs never crossed the threshold; Muon + GR had zero spikes [§3.3, Fig. 10].
- GatedNorm alone (AdamW, 3x optimal LR): spikes 32.0 to 3.2 per 10k steps, clip crossings 256 to 20 [§3.3, Fig. 12].
- Production check, first 276B tokens: GR lowers loss 0.026 vs Qwen3.5 structure + Muon, the full recipe a further 0.032; Muon alone had about 2x the median and 4.2x the p99.9 gradient norm of the gated runs and was the only run to cross the clip threshold. Full training had no loss spike and used no qk-clip or SwiGLU-clip [§3.3, §1].
- N-gram embeddings: under a fixed total parameter budget, loss was lowest at 10x vocabulary (25% of parameters) but benchmarks showed no clear gain over MoE-only; with extra parameters, loss fell monotonically from 20V to 200V while benchmarks saturated, except Chinese benchmarks, which kept improving [§2.3].

## Gotchas when reproducing

- Do not judge NoPE-in-global-layers or sparse residual reads by pretraining metrics: both looked free in pretraining and hurt after post-training (endless generation for NoPE; quality loss for reading only the top-2 branches) [§2.1.1, §2.2].
- Split fused weight matrices before Muon's orthogonalization [§3.1].
- Keep the router, embeddings, output head, gates and other elongated or vector-like parameters on AdamW [§3.1].
- Refit LR and batch size when changing architecture and optimizer together; the old recipe was 7.8e-3 worse in loss at 156B-A7B [§3.2].
- Read every benchmark claim as a Qwen self-report on Qwen's internal pipeline. Final-model comparisons are against Qwen's own predecessors, not other labs' models.

## Connections

- Components: confirms gated attention in production (attention output gate kept), and extends sigmoid gating to the GDN output and the residual read. Sigmoid beat SiLU/tanh in each place they tested [§2.1.1, §2.2]. See `gated-attention.md`, `gated-deltanet.md`.
- Hybrid ratio: 3:1 linear to global, same as Kimi Linear; but Qwen keeps RoPE in the global layers where Kimi Linear uses NoPE (`kimi-linear.md`, `rope-to-nope.md`).
- Residual stream: GR is the HC family with the mixing matrix dropped and an elementwise read gate (`hyper-connections.md`). It also replaces pre-norm.
- Stability: adds a measured stress-test protocol (constant LR at 2-4x optimal, spike = 0.1 above rolling median, clip 0.5) to the skill's Wortsman-style proxy guidance. GatedNorm is a stabilizer the skill does not list.
- Optimizer and sizing: Muon with RMS matching (`muon-moonlight.md`) here does not keep AdamW's optimum; the refit raised η about 2.6x and B 2x at 156B-A7B. Batch-size warmup was counterproductive with Muon, unlike Llama 3's ramp and DeepSeek-V2/V3's batch ramps.
- Sparse attention: QSA is DSA with block-compressed keys in the indexer (`sparse-attention-nsa-dsa.md`).

## Disagreements and errors

- Positional encoding in global layers: Qwen reports NoPE causing endless generation after post-training [§2.1.1]; Kimi Linear reports NoPE improving long-context scores with similar short-context scores [KL §5.2, Table 5]. Kimi does not report post-training termination rates; Qwen does not report long-context numbers for NoPE.
- Naming: the abstract and conclusion compare against the "397B-A17B predecessor"; §3.3 and §4 call it Qwen3.7-Plus, while the hyperparameter and stress-test baselines are labelled "Qwen3.5" recipe/structure. The text does not say whether Qwen3.7-Plus used the Qwen3.5 structure and recipe.
- Typos in figure text: Fig. 11's caption says "Gradient Residual" for Gated Residual; §3.3 writes "AdmaW".
