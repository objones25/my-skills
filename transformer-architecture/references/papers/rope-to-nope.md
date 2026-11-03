# Rope to Nope and Back Again: A New Hybrid Attention Strategy (plus NoPE evidence elsewhere)

Bowen Yang, Bharat Venkitesh, Dwarak Talupuru, Hangyu Lin, David Cairuz, Phil Blunsom, Acyr Locatelli (Cohere). arXiv:2501.18795 (v2, 22 Oct 2025; v1 30 Jan 2025). No venue on the arXiv page. Checked: full text of §1-6 from the arXiv HTML. Table 2 (the RoPE / QK-Norm / NoPE comparison) is an image in both renderings, so its numbers were not read.

## The idea

At 8B parameters, the authors compare pure RoPE, RoPE with QK-Norm, and NoPE (no positional encoding). Interleaving NoPE and RoPE layers produces a "division of labor": NoPE layers do long-range retrieval (attention mass on the needle) and RoPE layers act locally with strong recency bias [RNoPE §2.2.3]. Raising RoPE θ in the hybrid hurt retrieval, the opposite of pure RoPE practice. The final design, RNoPE-SWA, puts RoPE with a 4,096-token sliding window in three of every four layers and full-attention NoPE in the fourth, and drops QK-Norm. Trained on 5T tokens and compared with a RoPE baseline that raised θ to 8M for 128K, it matched short benchmarks and held up much better at 128K-256K [Tables 4-7]. KV cache shrinks up to 75% because three layers in four keep only a 4K window [§5.3].

## Notation

| Symbol | Meaning |
|---|---|
| RoPE / NoPE layer | Attention layer with / without rotary position embedding |
| θ | RoPE base |
| SWA | Sliding-window attention, window S |
| S | Window size (4,096 in the final model; 8,192 in the §2 ablation) |
| L | Full training context length |
| Attention mass | Sum of attention from the final query tokens to a context segment (Begin = first 10 tokens, Needle, Context, End), averaged over heads and layers [§2.2.2] |
| Needles score | NIAH score, maximum 10, with 16 seeds per context-depth cell [§2.2.1, Tables 3, 5] |

## The method

1. Base: Command R+-style decoder, 8B (Table 1: d_model 4096, FFN 28672, SwiGLU, 32 layers, 32 heads, 8 KV heads, vocab 256,000).
2. Groups of four layers: three SWA layers with RoPE (window 4,096, θ = 10,000 throughout), then one full-attention NoPE layer at the end of each group [§3].
3. No QK-Norm (removed because its attention patterns hurt long-context retrieval) [§3, §2.2.1].
4. No θ change during length extension for the hybrid; the baseline raised θ to 1M at 32K and 8M at 128K [§4].

## Hyperparameters

| Setting | Value | Source |
|---|---|---|
| §2 analysis runs | 8B, 750B tokens, batch 4M tokens, AdamW peak LR 7e-3, 2000 warmup, cosine to 3.5e-4 over 179,000 steps | §2.1 |
| §2 SFT | short:long data 3:1 at 8,192 and 65,536 tokens, batch 0.5M tokens | §2.1 |
| Final pretraining | 5T tokens, batch 8M, FP8, cosine peak 5e-3 to 5% with 8,000 warmup steps | §4 |
| Cooldown | linear 2.5e-4 to 1e-6 over 50,000 steps in BF16; 8K for 35,000 steps, 32K for 10,000, 128K for 5,000 | §4 |
| Interleave ratio | 1 full : 3 SWA (1:1 and 1:7 also tested) | §3 |

## Key results

- RNoPE hybrid with θ raised at SFT: needles-128K score 8.04 (θ 10k), 7.46 (100k), 7.02 (2M), 6.20 (4M); RoPE baseline 7.40; RNoPE-10k with 8K SWA on the RoPE layers 9.56 [Table 3].
- In the NoPE layers, needle attention mass fell from 0.0765 to 0.0369 as θ rose from 10k to 4M [§2.2.3].
- QK-Norm model was worst on long context in the §2 comparison; NoPE alone lagged on short benchmarks [§2.2.1] (numbers in an image).
- Final models, short benchmarks after SFT (baseline / RNoPE-SWA): MMLU 57.5 / 59.5, GSM8K 40.9 / 42.7, ARC-E 84.6 / 82.5 [Table 4].
- Long context: needles-256K 8.25 vs 9.97; RULER retrieval at 256K 57.1 vs 74.8, RULER QA 30.0 vs 42.5 [Tables 5-7].
- Speed: about 50% faster training at 64K and nearly 2x at 128K; 44% lower latency at 132K input and about 70% at 990K input [§5.3].

## Gotchas when reproducing

- Do not raise θ for the RoPE layers of a NoPE/RoPE hybrid; it widened RoPE layers' receptive field and degraded retrieval [§2.2.3].
- The 1:3 ratio was chosen by an ablation whose numbers the paper does not report [§3].
- Benchmarks are on SFT models, one seed each, Cohere-internal data.

## Other NoPE / iRoPE evidence (not papers, or not read)

- Kimi Linear uses NoPE in all its global MLA layers (3:1 with KDA) and reports better 128K results than the same model with RoPE (54.5 vs 51.8 average) [KL Table 5]; see `kimi-linear.md`.
- SmolLM3 (Hugging Face) removing RoPE from every fourth layer: blog, not paper; not read for this digest.
- Llama 4 "iRoPE" (interleaved NoPE layers): Meta published release materials, not a paper. arXiv 2601.11659 is a third-party compilation of public Llama 4 details (abstract only seen); do not cite it as Meta's own account.
- Theory: arXiv 2609.38109 argues local mixing layers (SWA or gated linear attention) induce a recency bias that global NoPE layers can read (abstract only).

## Connections

- Position (skill step 6): a third strategy beside "large θ natively" and "small θ plus YaRN": no RoPE in the global layers, short-window RoPE (or linear attention) elsewhere, no extension method needed.
- Stability vs long context: this paper reports QK-Norm hurting long-context retrieval. The skill recommends QK-norm above 1B for stability. The evidence is one 8B comparison with numbers in an image; flag as a trade-off, not a verdict.
- KV cache: three-quarters of layers capped at the window size (`kimi-linear.md` and `qwen38-next-design.md` reach a similar 75% with linear attention instead of SWA).
