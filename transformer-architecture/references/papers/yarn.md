# YaRN: Efficient Context Window Extension of Large Language Models

Bowen Peng, Jeffrey Quesnelle, Honglu Fan, Enrico Shippole (Nous Research, EleutherAI, University of Geneva). arXiv:2309.00071 (v3, 6 Feb 2026; v1 31 Aug 2023). The arXiv page lists no venue. Checked: full text of §1-6 from the arXiv HTML (v3); appendices (NTK-aware derivation, temperature fit, extra tables) not read.

## The idea

RoPE dimensions rotate at different speeds. Dimensions with a short wavelength (many full rotations inside the training length) carry relative position and should be left alone; dimensions whose wavelength exceeds the training length effectively carry absolute position and must be interpolated, or new lengths are out of distribution [YaRN §3.2]. "NTK-by-parts" interpolation applies a ramp: no change for fast dimensions, position-interpolation-style division by the scale factor s for slow ones, a linear blend between. YaRN adds an attention temperature, implemented by scaling the rotary embeddings by sqrt(1/t) = 0.1 ln s + 1, which needs no kernel change [§3.3, Eq. 15]. Fine-tuning Llama 2 for 400 steps at 64K (plus 200 more for s = 32) reached 128K with low perplexity and small short-benchmark losses on Llama 2 [Tables 1, 3].

## Notation

| Symbol | Meaning |
|---|---|
| \|D\| | Head dimension (RoPE works on \|D\|/2 complex pairs) |
| θ_d | RoPE frequency of pair d, θ_d = b^(−2d/\|D\|) |
| b | RoPE base (10000 in Llama) |
| L | Pretrained context length |
| L' | Target context length |
| s | Scale factor, s = L'/L |
| λ_d | Wavelength of pair d, λ_d = 2π/θ_d = 2π b^(2d/\|D\|) [Eq. 8] |
| r(d) | Rotations of pair d within L, r = L/λ_d [Eq. 10] |
| α, β | Ramp boundaries in units of r: r < α is fully interpolated, r > β untouched [Eq. 11] |
| γ(r) | Ramp: 0 below α, 1 above β, (r − α)/(β − α) between |
| h(θ_d) | New frequency: (1 − γ) θ_d / s + γ θ_d [Eq. 13] |
| t | Attention temperature; logits divided by t·sqrt(\|D\|) [Eq. 14] |

## The method

1. Pick s = L'/L and α, β (α = 1, β = 32 for Llama) [§3.2].
2. For each RoPE pair, compute r(d) and replace θ_d by h(θ_d) = (1 − γ(r)) θ_d/s + γ(r) θ_d; positions unchanged [Eq. 12-13].
3. Scale both q and k by sqrt(1/t) = 0.1 ln s + 1 (multiply the cached cos/sin tables), equivalent to softmax temperature t [Eq. 14-15]. The 0.1 coefficient was fit on LLaMA 7B-65B without fine-tuning and also worked for Llama 2 [§3.3].
4. Fine-tune briefly on long data, or use without fine-tuning.
5. Optional Dynamic Scaling at inference: s = max(1, l'/L) for current length l'; cache keys before applying RoPE if s changes during generation [§3.4].

## Hyperparameters

| Setting | Value | Source |
|---|---|---|
| α, β | 1, 32 (Llama family) | §3.2 |
| Temperature | sqrt(1/t) = 0.1 ln s + 1 | Eq. 15 |
| Fine-tuning | AdamW (0.9, 0.95), LR 2e-5, no weight decay, 20 warmup steps | §4.1 |
| Data | PG19 chunked to 64K (32K for LLaMA ablations), BOS/EOS bookends | §4.1 |
| s = 16 | 400 steps, global batch 64 | §4.1 |
| s = 32 | from the s = 16 checkpoint, 200 more steps, still 64K data | §4.1 |

## Key results

- Llama 2 7B, sliding-window perplexity on 128K Proof-pile documents: s = 16 gives 2.42 at 64K and >10 at 128K; s = 32 gives 2.45 at 64K and 2.37 at 128K although trained only on 64K sequences [Table 1].
- Short benchmarks after extension, Llama 2 7B (base / s = 16 / s = 32): MMLU 43.8 / 42.5 / 41.7; ARC-c 53.1 / 52.3 / 52.1 [Table 3]. 13B MMLU 55.8 / 52.8 / 51.9.
- LLaMA 7B extended 16x (2K to 32K), 400 steps each method: MMLU base 35.7, PI 25.9, NTK-aware 27.7, NTK-by-parts 32.7, YaRN 30.0 [Table 2].
- Training cost: 7B to 64K in 256 A100-hours, to 128K another 128; PI to 16K (Chen et al.) 640 [Table 4].

## Gotchas when reproducing

- The temperature coefficient (0.1) is fit to Llama. DeepSeek-V2 used 0.0707 for MLA's decoupled key; DeepSeek-V3 used 0.1 [DSV2 §3.1.4, DSV3 §4.3].
- With Dynamic Scaling and a KV cache, cache keys before RoPE, or rotated keys go stale when s changes [§3.4].
- YaRN only touches RoPE dimensions. In MLA only the decoupled key/query dimensions get it (`deepseek-v2-mla.md`). In NoPE layers there is nothing to scale.
- Gated attention reports YaRN extension hurting inside the original window much less when the model has no attention sink (RULER at 32K: −6.89 vs −41.56) [GatedAttn Table 5]; ungated models lose ground inside the trained length.

## Connections

- Directly supports the skill's step 6 ("keep 1e4 and add explicit scaling later (DeepSeek plus YaRN x40)"). The x40 is DeepSeek's s = 40 [DSV2 §3.1.4].
- Alternative path: NoPE in global layers needs no extension method at all (Kimi Linear, Cohere RNoPE-SWA; see `rope-to-nope.md`, `kimi-linear.md`).

## Disagreements and errors

- The conclusion claims YaRN is a drop-in replacement for PI "with no downsides". On LLaMA 7B its MMLU after extension (30.0) is below NTK-by-parts (32.7) and the unextended model (35.7) [Table 2].
- §4.4 says results are in "Table 10 and Table 3"; the LLaMA comparison in the main text is Table 2.
