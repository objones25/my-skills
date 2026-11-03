# Paper notes: post-2024 decoder-only design

One file per paper or tightly linked pair, written on 2026-11-02 from the arXiv text (Hugging Face paper markdown, arXiv HTML with LaTeX kept, or the PDF through pdftotext for the three 2026 preprints). Each header says which sections were read. Claims carry location tags like [GatedAttn Table 4]; [inference] marks synthesis that is not in the paper; "plot only" marks a claim whose evidence is a figure with no table. Each file ends with the paper's internal inconsistencies or cross-paper disagreements where any were found.

| File | Paper | Read it when |
|---|---|---|
| `gated-attention.md` | Qiu et al., Gated Attention for LLMs (arXiv 2505.06708) | adding a per-head sigmoid gate after SDPA; attention sinks, massive activations, LR tolerance |
| `gated-attn-sinks-replication.md` | Rizwan et al., Do New Attention Mechanisms Actually Fix Attention Sinks at Million-Token Context? (2609.08574, Sep 2026) | judging whether gated attention's sink fix shows up in small proxy runs |
| `attention-sinks.md` | StreamingLLM (2309.17453), gpt-oss model card (2508.10925), Wang, Abstention and Noise Filtering (2609.22005, Sep 2026) | sink tokens, learned sink logits, streaming with a sliding cache, value gates vs output gates |
| `deepseekmoe.md` | Dai et al., DeepSeekMoE (2401.06066) | fine-grained and shared experts, expert/device balance losses, MoE sizing |
| `deepseek-v2-mla.md` | DeepSeek-V2 (2405.04434) | MLA, decoupled RoPE, the per-token KV cache formula, MHA vs GQA vs MQA from scratch |
| `deepseek-v3.md` | DeepSeek-V3 (2412.19437) | aux-loss-free (bias) load balancing, sigmoid routing, multi-token prediction, FP8 training recipe |
| `st-moe.md` | Zoph et al., ST-MoE (2202.08906) | router z-loss, which MoE stabilizers cost quality |
| `gated-deltanet.md` | Yang, Kautz, Hatamizadeh, Gated Delta Networks (2412.06464, ICLR 2025) | the gated delta rule, its block design, SWA hybrids |
| `kimi-linear.md` | Kimi Team, Kimi Linear (2510.26692) | 3:1 linear-to-global hybrids, channel-wise decay (KDA), NoPE in global layers, KV cache savings |
| `qwen38-next-design.md` | Qwen Team, On the Design of Qwen3.8-Next Architecture (2608.30320, Aug 2026) | a full 2026 design study: GDN 3:1 hybrid, QSA sparse attention, Gated Residual, n-gram embeddings, Muon setup, LR/batch refit, stress-test protocol |
| `muon-moonlight.md` | Liu et al., Muon is Scalable for LLM Training (2502.16982) | switching from AdamW to Muon: weight decay, update-RMS matching, LR transfer |
| `yarn.md` | Peng et al., YaRN (2309.00071) | extending a RoPE model's context; NTK-by-parts ramp and attention temperature |
| `rope-to-nope.md` | Yang et al. (Cohere), Rope to Nope and Back Again (2501.18795), plus notes on SmolLM3 / Llama 4 | NoPE/RoPE interleaving, QK-norm vs long context, SWA:full ratios |
| `sparse-attention-nsa-dsa.md` | NSA (2502.11089) and DeepSeek-V3.2 DSA (2512.02556) | trainable sparse attention, indexers, retrofitting sparsity to a dense checkpoint |
| `hyper-connections.md` | Hyper-Connections (2409.19606) and mHC (2512.24880) | widening the residual stream; why the mixing matrix must be constrained at scale |
| `rl-post-training.md` | GRPO (2402.03300), DAPO (2503.14476), Dr. GRPO (2503.20783), GSPO (2507.18071), DeepSeek-V3.2 §3.1 | choosing an RL-with-verifiable-rewards objective; loss normalization, clipping, MoE routing in RL |

Also in this folder: `sources_additions.md` (rows for the skill's sources.md) and `skill_implications.md` (concrete edits the skill should make).

Known caveats from the distillation pass:
- Three papers were posted after June 2026 (2608.30320, 2609.08574, 2609.22005). They are read from the PDF only, have no independent replication, and two of them are small-scale (1M-350M parameters). Their headers say so.
- Almost every large-model paper here is a lab benchmarking its own model on its own harness (DeepSeek, Qwen, Kimi, OpenAI). Single-variable ablations inside a paper are more trustworthy than headline comparisons with other labs.
- Several tables were images or lost alignment in extraction: Cohere's Table 2 (RoPE vs QK-Norm vs NoPE) was not read; Qwen3.8-Next Tables 2, 3, 7-11 are quoted only where the text restates the numbers.
- Qwen3-Next and Llama 4 have no technical paper (Qwen3-Next: blog; Llama 4: Meta release materials; arXiv 2601.11659 is a third-party compilation). SmolLM3's NoPE choice is from a blog not read here.
- The gpt-oss model card was read only for §2 (architecture); it has no ablations of the learned sink.
- Appendices were mostly not read; where a hyperparameter lives only in an appendix (Dr. GRPO App. G, mHC App. A.1, DeepSeekMoE App. A) the digest says so instead of filling it in.
