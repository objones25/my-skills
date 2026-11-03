# Sources

What each paper establishes, so a claim can be cited to the thing that
actually supports it rather than to the field in general. Cite these by what
they showed, not by their titles.

## Architecture

| Paper | arXiv | What it establishes |
|---|---|---|
| Attention Is All You Need | 1706.03762 | Scaled dot-product attention; the `1/sqrt(d_k)` divisor and its reason (large `d_k` pushes softmax into small-gradient regions); base config d_model 512, N 6, h 8, d_k=d_v=64, d_ff 2048, dropout 0.1, label smoothing 0.1, warmup 4000; Table 1's O(n^2 d) vs O(n d^2) complexity comparison |
| RoFormer (RoPE) | 2104.09864 | Rotation encodes relative position exactly inside the QK dot product; `theta_i = 10000^(-2i/d)`; block-diagonal 2x2 rotation structure; orthogonality; the long-term decay bound (Section 3.4.3, Abel summation) |
| Root Mean Square Layer Normalization | 1910.07467 | RMSNorm drops mean-centering and the shift term; comparable convergence at 7-64% speedup; Transformer MT 26.8 vs LayerNorm 26.6 BLEU at 7-9% faster |
| GLU Variants Improve Transformer | 2002.05202 | SwiGLU and GEGLU beat ReLU/GELU/Swish at matched parameters (log-ppl 1.636/1.633 vs 1.677); GLU variants best on most, not all, fine-tuning tasks, results noisy; the 2/3 `d_ff` reduction rule (T5-base 3072 to 2048); no biases; "we attribute their success, as all else, to divine benevolence" |
| GQA | 2305.13245 | Mean-pool KV heads within a group to convert an MHA checkpoint; uptrain at alpha=0.05; 8 groups as the middle ground; T5-XXL MHA 47.2 at 1.51s vs GQA-8 47.1 at 0.28s vs MQA 46.6 at 0.24s. Caveats: ROUGE-based, encoder-decoder only, no from-scratch comparison |
| Using the Output Embedding to Improve Language Models | 1608.05859 | Weight tying; PTB test perplexity 78.4 to 74.3 while cutting 66M to 51M params, with training perplexity rising (less overfitting); the tied matrix behaves like the output embedding; tying is harmful for word2vec-class models |

## Scaling

| Paper | arXiv | What it establishes |
|---|---|---|
| Training Compute-Optimal LLMs (Chinchilla) | 2203.15556 | Three independent estimation approaches giving a ~0.5/0.5 split between params and tokens (vs Kaplan's 0.73/0.27); model size and tokens should scale equally; Chinchilla 70B on 1.4T beats Gopher 280B on 300B at equal compute. Caveats: power-law assumed with no curvature, fitted on 70M-16B over 5B-500B tokens, extrapolation uncertain |
| Tensor Programs V (muP) | 2203.03466 | Width-dependent init variance and per-tensor LR that give zero-shot LR transfer from a narrow proxy; output weights get `1/fan_in^2` variance and `eta/fan_in` Adam LR; use `1/d` not `1/sqrt(d)` in attention; 40M proxy transfers to 6.7B |
| MiniCPM | 2404.06395 | WSD schedule; ~10% of steps in decay is sufficient; checkpoint reusability and cheap scaling-law fitting; muP-scaled optimal base LR ~0.01 across 0.036B-2.1B |
| Scaling Laws Beyond Fixed Training Durations (Hägele et al.) | 2405.18392 | Constant-LR-plus-cooldown scales as predictably as cosine; a cooldown under 20% of steps matches cosine, passing it between 10% and 20% at 124M; the needed fraction shrinks with run length (Takeaway 3, Fig. 5-6) |
| Training Dynamics of the Cooldown Stage in WSD (Dremov, Hägele, Kosson, Jaggi) | 2508.01483 | Cooldown shape as a bias-variance trade-off; higher AdamW beta2 during cooldown helps (abstract checked Nov 2026) |
| Understanding WSD: a River Valley Perspective | 2410.05192 | Why the decay phase drops loss sharply (river-valley landscape); the WSD-S variant that resumes from decayed checkpoints |

## Stability

| Paper | arXiv | What it establishes |
|---|---|---|
| On Layer Normalization in the Transformer | 2002.04745 | Post-norm has large output-layer gradients at init, hence the warmup requirement; pre-norm's gradients are well-behaved at init and it trains without warmup |
| Peri-LN | 2502.02732 | Pre-norm's residual variance growth can be exponential and persists through training; ICML 2025; normalizing the sublayer output bounds it; Gemma 2 already implements this pattern |
| 2 OLMo 2 Furious | 2501.00656 | Reordered norm plus QK-norm cut spike score 0.108 to 0.069; flat `N(0, 0.02^2)` init beat depth-position-scaled init (0.40 to 0.03); excluding embeddings from weight decay cut spike score 0.16 to 0.092; AdamW eps 1e-5 to 1e-8; repeated n-grams as a data-side spike cause; the growth exponent `lambda` |
| Small-scale proxies for large-scale instabilities | 2309.14322 | Attention logit growth and output logit divergence reproduce at small scale by raising LR; all runs above 1e4 attention logits diverged; QK-LayerNorm and z-loss as fixes; AdamW eps down to 1e-15; independent (LR-unscaled) weight decay lowers LR sensitivity; studies slow divergence, not loss spikes |
| PaLM | 2204.02311 | Origin of z-loss, `1e-4 * log^2 Z`; the restart-100-steps-back-and-skip-batches mitigation; spikes are a batch-times-parameter-state interaction, not bad data alone |
| Chameleon | 2405.09818 | Final-layer output norm growth strongly correlates with later divergence (divergences appeared as late as 20-30% into training); QK-norm essential at both sizes; z-loss 1e-5; 7B needed dropout plus z-loss, 34B needed z-loss plus norm reordering and no dropout |
| Stabilizing Transformer Training by Preventing Attention Entropy Collapse | 2303.06296 | A tight lower bound on attention entropy that decays exponentially in the spectral norm of the attention logits; low entropy observed alongside instability; sigma-Reparam |
| Methods of Improving LLM Training Stability (NVIDIA) | 2410.16682 | 830M LR-tolerance ablation: baseline converges at 6e-3, diverges at 8e-3; QK-norm alone survives 40e-3; QK-norm + softcap and QKV-norm survive 60e-3 (1.5x); norms after Proj and FC2 add nothing over QK-norm; QKV, Proj and FC2 output norms grow >2x when diverging |
| A Theory on Adam Instability | 2304.09871 | Adam can enter a state where the update has large norm and is uncorrelated with the descent direction, more likely in deep models with large batches |
| DeepNet | 2203.00555 | `alpha = (2M)^(1/4)` residual scaling and `beta = (8M)^(-1/4)` init gain lets 1000-layer transformers train |
| WeSaR | 2410.05052 | Elevated per-tensor update/parameter ratio on small-norm tensors correlates with early loss spikes |
| Gemma 3 Technical Report | 2503.19786 | "we replace the soft-capping of Gemma 2 with QK-norm"; 5:1 local:global attention; dual RoPE base (1e6 global, 1e4 local) |
| Kimi K2 | 2507.20534 | QK-clip: per-head post-step rescaling of W_q and W_k by gamma^0.5 each, so the max logit shrinks by gamma, at `tau = 100`; 15.5T tokens with zero loss spikes under Muon |
| Query-Key Normalization for Transformers (Henry et al.) | 2010.04245 | First proposal of normalizing queries and keys before the dot product (2020) |
| On the SDEs and Scaling Rules for Adaptive Gradient Algorithms | 2205.10287 | Square-root LR scaling with batch size for Adam and RMSprop (NeurIPS 2022) |
| ShortGPT (Men et al.) | 2403.03853 | Block Influence = 1 - cos(block input, output), the metric `residual_probe.py` reports; pre-norm raises input/output similarity |
| The Unreasonable Ineffectiveness of the Deeper Layers (Gromov et al.) | 2403.17887 | Pruning by angular distance between layers; deeper blocks most similar; never drop the final layer (ICLR 2025) |
| Massive Activations in Large Language Models (Sun et al.) | 2402.17762 | A few activations up to ~100,000x larger, on the first token and delimiters; why the probe skips position 0 |

## Post-training

| Paper | arXiv | What it establishes |
|---|---|---|
| DPO | 2305.18290 | The full derivation, including where `Z(x)` cancels; beta 0.1 default (0.5 for TL;DR); LR 1e-6, RMSprop, 150-step warmup, batch 64; the implicit reward; the reward-vs-KL frontier as the evaluation tool; GPT-4 judges prefer longer summaries than humans do. Limitations: tested only to 6B |
| Unintentional Unalignment (likelihood displacement) | 2410.08847 | Chosen logprob falls during DPO; "No" over "Never" raises "Yes"; Llama-3-8B-Instruct refusal rate 74.4% to 33.4%; CHES score predicts severity; CHES filtering beats adding an SFT term |
| 3D-Properties | 2406.07327 | Gradient ratio `-pi_chosen/pi_rejected` explains the rejected-side blowup and chosen-side vanishing; probability disperses to unseen responses; on-policy pairs are mildest; asymmetric beta |
| SimPO | 2405.14734 | Length-normalized reference-free reward; length normalization on DPO helps Mistral-Base (15.1 to 21.0 LC) and hurts Mistral-Instruct (26.8 to 21.7) |
| IPO | 2310.12036 | Identity transform keeps the optimum regularized under near-deterministic preferences |
| KTO | 2402.01306 | Unpaired binary feedback via a Kahneman-Tversky loss; "the default LR for DPO is 5e-7" |
| ORPO | 2403.07691 | Reference-free odds-ratio penalty fused with SFT; SFT alone raises the rejected response's likelihood in lockstep with the chosen |
| Llama 2 | 2307.09288 | SFT 2e-5 cosine, batch 64, 2 epochs, seq 4096; prompt-token loss zeroed; stopped at 27,540 SFT annotations |
| Llama 3 Herd | 2407.21783 | 6 iterative SFT-then-DPO rounds with on-policy refresh; DPO LR 1e-5, beta 0.1, plus NLL alpha 0.2; cross-document attention masking; batch-size ramping during the run |
| Zephyr | 2310.16944 | 1 SFT epoch then 3 DPO epochs is optimal; more SFT epochs cause DPO regression; skipping SFT gives MT-Bench 4.76 vs 7.00 |
| Tulu 2 | 2311.10702 | QLoRA at SFT underperforms full FT by 10-20% on AlpacaEval, narrowing to 3% at 70B |
| LIMA | 2305.11206 | Superficial Alignment Hypothesis; 1000 curated examples; the explicit EOT token to avoid conflating with the pretrained EOS |
| Does Prompt Loss Matter? | 2401.13586 | Small non-zero prompt-loss weight beats full masking on short-completion data; irrelevant on long-completion data |
| Length-Controlled AlpacaEval | 2404.04475 | Regression debiasing of pairwise win rate for the length confound |
| TR-DPO | 2404.09656 | EMA-updating the reference model every k steps beats freezing it |
| Efficient Sequence Packing without Cross-contamination | 2107.02027 | Up to 50% of tokens can be padding; a document mask recovers packing's throughput with no accuracy loss |
| LoRA | 2106.09685 | `W = W_0 + BA` with tiny rank; original paper applied it to attention only |
| LoRA Without Regret (Thinking Machines, 2025; vendor blog, not peer reviewed) | thinkingmachines.ai/blog/lora | LoRA on all layers, especially MLP, beats attention-only at matched parameters; ~10x the full fine-tuning LR; larger penalty at large batch. The vendor's Tinker docs also say 20-100x |
| LoRA Learns Less and Forgets Less | 2405.09673 | LoRA forgets less than full fine-tuning but learns less; TMLR 2024 |
| QLoRA | 2305.14314 | NF4 quantile-based 4-bit quantization of the frozen base; double quantization of the quantization constants |


## Added 2026-11-02: MoE, MLA, hybrids, gating, Muon, RL post-training

Rows below carry a fourth column saying how each paper was read (main body from arXiv HTML, Hugging Face paper markdown, or PDF; appendices only where noted). Full digests with located claims are in `references/papers/`. Papers posted after June 2026 (2608.30320, 2609.08574, 2609.22005) have no independent replication yet.

### Architecture

| Paper | arXiv | What it establishes | Checked |
|---|---|---|---|
| Gated Attention for LLMs (Qiu et al., Qwen) | 2505.06708 | Head-specific sigmoid gate on the SDPA output (gate from the pre-norm hidden state, before W_o) beat ~30 variants: 15A2B MoE PPL 6.026 to 5.761, MMLU 58.79 to 60.82; headwise gate adds 1.6M params; first-token attention 0.467 to 0.048, max activation 1053 to 94; 48-layer 1.7B dense trains at LR 8e-3 where the baseline diverges; RULER at 128K after YaRN 58.82 vs 31.65. Caveats: Qwen self-report, internal data; three internal table inconsistencies | Full text incl. App. A |
| DeepSeekMoE | 2401.06066 | Fine-grained experts (split each into m smaller, activate m times more) plus always-on shared experts; 2B/100B tokens matches GShard×1.5 and nearly Dense×16 (Pile 1.808 vs 1.808 vs 1.806); disabling the shared expert raises Pile loss 1.808 to 2.414; 16B (2.8B active) ≈ DeepSeek 7B at 40.5% FLOPs but weaker on MMLU (45.0 vs 48.2) | Full text §2-7 |
| DeepSeek-V2 (MLA) | 2405.04434 | Multi-head Latent Attention: cache a d_c latent plus a shared d_h^R RoPE key, (d_c + d_h^R)·l elements per token (34,560 for V2 = GQA with 2.25 groups); decoupled RoPE because RoPE blocks absorbing W^UK into W^Q; MLA beat MHA at 16B and 250B (MMLU 59.0 vs 57.5) with 4-14% of the cache; from-scratch 7B MMLU MQA 37.9 / GQA-8 41.2 / MHA 45.2 | Full text §2-3, App. D |
| DeepSeek-V3 | 2412.19437 | Aux-loss-free balancing (per-expert bias in top-K only, ±γ per step, γ = 0.001); sigmoid affinities normalized over selected experts; the gain is batch- vs sequence-level balance (1B: 2.258 seq-loss vs 2.253 bias vs 2.253 batch-loss); 1-depth MTP at λ 0.3→0.1 (HumanEval 44.5 to 53.7 at 229B); 671B/37B, first 3 layers dense, init std 0.006, no QK-norm or z-loss mentioned | Full text §2, §3.3, §4.2-4.5, App. B-C |
| Gated Delta Networks | 2412.06464 | S_t = α_t S_{t-1}(I − β_t k k^T) + β_t v k^T; L2-normalized q/k, short conv and output gate all needed; 1.3B/100B tokens beats Mamba2, DeltaNet and Transformer++ (Wiki ppl 16.42 vs 18.53); SWA hybrids best on recall (avg 40.1 vs 37.0) | Full text §1-4, App. B |
| Kimi Linear | 2510.26692 | 3 KDA (channel-wise-decay Gated DeltaNet) : 1 MLA layers, NoPE in MLA; 48B/3B at 1.4T tokens beats full MLA on MMLU-Pro (51.0 vs 47.2) and 128K long-context average (54.5 vs 52.2); ratio ablation 3:1 best on validation PPL (5.65 vs 5.70 at 7:1, 5.77 all-MLA); sigmoid output gate beats swish (5.65 vs 5.81); 6.3x faster TPOT at 1M | Full text §1-5 |
| On the Design of Qwen3.8-Next Architecture (posted Aug 2026) | 2608.30320 | 25B-A3B: GDN 3:1 hybrid avg 53.81 vs SWA-hybrid 51.15 vs full attention 49.87; NoPE in global layers caused endless generation after post-training; Gated Residual (4 branches, elementwise sigmoid read, no mixing matrix) loss 1.617 to 1.590, avg 50.91 to 54.66; QSA sparse attention matches full attention (MRCR 1M 20.71 to 26.44); n-gram vocab scaling lowers loss but benchmarks saturate | Full text (PDF); some tables misaligned in extraction |
| Hyper-Connections | 2409.19606 | n parallel residual streams with learned read/write/mixing; n = 1 gives nothing, n = 4 cuts OLMo-1B V2 loss 2.811 to 2.781; OLMoE-1B-7B ARC-C 41.8 to 47.8 | Full text §1-4 |
| mHC | 2512.24880 | HC's unconstrained mixing compounds to ~3000x gain and a loss surge at 27B; Sinkhorn projection onto doubly stochastic matrices (20 iterations) bounds it near 1.6; 27B BBH 43.8 / 48.9 / 51.0 (base/HC/mHC); 6.7% time overhead at n = 4 | Full text §1-6 |

### Scaling

| Paper | arXiv | What it establishes | Checked |
|---|---|---|---|
| Muon is Scalable for LLM Training (Moonlight) | 2502.16982 | Muon needs AdamW-style decoupled weight decay at scale and a per-matrix update scale 0.2·sqrt(max(A,B)) to match AdamW's update RMS, then reuses AdamW's LR and wd; ~52% of AdamW's FLOPs at compute-optimal (399M-1.5B dense); Muon-pretrained models fine-tune worse with AdamW (MMLU 50.2 vs 52.0) | Full text §1-5 |
| On the Design of Qwen3.8-Next Architecture (posted Aug 2026) | 2608.30320 | Muon + gated architecture shift the optimum: 156B-A7B optimum B 8.4M, η 1.76e-3 vs previous 4.2M, 6.8e-4 (7.8e-3 loss gap); optimum flat over sqrt(2) in η; batch-size warmup no better and 18.8% more steps | Full text (PDF) |

### Stability

| Paper | arXiv | What it establishes | Checked |
|---|---|---|---|
| ST-MoE | 2202.08906 | Router z-loss (mean squared log-sum-exp of router logits), c_z = 0.001: 3/3 seeds stable vs 4/6 baseline with slight quality gain; dropout, input jitter, update clipping, removing GEGLU or RMSNorm gains all trade quality for stability. Encoder-decoder, Adafactor | Full text §3, §5 |
| On the Design of Qwen3.8-Next Architecture (posted Aug 2026) | 2608.30320 | Stress test at constant 2-4x optimal LR: AdamW/previous structure 183 spikes per 10k steps at 4x vs 0 for Muon + Gated Residual; GatedNorm alone cuts spikes 32.0 to 3.2 per 10k at 3x; full run without qk-clip had no loss spikes; grad clip 0.5 | Full text (PDF) |
| Efficient Streaming LMs with Attention Sinks (StreamingLLM) | 2309.17453 | Evicting the first tokens from a sliding cache blows up PPL (Llama-2-13B 5158 vs 5.40 with 4 kept); position not content makes the sink; 4 sink tokens suffice; a learnable sink token in pretraining makes 1 enough (160M) | Full text §3-4 |
| gpt-oss model card | 2508.10925 | Per-head learned bias in the softmax denominator so a head can attend to nothing; alternating 128-token banded and dense layers; 64 Q / 8 KV heads of dim 64; vendor card, no ablation | §2 only |
| Do New Attention Mechanisms Actually Fix Attention Sinks? (posted Sep 2026) | 2609.08574 | ~1M-param synthetic study: next-token loss at every position creates the sink (0.316 vs 0.049 answer-only); output gate's sink reduction (0.316 to 0.261) within seed noise because the gate stayed open (mean 0.686 vs 0.116 in Qiu et al.) | Full text (PDF) |
| Abstention and Noise Filtering (posted Sep 2026) | 2609.22005 | Sink logit = abstention, value-computed gate = filtering; at 10M the gate's gain is mostly abstention, at 350M about half filtering; gains of 0.005-0.02 nats; gpt-oss-20b puts 34-48% of attention on its phantom key | Abstract, §1-2, §4 |

### Post-training

| Paper | arXiv | What it establishes | Checked |
|---|---|---|---|
| DeepSeekMath (GRPO) | 2402.03300 | Critic-free PPO with group-mean/std advantage, per-response token averaging, k3 KL in the loss; LR 1e-6, β 0.04, G 64, max length 1024; MATH 46.8 to 51.7; RL raises Maj@K not Pass@K | §4, §5.2 |
| DAPO | 2503.14476 | No KL; clip-higher (ε 0.2 / 0.28); drop all-correct and all-wrong groups; token-level loss over the group; soft overlong penalty; Qwen2.5-32B AIME24 30 → 50 with each step ablated | Full text |
| Understanding R1-Zero-Like Training (Dr. GRPO) | 2503.20783 | GRPO's 1/\|o_i\| and per-group std create length and difficulty biases; remove both and normalize by a constant; common PPO repos (trl, OpenRLHF, verl) carry the length bias | §1, §3 |
| Group Sequence Policy Optimization (GSPO) | 2507.18071 | Sequence-level, length-normalized importance ratio and clipping (ε 3e-4 / 4e-4); MoE RL converges without routing replay; ~10% of activated experts change per update on Qwen3-30B-A3B | Full text |
| DeepSeek-V3.2 | 2512.02556 | DSA: lightning indexer + top-2048 token selection retrofitted to a 128K MLA model (2.1B-token indexer warm-up, 943.7B sparse tokens); GRPO with R − mean advantage, importance-corrected k3 KL, off-policy negative-sequence masking, keep-routing and keep-sampling-mask for MoE RL | §1-3.1 |

### Efficiency and long context

| Paper | arXiv | What it establishes | Checked |
|---|---|---|---|
| YaRN | 2309.00071 | NTK-by-parts (leave fast RoPE dims, interpolate slow ones, ramp α = 1 to β = 32) plus attention temperature sqrt(1/t) = 0.1 ln s + 1; Llama 2 to 128K with 400+200 steps on 64K data; LLaMA 7B MMLU 35.7 → 30.0 after 16x extension | Full text §1-6 |
| Rope to Nope and Back Again (Cohere) | 2501.18795 | Interleaved NoPE/RoPE layers split retrieval (NoPE) and local work (RoPE); raising θ in the hybrid hurts retrieval (needles 8.04 → 6.20); RNoPE-SWA (3 RoPE SWA-4K : 1 NoPE full, no QK-norm) RULER retrieval at 256K 74.8 vs 57.1 | Full text §1-6 (Table 2 is an image) |
| Native Sparse Attention | 2502.11089 | Compressed + selected-block + sliding-window branches with sigmoid gates, trained from scratch; 27B/3B MoE: general avg 0.456 vs 0.443, LongBench 0.469 vs 0.437; 9.0x forward kernel speedup and 11.6x expected decode memory reduction at 64K | Full text §2-6 |
| DeepSeek-V3.2 | 2512.02556 | (see Post-training) DSA cuts main attention to O(Lk); no substantial regression claimed vs V3.1-Terminus | §2 |
| Kimi Linear | 2510.26692 | (see Architecture) KV cache up to 75% smaller with 3:1 hybrids | Full text §1-5 |

## Configs

Dense architecture numbers in `reference-configs.md` were read from `config.json` on
Hugging Face in August 2026. MoE configs, the olmo2-7b correction, and every exact
total were re-read on 2026-11-02; they are in `assets/hf-configs/` with a README
describing the three-way check (safetensors index, meta-device instantiation,
`config_budget.py`). Meta's repos are gated, so Llama rows come from
the `unsloth/*` mirrors, which re-upload the identical config. Token counts
come from each lab's blog post or tech report:

- Llama 3.1 8B/70B: 15T (Meta's Llama 3 blog). Llama 3.2 1B/3B: 9T, distilled.
- Qwen2.5: 18T (Qwen2.5 blog). Qwen3: 36T (Qwen3 blog).
- Gemma 2: 2T (2B) and 8T (9B). Gemma 3: 2T (1B), 12T (12B).
- OLMo 2: 4T (1B), 5T (13B). SmolLM2: 2T (135M), 11T (1.7B). SmolLM3: 11.2T.
- Phi-3-mini: 3.3T (2404.14219). DeepSeek-V3: 14.8T.
- Mistral 7B, Phi-4-mini, gpt-oss and GPT-2 have not published token counts.

## Where this file is silent

Three things commonly stated as fact have no primary source in this research
and should not be asserted:

- The "keep the update-to-parameter ratio near 1e-3" rule of thumb. WeSaR
  supports the diagnostic; the specific number is folklore.
- Watching for dead or collapsed attention heads as an early instability
  signal. The head-analysis literature is about post-hoc pruning.
- Excluding norm gains and biases from weight decay. Universal practice, no
  source isolates and tests it the way OLMo 2 does for embeddings.
