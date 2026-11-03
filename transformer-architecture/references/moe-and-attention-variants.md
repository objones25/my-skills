# MoE and attention variants

What changed on top of the dense block between 2024 and 2026, how each piece works, what it costs, and the bug it invites. Equations use the toy sizes in `scripts/modern_blocks.py`, which implements and tests every component here (`python scripts/modern_blocks.py --verify`). Paper digests with located claims are in `references/papers/`.

## Contents

- [Mixture of experts](#mixture-of-experts)
- [Multi-head latent attention (MLA)](#multi-head-latent-attention-mla)
- [Gated attention](#gated-attention)
- [Attention sinks](#attention-sinks)
- [Sliding-window and hybrid layers](#sliding-window-and-hybrid-layers)
- [Muon](#muon)
- [Choosing](#choosing)

## Mixture of experts

**The idea.** Replace the MLP with E small MLPs ("experts") and a router that sends each token to k of them. Parameters grow with E; FLOPs per token grow with k. A model can then hold 30B parameters and spend 3B of them per token.

**Notation.**

| Symbol | Meaning |
|---|---|
| E | routed experts per MoE layer (`n_routed_experts`, `num_experts`, `num_local_experts`) |
| k | experts each token uses (`num_experts_per_tok`) |
| E_s | shared experts, always on (`n_shared_experts`; DeepSeekMoE) |
| d_e | hidden width of one expert (`moe_intermediate_size`, or `intermediate_size` in Mixtral, OLMoE, gpt-oss) |
| W_r | router, shape (d_model, E) |
| s | per-token expert scores, shape (tokens, E) |
| b | per-expert balancing bias, shape (E,), DeepSeek-V3 only |

**Mechanism.**

1. Score: `s = softmax(x W_r)` (Mixtral, Qwen3, DeepSeek-V2, OLMoE) or `sigmoid(x W_r)` (DeepSeek-V3).
2. Select: the top k entries of `s` (or of `s + b` in DeepSeek-V3).
3. Weight: the selected scores, renormalized to sum to 1 when the config says `norm_topk_prob: true` (Qwen3, DeepSeek-V3) and left raw when false (DeepSeek-V2, OLMoE). gpt-oss takes a softmax over only the selected logits [gpt-oss model card §2.2], which is the same thing as renormalizing a full softmax's top k (`modern_blocks.py` checks the identity).
4. Combine: `y = sum_selected g_i E_i(x) + sum_shared E_s(x)`. DeepSeek-V3 also multiplies the routed sum by `routed_scaling_factor` (2.5 in its config) and restricts selection to the best `topk_group` of `n_group` expert groups (4 of 8) before taking the top 8. A port that skips either does not reproduce the checkpoint; `modern_blocks.py` omits both for clarity.
5. Balance: without pressure, a few experts win early and the rest starve. Two families of fixes, below.

**Balancing.** The Switch Transformer loss `N * sum_i f_i P_i` (f_i the fraction of tokens routed to expert i, P_i its mean router probability) is 1 at perfect balance and grows as routing concentrates; it is added to the loss with a small coefficient (the configs here use 0.001 to 0.02; gpt-oss's config says 0.9). DeepSeek-V3 instead keeps a bias b that is added to the scores only when choosing experts, never in the weights, and nudges it after each step: down for overloaded experts, up for underloaded ones. In the released checkpoint b is a registered buffer (`e_score_correction_bias`), not a parameter, so no optimizer or weight decay touches it (checked by instantiating the config, Nov 2026). The router z-loss (mean of `logsumexp(router logits)^2`, from ST-MoE) is a separate stabilizer that keeps router logits small.

**Sizing.** The two numbers that matter are total parameters (memory, checkpoint size, optimizer state) and active parameters (FLOPs per token). `config_budget.py` reports both and refuses MoE layouts it cannot count exactly.

| Model | E / k / E_s | d_e / d_model | active / total |
|---|---|---|---|
| Mixtral-8x7B | 8 / 2 / 0 | 3.5 | 27.6% |
| OLMoE-1B-7B | 64 / 8 / 0 | 0.5 | 18.5% |
| Qwen3-30B-A3B | 128 / 8 / 0 | 0.375 | 11.0% |
| DeepSeek-V2-Lite | 64 / 6 / 2 | 0.69 | 16.9% |
| gpt-oss-20b | 32 / 4 / 0 | 1.0 | 20.0% |
| DeepSeek-V3 | 256 / 8 / 1 | 0.29 | 5.6% |

(Active counts both embedding tables; computed by `config_budget.py` from the presets.) The trend is toward many narrow experts ("fine-grained") and lower active fractions.

**"Active parameters" is not one number.** Qwen's model card counts both embedding tables (3.35B computed, 3.3B claimed); the gpt-oss card counts the unembedding but not the input embedding ("Unembedding parameters are counted towards active, but not embeddings" [gpt-oss model card Table 1]); DeepSeek-V2-Lite's 2.4B also matches that convention. Comparing two labs' active counts without converting is comparing different quantities.

**Bugs it invites.**

| Bug | Symptom |
|---|---|
| Balancing bias added to the gate weights too | Every bias update changes the layer's output; trains anyway. `modern_blocks.py --bug moe-bias-in-weights` |
| `norm_topk_prob` mismatch against a checkpoint | Each token's routed output scaled by a different factor; silent quality loss on a pretrained model |
| Router in bf16 | ST-MoE keeps the router computation in fp32 for stability [ST-MoE §3.3]; check your framework's default |
| Counting `intermediate_size` as the dense width in Mixtral-style configs | In Mixtral, OLMoE and gpt-oss it is the expert width; in Qwen3-MoE and DeepSeek it is the dense layers' width and `moe_intermediate_size` is the expert's |
| Comparing MoE and dense at equal total parameters | Compare at equal active parameters (FLOPs) and report memory separately |

## Multi-head latent attention (MLA)

**The idea.** Cache one small latent vector per token instead of per-head keys and values, and reconstruct K and V from it on the fly. DeepSeek-V3 caches 576 numbers per token per layer where an expanded per-head cache would hold 40,960 (`config_budget.py --preset deepseek-v3`).

**Notation** (DeepSeek-V2/V3 config names).

| Symbol | Meaning | DeepSeek-V3 |
|---|---|---|
| `kv_lora_rank` | width of the cached latent c_KV | 512 |
| `qk_rope_head_dim` | width of the RoPE key, one per token, shared by all heads, also cached | 64 |
| `qk_nope_head_dim` | per-head query/key width without RoPE | 128 |
| `v_head_dim` | per-head value width | 128 |
| `q_lora_rank` | optional low-rank query path (None in V2-Lite) | 1536 |

**Mechanism.**

1. Compress: `c_KV = W_DKV x` (512 wide) and `k_R = RoPE(W_KR x)` (64 wide). Cache both.
2. Expand: `k_C = W_UK c_KV`, `v = W_UV c_KV`, per head.
3. Score: `q_C . k_C + q_R . k_R`, scaled by `1/sqrt(qk_nope + qk_rope)`.
4. Absorb at decode: `q_C . (W_UK c) = (W_UK^T q_C) . c`, so scores come straight from the cached latent and K is never materialized; `W_UV` is folded into the output side the same way.

**Why RoPE gets its own key.** Step 4 needs nothing position-dependent between `W_UK` and `c`. RoPE is a rotation `R_t` that depends on position, and `q . (R_t W_UK c)` cannot be rewritten as `(something) . c`. So RoPE is applied only to the small shared key `k_R`, which is cached directly. Applying RoPE to `k_C` trains fine (the explicit path is self-consistent) and breaks absorbed decode; `modern_blocks.py` detects it.

**Costs.** Parameters per layer are in `config_budget.py --show-work`; the latent and query norms add `kv_lora_rank` (+ `q_lora_rank`) gains. A Hugging Face-style implementation that caches expanded K and V gives up the memory saving; check what your serving stack caches.

## Gated attention

**The idea.** Multiply each head's attention output by a sigmoid gate computed from the layer's input: `Y <- Y * sigmoid(X W_g)`. Unlike softmax weights, the gate can go to zero, so a head with nothing useful to read can output nothing instead of dumping its mass on token 0.

**Costs.** Elementwise gate: `d_model x n_heads*head_dim` parameters per layer (as large as `W_q`). Headwise: `d_model x n_heads`. `modern_blocks.py` checks both counts and that a gate can silence a head while softmax rows always sum to 1.

**Evidence.** Qiu et al. compared about 30 gating variants on 15B-A2B MoE models (400B tokens in the main ablation table) and 1.7B dense models (up to 3.5T tokens). At 15B-A2B, the head-specific elementwise gate after SDPA lowered perplexity from 6.026 to 5.761, cut first-token attention from 0.467 to 0.048 and the largest activation from 1053 to 94; the cheaper headwise gate (one scalar per head) reached 5.792 and 0.073 [Gated Attention Tables 1, 4]. The gate let the 1.7B dense model train at LR 8e-3 where the baseline diverged [Table 2]. Bugs they report: computing the gate from the value path keeps the sink; a gate shared across heads keeps most of the perplexity gain (5.801) but loses the sink reduction (first-token attention 0.301) [Table 4]. Caveats: a Qwen self-report on internal data with three internal table inconsistencies (`references/papers/gated-attention.md`), and a September 2026 replication at about 1M parameters did not reproduce the sink reduction, though its gates never became sparse, so it is weak evidence against (`references/papers/gated-attn-sinks-replication.md`). Qwen3.8-Next uses the gate in production [§3.1].

## Attention sinks

**The idea.** Causal LMs put large attention mass on the first token whether or not it is relevant, because softmax weights must sum to 1 and the first token is visible to every query. Two responses: keep the first tokens in the cache when streaming (StreamingLLM), or give each head a learned logit `s_h` that only appears in the denominator, `p_ij = exp(l_ij) / (exp(s_h) + sum_k exp(l_ik))`, as gpt-oss does (`sinks`, one per head, in the checkpoint; 64 parameters per layer in gpt-oss-20b). The gpt-oss card cites Miller's "Attention is off by one" and StreamingLLM for this.

**Bug.** Implementing the sink by adding `s_h` to every logit does nothing: softmax ignores a constant shift, the parameter's gradient is zero, and rows still sum to 1. It must be a separate column (`modern_blocks.py --bug sink-added-to-logits`).

## Sliding-window and hybrid layers

**Sliding window.** A layer whose queries see only the last W keys caches at most W tokens, so its cache stops growing. Patterns in shipping configs: every layer (Mistral 7B v0.1, W 4096), 1 global in 6 (Gemma 3, W 512, 22 of 26 layers local in the 1B), alternating (gpt-oss, W 128). `config_budget.py` counts only full-attention layers in per-token KV growth: gpt-oss-20b's cache grows 24 KiB per token, not 48. Off-by-one conventions differ (does a window of W include the current token?); `modern_blocks.causal_mask` follows Mistral's "at most W keys including itself" and its decode test pins it.

**Linear-attention and state-space hybrids.** Some 2025-2026 models replace most attention layers with layers whose state has a fixed size (Gated DeltaNet in Qwen3-Next, Mamba layers in other hybrids), keeping one full-attention layer every few. The KV cache then grows only in the full layers. `config_budget.py --from-hf-config` refuses these configs rather than miscount them. Evidence: Kimi Linear's 3 linear : 1 MLA layout beat full MLA at 48B-A3B on 1.4T tokens (MMLU-Pro 51.0 vs 47.2) [Kimi Linear §5]. Its ratio and gate ablation, run on a small 16-layer scaling-law model, gave validation perplexity 5.65 at 3:1, 5.66 at 1:1, 5.70 at 7:1 and 5.77 all-MLA, and a sigmoid output gate beat swish (5.65 vs 5.81) [Table 1]; Qwen3.8-Next's Gated DeltaNet 3:1 hybrid averaged 53.81 against 49.87 for full attention at 25B-A3B [§2]. Both are self-reports on the labs' own data. Digests: `references/papers/gated-deltanet.md`, `references/papers/kimi-linear.md`, `references/papers/qwen38-next-design.md`.

## Muon

**The idea.** For each 2D weight matrix, replace the gradient's momentum with its nearest orthogonal matrix (a few Newton-Schulz iterations), so the update has every singular value near 1 regardless of the gradient's conditioning. Embeddings, the output head, norms and biases stay on AdamW.

**In PyTorch.** `torch.optim.Muon` ships in torch 2.15 (checked with `inspect`, Nov 2026): defaults `lr=1e-3, weight_decay=0.1, momentum=0.95, nesterov=True, ns_steps=5`, coefficients `(3.4445, -4.775, 2.0315)`. Its `adjust_lr_fn` sets how the LR scales with the matrix shape (A, B):

| `adjust_lr_fn` | scale | follows |
|---|---|---|
| `None` / `"original"` (default) | `sqrt(max(1, A/B))` | Keller Jordan's implementation |
| `"match_rms_adamw"` | `0.2 * sqrt(max(A, B))` | Moonshot (Moonlight): update RMS about 0.2, like AdamW's, so AdamW-tuned LR and weight decay roughly carry over |
| `"spectral_unclamped"` | `sqrt(A/B)` | Bernstein |

Measured with `modern_blocks.py`: `match_rms_adamw` gives update RMS 0.17 on both a 256x64 and a 64x256 matrix, below the nominal 0.2 because Newton-Schulz leaves singular values between 0.68 and 1.14 rather than exactly 1.

**Stability.** Muon runs grow attention logits; Kimi K2's QK-clip (`stability.md`) was built for exactly this.

## Choosing

```
memory per replica is the binding constraint ─────────────► dense, GQA; MLA if long context
FLOPs (training or serving) bind, memory is pooled ───────► MoE, fine-grained, measure active/total
long context, many concurrent sequences ──────────────────► MLA, or sliding-window / hybrid layers
heads collapsing onto token 0, sink-driven outliers ──────► gated attention or learned sinks
training with Muon ───────────────────────────────────────► QK-clip or QK-norm from step 0
```
