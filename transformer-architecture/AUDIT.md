# Audit, 2026-11-02

Previous version: 1.1.0 (verified against torch 2.13.0 and 2.14.1). SKILL.md, 5 reference files, 3 scripts, 38 tests; no evals, examples, assets or paper digests. Its 38 tests still passed on torch 2.15.0.

## Errors found in the previous version

| Claim or behavior | Evidence | Fix |
|---|---|---|
| `olmo2-7b` preset counted 7,298,363,392 parameters | checkpoint holds 7,298,617,344 (safetensors index and meta-device instantiation agree). OLMo 2 normalizes the full Q and K projections, not each head | `qk_norm_scope="full"`; preset now exact and in the self-test |
| "QK-norm costs `2*head_dim` parameters per layer" (SKILL.md, stability.md, which attributed it to OLMo 2) | true for per-head QK-norm (Qwen3, Gemma 3); OLMo 2 and OLMoE spend `n_heads*head_dim + n_kv_heads*head_dim` (8,192 per layer in OLMo-2-7B) | both scopes documented |
| `--from-hf-config` ignored QK-norm | Qwen3-8B came out 9,216 parameters short | QK-norm scope read from `model_type`; Qwen3-8B now exact |
| "eps outside the sqrt turns an all-zero vector into NaN" | true for `x*rsqrt(ms)+eps`; the common `x/(sqrt(ms)+eps)` is finite forward and NaN in the backward pass, and passed all three of `--verify`'s RMSNorm checks | gradient check added to `--verify`, detector covers both, docs rewritten |
| `--verify --impl` reported a reference-only property as a failure of the user's `repeat_kv` | ran a buggy port: the "repeat is a permutation" check failed because it was testing the user's function | skipped under `--impl` with a note |
| "GQA cuts the cache 4x for ~0.1 points"; "`n_kv_heads` is an inference decision, not a quality decision" | the 0.1 comes from T5 uptraining scored by ROUGE; DeepSeek-V2's from-scratch 7B ablation: MMLU 45.2 MHA, 41.2 GQA-8, 37.9 MQA [App. D.1 Table 8], checked against the paper text | qualified; MLA added as an option |
| "Hägele et al. (arXiv:2508.01483)" for 20% cooldowns | 2508.01483 is Dremov, Hägele, Kosson, Jaggi on cooldown shape; the 20% result is Hägele et al. 2405.18392 (Takeaway 3) | attribution fixed, both papers registered |
| "gpt-oss reversed the no-bias consensus specifically to support attention sinks" | the model card describes the sink as a separate per-head denominator term and does not link it to the QKV biases | reworded |
| Optimizer row: "everyone except PaLM and Kimi K2" uses AdamW | Moonlight, Kimi Linear, Qwen3.8-Next train with Muon | updated, Muon section added |
| `config_budget.py` refused every MoE or MLA config | | MoE, MLA, sinks, sliding windows, sandwich norms supported and exact (below) |

## Gaps found

No MoE, MLA, gated attention, sinks, sliding-window or hybrid layers, Muon, or RL post-training (GRPO family); no evals; no worked examples; no templates; no paper digests; no structural tests; torch's own `torch.optim.Muon` (present in 2.15) unmentioned.

## Added

- `config_budget.py`: MoE (routed, shared, leading dense layers, router and expert biases), MLA (with and without the query low-rank path), learned sinks, per-head or full-width QK-norm, sandwich norms, sliding-window KV accounting, active parameters under three lab conventions. Presets 8 to 15; every one with a published total reproduces it exactly (13, including DeepSeek-V3 at 671,026,404,352).
- `modern_blocks.py` (34 checks): gated attention, sinks, sliding-window rolling cache, MoE sparse dispatch vs dense oracle, DeepSeek-V3 bias balancing, Switch loss, router z-loss, MLA absorbed decode vs full forward, Muon update RMS; 4 bug detectors.
- `post_training_losses.py` (20 checks): DPO, IPO, SimPO, GRPO, Dr. GRPO, DAPO, GSPO, k3 KL; length and difficulty bias shown numerically.
- `spike_monitor.py`: OLMo 2 spike score, grad-norm growth, attention-logit and output-norm flags.
- `references/moe-and-attention-variants.md`; RL section in `post-training.md`; Muon, gating, stress-test sections and five new disagreements in `stability.md`; four MoE rows in `reference-configs.md`.
- `references/papers/`: 16 digests (gated attention and its replication, sinks, DeepSeekMoE, DeepSeek-V2 MLA, DeepSeek-V3, ST-MoE, Gated DeltaNet, Kimi Linear, Qwen3.8-Next, Moonlight, YaRN, Rope to NoPE, NSA and DSA, hyper-connections and mHC, GRPO family).
- `examples/`: six worked answers with real script output; `examples/buggy_port.py`.
- `assets/`: 9 real `config.json` files with a provenance README, a synthetic training log, an architecture write-up template, a run plan.
- `evals/evals.json`: 16 cases. Tests: 38 to 181, including `test_skill_structure.py` (paths resolve, every cited arXiv id registered, no em dashes, every script tested, evals and examples well formed).

## How things were checked

- Library APIs with Python's `inspect` on installed torch 2.15.0, transformers 5.22.1 (SDPA, `nn.RMSNorm`, `flex_attention`, `torch.optim.Muon` source including `_adjust_lr`). Context7 for pytest and Muon docs.
- Parameter totals three ways: safetensors `total_size`, meta-device instantiation with transformers, and the script.
- Papers through Hugging Face paper markdown and the Firecrawl paper index (full text or passages; each digest header says which). Load-bearing numbers re-read in the paper by the main session: OLMo 2 spike score definition (§3.2), DeepSeek-V2 Tables 8-9, gpt-oss card Table 1 and §2.2, Hägele et al. Takeaway 3, DAPO Eq. 10 and §4.1, Dr. GRPO §3.1-3.2, GSPO Eqs. 5-7.
- An independent fact-check pass on the new material by a separate agent that re-read the papers and reran every script: 4 errors and 11 imprecisions, all fixed (CHANGELOG lists them).

## Not verified

- Qwen3-Next and Llama 4 have no technical paper; Qwen3-Next's use of gated attention rests on a third-party abstract. SmolLM3's NoPE choice is from a blog not read.
- Three papers posted after June 2026 (2608.30320, 2609.08574, 2609.22005) have no independent replication; read from PDF only.
- Cohere's QK-norm vs long-context numbers (Table 2 is an image).
- GSPO's exact clip range (the paper says it differs from GRPO's by orders of magnitude; the value was not extracted).
