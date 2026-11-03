# Reference configs: what shipping models actually do

Every row was read from the model's `config.json` on Hugging Face (August 2026)
or its tech report. Use this to answer "is my choice weird?" before you commit
compute to it. A config that sits inside the observed spread needs no defense.
One outside it needs a reason you can say out loud.

## Contents

- [Dense models](#dense-models)
- [Derived ratios](#derived-ratios)
- [Tokens per parameter](#tokens-per-parameter)
- [MoE and exotic attention](#moe-and-exotic-attention)
- [Consensus: settled](#consensus-settled)
- [Spread: genuinely open](#spread-genuinely-open)
- [Reading a config.json without being fooled](#reading-a-configjson-without-being-fooled)

## Dense models

`n_kv` = `num_key_value_heads`. `hd` = declared `head_dim`. Bias column is
attention bias. "tied" = `tie_word_embeddings`.

| Model | params | d_model | L | heads | n_kv | hd | d_ff | vocab | rope_theta | norm eps | tied | bias | sliding window |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GPT-2 124M | 124M | 768 | 12 | 12 | 12 | 64 | 3072 | 50257 | learned abs | 1e-5 LN | yes | all | none |
| GPT-2 XL | 1.56B | 1600 | 48 | 25 | 25 | 64 | 6400 | 50257 | learned abs | 1e-5 LN | yes | all | none |
| SmolLM2-135M | 135M | 576 | 30 | 9 | 3 | 64 | 1536 | 49152 | 100000 | 1e-5 | yes | none | none |
| SmolLM2-360M | 362M | 960 | 32 | 15 | 5 | 64 | 2560 | 49152 | 100000 | 1e-5 | yes | none | none |
| Gemma3-270M | 268M | 640 | 18 | 4 | 1 | 256 | 2048 | 262144 | 1e6 / 1e4 local | 1e-6 | yes | none | 512, 5:1 |
| Qwen2.5-0.5B | 494M | 896 | 24 | 14 | 2 | 64 | 4864 | 151936 | 1000000 | 1e-6 | yes | qkv | off |
| Qwen3-0.6B | 596M | 1024 | 28 | 16 | 8 | 128 | 3072 | 151936 | 1000000 | 1e-6 | yes | none | none |
| Gemma3-1B | 1.00B | 1152 | 26 | 4 | 1 | 256 | 6912 | 262144 | 1e6 / 1e4 local | 1e-6 | yes | none | 512, 5:1 |
| Llama-3.2-1B | 1.24B | 2048 | 16 | 32 | 8 | 64 | 8192 | 128256 | 500000 | 1e-5 | yes | none | none |
| OLMo2-1B | 1.48B | 2048 | 16 | 16 | 16 | 128 | 8192 | 100352 | 500000 | 1e-6 | no | none | none |
| Qwen2.5-1.5B | 1.54B | 1536 | 28 | 12 | 2 | 128 | 8960 | 151936 | 1000000 | 1e-6 | yes | qkv | off |
| SmolLM2-1.7B | 1.71B | 2048 | 24 | 32 | 32 | 64 | 8192 | 49152 | 130000 | 1e-5 | yes | none | none |
| Qwen3-1.7B | 1.72B | 2048 | 28 | 16 | 8 | 128 | 6144 | 151936 | 1000000 | 1e-6 | yes | none | none |
| Gemma2-2B | 2.61B | 2304 | 26 | 8 | 4 | 256 | 9216 | 256000 | 10000 | 1e-6 | yes | none | 4096, 1:1 |
| Qwen2.5-3B | 3.09B | 2048 | 36 | 16 | 2 | 128 | 11008 | 151936 | 1000000 | 1e-6 | yes | qkv | off |
| SmolLM3-3B | 3.08B | 2048 | 36 | 16 | 4 | 128 | 11008 | 128256 | 5000000 | 1e-6 | yes | none | NoPE every 4th |
| Llama-3.2-3B | 3.21B | 3072 | 28 | 24 | 8 | 128 | 8192 | 128256 | 500000 | 1e-5 | yes | none | none |
| Phi-3-mini | 3.82B | 3072 | 32 | 32 | 32 | 96 | 8192 | 32064 | 10000 | 1e-5 | no | none | 2047 |
| Phi-4-mini | 3.84B | 3072 | 32 | 24 | 8 | 128 | 8192 | 200064 | 10000 + LongRoPE | 1e-5 | yes | none | off |
| Gemma3-4B | 3.88B | 2560 | 34 | 8 | 4 | 256 | 10240 | 262144 | 1e6 / 1e4 local | 1e-6 | yes | none | 1024, 5:1 |
| Qwen3-4B | 4.02B | 2560 | 36 | 32 | 8 | 128 | 9728 | 151936 | 1000000 | 1e-6 | yes | none | none |
| Mistral-7B v0.1 | 7.24B | 4096 | 32 | 32 | 8 | 128 | 14336 | 32000 | 10000 | 1e-5 | no | none | 4096, every layer |
| OLMo2-7B | 7.30B | 4096 | 32 | 32 | 32 | 128 | 11008 | 100352 | 500000 | 1e-6 | no | none | none |
| Qwen2.5-7B | 7.62B | 3584 | 28 | 28 | 4 | 128 | 18944 | 152064 | 1000000 | 1e-6 | no | qkv | off |
| Llama-3.1-8B | 8.03B | 4096 | 32 | 32 | 8 | 128 | 14336 | 128256 | 500000 | 1e-5 | no | none | none |
| Qwen3-8B | 8.19B | 4096 | 36 | 32 | 8 | 128 | 12288 | 151936 | 1000000 | 1e-6 | no | none | none |
| Gemma2-9B | 9.24B | 3584 | 42 | 16 | 8 | 256 | 14336 | 256000 | 10000 | 1e-6 | yes | none | 4096, 1:1 |
| Gemma3-12B | 11.8B | 3840 | 48 | 16 | 8 | 256 | 15360 | 262144 | 1e6 / 1e4 local | 1e-6 | yes | none | 1024, 5:1 |
| OLMo2-13B | 13.7B | 5120 | 40 | 40 | 40 | 128 | 13824 | 100352 | 500000 | 1e-6 | no | none | none |
| Qwen2.5-14B | 14.8B | 5120 | 48 | 40 | 8 | 128 | 13824 | 152064 | 1000000 | 1e-5 | no | qkv | off |
| Qwen3-14B | 14.8B | 5120 | 40 | 40 | 8 | 128 | 17408 | 151936 | 1000000 | 1e-6 | no | none | none |
| Llama-3.1-70B | 70.6B | 8192 | 80 | 64 | 8 | 128 | 28672 | 128256 | 500000 | 1e-5 | no | none | none |

Every model from SmolLM2 down the table uses RMSNorm, RoPE, a gated MLP
(SwiGLU or Gemma's GeGLU), and pre-norm. GPT-2 is in the table as the "before"
picture: LayerNorm, learned absolute positions, plain GELU MLP, biases
everywhere.

## Derived ratios

| Model | d_ff/d_model | d_model/L | hd vs d/heads | GQA group | embed share |
|---|---|---|---|---|---|
| GPT-2 124M | 4.00 | 64.0 | 64 = 64 | 1x | 31.0% |
| SmolLM2-135M | 2.67 | 19.2 | 64 = 64 | 3x | 21.0% |
| Gemma3-270M | 3.20 | 35.6 | 256 vs 160 | 4x | 62.6% |
| Qwen2.5-0.5B | 5.43 | 37.3 | 64 = 64 | 7x | 27.6% |
| Qwen3-0.6B | 3.00 | 36.6 | 128 vs 64 | 2x | 26.1% |
| Gemma3-1B | 6.00 | 44.3 | 256 vs 288 | 4x | 30.2% |
| Llama-3.2-1B | 4.00 | 128.0 | 64 = 64 | 4x | 21.3% |
| OLMo2-1B | 4.00 | 128.0 | 128 = 128 | 1x | 13.8% |
| SmolLM3-3B | 5.38 | 56.9 | 128 = 128 | 4x | 8.5% |
| Llama-3.2-3B | 2.67 | 109.7 | 128 = 128 | 3x | 12.3% |
| Qwen3-4B | 3.80 | 71.1 | 128 vs 80 | 4x | 9.7% |
| Mistral-7B | 3.50 | 128.0 | 128 = 128 | 4x | 1.8% |
| OLMo2-7B | 2.69 | 128.0 | 128 = 128 | 1x | 5.6% |
| Qwen2.5-7B | 5.29 | 128.0 | 128 = 128 | 7x | 7.2% |
| Llama-3.1-8B | 3.50 | 128.0 | 128 = 128 | 4x | 6.5% |
| Qwen3-8B | 3.00 | 113.8 | 128 = 128 | 4x | 7.6% |
| Llama-3.1-70B | 3.50 | 102.4 | 128 = 128 | 8x | 1.5% |

Observed ranges, which is what the validator in `scripts/config_budget.py`
checks against:

```
d_ff / d_model      2.67 ......................... 6.00     no consensus value
d_model / n_layers  19 (SmolLM2-135M) ........... 128       still 33-128 above 1B
head_dim            64 (sub-1B) .. 128 (default) .. 256 (Gemma)
GQA group           1x (OLMo2) .. 4x (modal) .. 8x (Llama-70B)
embed share         1.5% (Llama-70B) ............. 63% (Gemma3-270M)
```

## Tokens per parameter

Chinchilla's compute-optimal answer is about 20. Almost nothing ships at 20,
because Chinchilla optimizes training cost and everyone shipping a small model
is optimizing inference cost instead. Deliberate overtraining is the norm.

| Model | tokens | tokens/param |
|---|---|---|
| OLMo2-13B | 5T | 365 |
| Gemma2-2B | 2T | 765 |
| Phi-3-mini | 3.3T | 864 |
| Gemma2-9B | 8T | 866 |
| Gemma3-12B | 12T | ~1020 |
| Llama-3.1-8B | 15T | 1868 |
| Gemma3-1B | 2T | 2000 |
| Qwen2.5-7B | 18T | 2364 |
| OLMo2-1B | 4T | 2694 |
| SmolLM3-3B | 11.2T | 3642 |
| Qwen3-8B | 36T | 4395 |
| SmolLM2-1.7B | 11T | 6428 |
| Llama-3.2-1B | 9T (distilled) | 7283 |
| SmolLM2-135M | 2T | 14868 |
| Qwen2.5-0.5B | 18T | 36437 |
| Qwen3-0.6B | 36T | 60398 |

The pattern: tokens/param rises sharply as parameters fall. A 500M model
trained at Chinchilla-optimal (10B tokens) would be much worse than the same
model trained on 18T, and the extra training is paid once while the inference
saving is paid on every request.

## MoE and exotic attention

| Model | total / active | d_model | L | attention | experts | vocab |
|---|---|---|---|---|---|---|
| DeepSeek-V3 | 671B / 37B | 7168 | 61 (3 dense, 58 MoE) | MLA: q_lora 1536, kv_lora 512, qk_nope 128, qk_rope 64, v_head 128, 128 heads | 256 routed + 1 shared, top-8, expert d_ff 2048 | 129280 |
| gpt-oss-20b | 20.9B / 3.6B | 2880 | 24, alternating sliding(128)/full | 64 Q / 8 KV, hd 64, learned attention-sink bias, `attention_bias: true` | 32 experts, top-4, expert d_ff = d_model | 201088 |
| gpt-oss-120b | 117B / 5.1B | 2880 | 36, alternating | same | 128 experts, top-4 | 201088 |
| Mixtral-8x7B | 46.7B / 12.9B | 4096 | 32 | GQA 32 Q / 8 KV, hd 128 | 8 experts, top-2, expert d_ff 14336 (3.5x) | 32000 |
| OLMoE-1B-7B | 6.92B / 1.28B | 2048 | 16 | MHA 16 heads, full-width QK-norm | 64 experts, top-8, expert d_ff 1024 (0.5x) | 50304 |
| Qwen3-30B-A3B | 30.5B / 3.35B | 2048 | 48 | GQA 32 Q / 4 KV, hd 128, per-head QK-norm | 128 experts, top-8, expert d_ff 768 (0.375x) | 151936 |
| DeepSeek-V2-Lite | 15.7B / 2.66B | 2048 | 27 (1 dense, 26 MoE) | MLA: no q_lora, kv_lora 512, qk_nope 128, qk_rope 64, v_head 128, 16 heads | 64 routed + 2 shared, top-6, expert d_ff 1408 (0.69x) | 102400 |

Totals in this table are exact checkpoint counts (`assets/hf-configs/README.md`); the
`config_budget.py` presets reproduce every one. The active column mixes conventions:
the first three rows are as each lab published them (DeepSeek-V3's 37B and gpt-oss's
3.6B and 5.1B), while the four new rows are computed with both embedding tables
counted. Under that convention gpt-oss-20b is 4.19B. Published active figures use three
different conventions: Qwen counts both tables (3.3B claimed, 3.35B computed), gpt-oss
counts the unembedding but not the input embedding (its model card says so in Table 1;
3.61B), and DeepSeek-V2-Lite's 2.4B matches the gpt-oss convention (2.45B). Compare
active counts only under one convention; `config_budget.py --self-test` prints all three.

Two things here are worth carrying into any MoE design. Fine-grained MoE uses
narrow experts (expert `d_ff` 0.29x `d_model` in DeepSeek-V3, 0.375x in Qwen3-30B-A3B,
versus 2.67-6x for a dense MLP and 3.5x in the coarse-grained Mixtral) because the
capacity comes from expert count, not expert width. Design details for routing,
balancing and the attention variants are in `moe-and-attention-variants.md`.
And gpt-oss re-added linear biases (`attention_bias: true`, plus biases in the
router and experts) and a separate learned per-head sink logit in the softmax
denominator [gpt-oss model card §2.2]. The card does not link the two, so do not
read the biases as existing to support the sinks. Either way it is a reminder that
"everyone does X" is a prior, not a rule.

## Consensus: settled

These are the defaults. Deviating is fine but you should be able to say why.

| Choice | Value | Holdouts |
|---|---|---|
| Normalization | RMSNorm | GPT-2 era only |
| Norm placement | pre-norm, or pre + post (Gemma 2/3, "peri-LN") | OLMo 2 normalizes the sublayer output instead |
| Position | RoPE | GPT-2 era; SmolLM3 drops RoPE on every 4th layer (NoPE) |
| MLP | gated (SwiGLU / GeGLU), 3 matrices | GPT-2 era |
| Linear biases | none | GPT-2; Qwen2/2.5 keeps QKV bias; gpt-oss re-added them |
| head_dim | 128 above 1B, 64 below | Gemma uses 256 at every size |
| Attention kernel | FlashAttention (exact, not an approximation) | |
| Precision | bf16 compute, fp32 master weights and optimizer state | |
| Optimizer | AdamW, beta (0.9, 0.95), grad clip 1.0 | PaLM used Adafactor; Muon for 2D weights in Kimi K2, Moonlight, Kimi Linear and Qwen3.8-Next (`stability.md`) |

## Spread: genuinely open

Nobody has settled these. Pick with a reason, and expect the reason to be
about your deployment rather than about quality.

| Choice | Range | What decides it |
|---|---|---|
| `d_ff / d_model` | 2.67 to 6.0 | FLOPs-per-parameter preference. Wide MLP concentrates parameters where they are cheapest per FLOP. |
| GQA group size | 1x to 8x | KV-cache budget at your serving batch and context. OLMo 2 skips GQA entirely at 13B to keep the model simple. |
| `tie_word_embeddings` | scale- and family-dependent | Every model above 21% embedding share ties. Below that, size predicts better than share (tie under ~5B: 28 of 32 correct, vs 25 of 32 for a 10% share rule). Gemma ties at 9B and 12B anyway (shares 9.9%, 8.5%). |
| `rope_theta` | 1e4 to 1e7 | Two strategies, and reading theta alone misleads. Either large native theta (Llama 5e5, Qwen 1e6) or small theta plus explicit scaling at the end (Mistral 1e4 + sliding window; DeepSeek 1e4 + YaRN x40). |
| Vocabulary size | 32k to 262k | Language and code coverage. A large vocab shortens sequences (helps attention cost) and inflates embedding share (hurts small models). |
| Sliding window | none to 5:1 local:global | Context length and KV budget. Trend is toward local-heavy as contexts grew: Mistral every layer, Gemma 2 and gpt-oss at 1:1 (alternating), Gemma 3 at 5:1. |
| QK-norm vs logit softcap | either, or both | Gemma 2 chose softcap, Gemma 3 replaced it with QK-norm one generation later. NVIDIA's ablation found the two are complementary, not substitutes. |

## Reading a config.json without being fooled

Four traps, all of them present in real configs in the table above.

**`sliding_window` present does not mean enabled.** Qwen2.5 ships a
`sliding_window` value with `use_sliding_window: false`. Check the flag.

**`num_key_value_heads == num_attention_heads` does not mean MHA.**
DeepSeek-V3's config reports 128 for both, but it uses MLA: K and V are
reconstructed from a 512-dim shared latent, and there is no per-head KV cache
at all. Check `model_type` and the presence of `kv_lora_rank`.

**`head_dim` is not always `d_model / n_heads`.** Qwen3-0.6B declares 128 with
`d_model/n_heads = 64`; Gemma declares 256 everywhere. Both over-provision
attention relative to the backbone on purpose. If you compute `head_dim`
yourself instead of reading the field, your parameter count will be wrong.

**Some real architecture choices are not in the config at all.** OLMo 2's
reordered norm and QK-norm, Gemma 3's QK-norm, and any custom init are
invisible from `config.json`. If the number matters, read the tech report.

## Source

Dense configs read from Hugging Face `config.json`, August 2026; MoE rows and the
exact totals re-read on 2026-11-02 (`assets/hf-configs/`). Meta repos are
gated, so Llama rows come from the `unsloth/*` mirrors, which re-upload the
identical config. Token counts from each lab's blog or tech report; see
`sources.md`.
