# 03: "Should my 8B-class model be an MoE?"

*Illustrative scenario.* A team serving on one 80 GB GPU per replica asks whether to train an MoE instead of an 8B dense model, "because MoE is cheaper".

## Verdict

MoE is cheaper per token, not per byte. Pick it when training or inference FLOPs are the bottleneck and memory is not. On one 80 GB GPU per replica, a 30B-total MoE spends most of the card on weights; the dense 8B leaves room for KV cache and batch.

## The numbers (all at seq_len 8192, bf16)

Computed with `config_budget.py` presets (totals match the published checkpoints exactly; see `--self-test`):

| preset | total | active | train FLOPs/token | bf16 weights | KV per token | KV at 8k | train state (16 B/param) |
|---|---|---|---|---|---|---|---|
| qwen3-8b (dense) | 8.19B | 8.19B | 59.9B | 15.26 GiB | 144 KiB | 1.12 GiB | 122 GiB |
| qwen3-30b-a3b | 30.5B | 3.35B | 37.6B | 56.87 GiB | 96 KiB | 768 MiB | 455 GiB |
| gpt-oss-20b | 20.9B | 4.19B | 26.6B | 38.96 GiB | 24 KiB | 195 MiB | 312 GiB |
| deepseek-v2-lite (MLA) | 15.7B | 2.66B | 21.5B | 29.26 GiB | 30.4 KiB | 243 MiB | 234 GiB |

("Active" here counts both embedding tables; labs differ, and `--self-test` shows which convention each published number uses.)

## Reading it

- Qwen3-30B-A3B needs 63% of the dense model's FLOPs per token but 3.7x its weight memory and 3.7x its training state, because every expert must live somewhere even though each token uses 8 of 128.
- The KV column is where the attention choice shows: gpt-oss's 12 sliding-window layers stop growing at 128 tokens, so only 12 layers' cache grows; DeepSeek-V2-Lite's MLA caches a 576-number latent per layer instead of per-head K and V.
- Expert parallelism across GPUs changes the memory picture; on one GPU per replica it does not apply.

## What would change it

Many GPUs per replica (memory is pooled, FLOPs dominate), a training budget that is FLOP-bound, or a quality target where the published MoE results at matched active parameters justify the memory. Check `references/moe-and-attention-variants.md` for routing and balancing choices before committing.
