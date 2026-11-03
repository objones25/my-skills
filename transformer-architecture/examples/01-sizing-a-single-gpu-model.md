# 01: "How big a model can I pretrain on one 24 GB GPU in a weekend?"

*Illustrative scenario.* A student has one 24 GB consumer GPU, about 60 hours, and an English dataset of ~20B tokens. They plan to "scale Llama down to ~1B".

## Verdict

Train about 360M parameters, not 1B. The binding constraint is compute, not memory: at fixed compute, tokens fall in proportion to FLOPs per token, so a 1B model would see roughly a third of the tokens a 362M model sees, and fewer than 20 per parameter on any budget that does not already fit the smaller model comfortably. Use a SmolLM2-style shape (tied embeddings, GQA, d_model/n_layers near 30), not a scaled-down Llama.

## The numbers

```
$ python scripts/config_budget.py --d-model 960 --n-layers 32 --n-heads 15 --n-kv-heads 5 \
    --d-ff 2560 --vocab 49152 --seq-len 2048 --tie --tokens 20e9 --batch 8
```

| quantity | value |
|---|---|
| total parameters | 362M (embedding 13.0%, tied) |
| training FLOPs per token | 2.93B (6N 1.89B, output projection 283M, attention 755M) |
| 20B tokens | 55 tokens/param, 5.85e19 FLOPs |
| optimizer state | 5.39 GiB (16 bytes/param) |
| activations, batch 8 x 2048, checkpointed / not | 0.94 GiB / 8.44 GiB |

Memory fits comfortably on 24 GB even without checkpointing. Time is the limit. The script's GPU-hour range assumes an H100 sustaining 150-400 TFLOP/s (41-108 hours for 20B tokens); a consumer card sustains less, so measure your own tokens/second in the first hour, multiply by 60 hours, and shrink the token budget or the model to fit. The FLOPs per token above turn that measurement into a model size.

## Load-bearing choices

1. **Size from compute.** Chinchilla-optimal for 362M is 7.24B tokens; small models ship far past that (Qwen3-0.6B ran about 60,000 tokens/param per `reference-configs.md`), so 20-55 tokens/param is a reasonable floor, not a target.
2. **Tie the embeddings.** At 13% embedding share it saves 47M parameters; every model above a 21% share in the reference table ties, and the rule "tie under about 5B parameters" matches 28 of the 32 reference configs.
3. **Shape.** d_model/n_layers = 30 sits with SmolLM2 (19-45 for small models). The validator's only note is d_model = 960 not being a multiple of 128, which matters for tensor parallelism you will not use on one GPU.

## What would change it

A fixed dataset smaller than ~10B tokens caps useful size further. A serving constraint (long context, many users) would move GQA groups and depth before width.
