# Pretraining run plan: <name>

<!-- One page. Fill before launching anything longer than a smoke test. -->

## Budget

- Model: <config_budget.py command>  ->  <total / active params>, <FLOPs/token>
- Tokens: <N> (<tokens per active param>)  ->  <total FLOPs>, <GPU-hours at measured throughput>
- Measured throughput from a 30-minute smoke run: <tokens/s>, MFU <x%>

## Optimizer and schedule

| setting | value | why |
|---|---|---|
| optimizer | <AdamW (0.9, 0.95) / Muon for 2D weights + AdamW for the rest> | |
| peak LR, warmup | | <muP proxy sweep / reference config> |
| schedule | <WSD with ~10% decay / cosine> | |
| weight decay | <0.1, excluding embeddings> | OLMo 2 |
| grad clip | 1.0 | |
| batch size (tokens), ramp | | |
| precision | bf16 compute, fp32 master + optimizer state | |

## Stability insurance (decide before launch)

- [ ] QK-norm (per-head or full) or gated attention
- [ ] z-loss coefficient: <1e-4 / 1e-5>
- [ ] attention logit softcap or QK-clip (if Muon)
- [ ] repeated n-gram filter on the data (OLMo 2: 32+ repeats of 1-13 token spans)

## What to log every step

loss, grad_norm, LR, max attention logit per layer, final-layer output norm, per-tensor update/weight ratio for the embedding and output head, tokens/s. For MoE: per-expert load (max / mean), router z-loss, dropped tokens.

## Checks

- Every N steps: `python scripts/spike_monitor.py <log>` (use `--window 200` for the first few thousand steps)
- Restart rule: <e.g. two unrecovered spikes, or max attention logit above 1e4 and rising -> restart 100 steps back, skip 200-500 batches, add QK-norm>

## Evaluation

<held-out loss on a fixed set; downstream suite and the checkpoints it runs on>
