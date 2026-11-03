# 04: "Loss spiked at step 4300. Restart, lower the LR, or keep going?"

*Illustrative scenario.* A run logs loss, grad_norm, max attention logit and final-layer output norm every step. The log is `assets/example_training_log.csv` (synthetic, with planted events).

## Verdict

The loss spike recovered and is a data question; the attention logits are the real problem. They passed 1e4 at step 4423 and are still growing, which in Wortsman et al. preceded divergence in every run. Add QK-norm (or QK-clip) before the next restart; pull the spike batches separately.

## The run

```
$ python scripts/spike_monitor.py assets/example_training_log.csv
| loss.spike_score_pct | 0.0167 |
| loss.spike_steps | [4300] |
| grad_norm.spike_score_pct | 0.0667 |
| grad_norm.spike_steps | [3500, 3501, 3502, 3503] |
| grad_norm.growth_per_10k_steps | 1.6023 |
| max_attn_logit.max | 48363.07 |
| max_attn_logit.steps_over_1e4 | [4423, 4424, 4425, 4426, 4427] ... (50+ total) |
| output_norm.last_tenth_over_first_tenth | 1.524 |
```

Spike score follows OLMo 2: the share of values at least 7 standard deviations from the rolling mean of the previous 1,000.

## Reading it

- **The default missed a spike.** A second loss spike at step 2100 is in the log, but early in training the 1,000-step window still contains the loss decay, which inflates the std. `--window 200` finds both (`[2100, 4300]`). Use a short window early, or detrend.
- **Spikes vs slow growth are different problems.** The four grad-norm spikes at 3500-3503 are a batch-times-state event; OLMo 2 traced many such spikes to long repeated n-grams and removed them by filtering. The 1.6x-per-10k-steps growth is not fixed by data filtering (OLMo 2 says so explicitly); it points at init and norm placement.
- **Attention logits are the leading indicator.** Loss and grad norm are late signals. `references/stability.md` has the fixes in order of cost: QK-norm, logit softcapping, QK-clip for Muon.

## If you do restart

PaLM's mitigation for recurring spikes: restart about 100 steps before the spike and skip 200-500 batches.
