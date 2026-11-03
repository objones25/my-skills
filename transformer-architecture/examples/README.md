# Worked examples

Each file is one question answered end to end: the verdict, the numbers that decide it, the script run that produced them, and what would change the answer. Script output is from real runs (torch 2.15.0, Python 3.13, Nov 2026). The people and projects are invented; the model configs and checkpoint totals are real. Never present an example's scenario as a fact about the user's model.

| File | Question | Scripts | References |
|---|---|---|---|
| `01-sizing-a-single-gpu-model.md` | "How big a model can I pretrain on one 24 GB GPU?" | `config_budget.py` | `reference-configs.md` |
| `02-reviewing-a-port.md` | "My RoPE/GQA/RMSNorm port trains fine, is it right?" | `reference_block.py --impl`, `buggy_port.py` | `components.md` |
| `03-moe-or-dense.md` | "Should my 8B-class model be an MoE?" | `config_budget.py` presets | `moe-and-attention-variants.md` |
| `04-loss-spike-triage.md` | "Loss spiked at step 4300, what now?" | `spike_monitor.py` | `stability.md` |
| `05-grpo-answers-getting-longer.md` | "GRPO responses keep growing but accuracy is flat" | `post_training_losses.py` | `post-training.md` |
| `06-mla-port-decode-mismatch.md` | "My MLA port's generation diverges from its forward pass" | `modern_blocks.py` | `moe-and-attention-variants.md` |

The shape every answer keeps: the config or verdict first, then the budget table from a script, then the two or three load-bearing choices and what would change them.
