# Changelog

## 1.2.0, 2026-11-02: MoE, MLA and 2025-2026 components; audit fixes

Audit details, with the evidence for each fix, are in `AUDIT.md`.

### Fixed

| Problem | Fix |
|---|---|
| `olmo2-7b` preset off by 253,952 parameters (full-width QK-norm counted as per-head) | `qk_norm_scope`; exact |
| `--from-hf-config` dropped QK-norm (Qwen3-8B 9,216 short) | read from `model_type`; exact |
| `--verify --impl` passed an RMSNorm with eps added to the sqrt (NaN gradient on zero rows) | gradient check; detector covers both wrong placements |
| `--verify --impl` failed a reference-only property against the user's `repeat_kv` | skipped under `--impl` |
| GQA "~0.1 points" and "not a quality decision" | qualified with DeepSeek-V2's from-scratch ablation |
| WSD cooldown citation pointed at the wrong paper | 2405.18392 for the 20% result; 2508.01483 for cooldown shape |
| gpt-oss biases described as existing for the sinks | reworded per the model card |
| Optimizer consensus row out of date | Muon users listed; Muon section |
| `stability.md` cited an arXiv id missing from `sources.md` | registered; a structure test now fails on any such gap |

### Added

- `config_budget.py`: MoE, MLA, sinks, sliding-window KV, sandwich norms, active parameters under three conventions; 7 new presets; self-test checks 13 exact totals and 4 published active counts.
- Scripts: `modern_blocks.py`, `post_training_losses.py`, `spike_monitor.py`, each with tests.
- References: `moe-and-attention-variants.md`; RL section in `post-training.md`; Muon, gating, stress testing and new disagreements in `stability.md`; MoE rows in `reference-configs.md`; 16 paper digests in `references/papers/`; 27 new rows in `sources.md` with how each was checked.
- `examples/` (six worked answers, `buggy_port.py`), `assets/` (real configs, training log, two templates), `evals/evals.json` (16 cases).

### Changed

- SKILL.md: description covers MoE, MLA, Muon and GRPO-style RL; routing table; "Dense or MoE first" step; QK-norm presented as one bound among several; RL bullet; self-test excerpt from the current run.
- Verified against torch 2.15.0 and transformers 5.22.1 (was torch 2.13.0 / 2.14.1).

### Verification at release

| Check | Result |
|---|---|
| `python -m pytest tests` | 181 passed (was 38) |
| `config_budget.py --self-test` | 13 of 13 totals exact; 4 of 4 active claims matched |
| `reference_block.py --verify` | all checks passed; `--impl examples/buggy_port.py` fails exactly the 2 planted bugs |
| `modern_blocks.py --verify` | 34 of 34 checks passed (including 4 bug detectors) |
| `post_training_losses.py --verify` | 20 of 20 checks passed |
| Independent fact-check of new material | a separate agent re-read the cited papers and reran every script; 4 errors and 11 imprecisions found and fixed before release (Moonlight Table 6 pair, an activation figure attributed to the wrong model, gpt-oss window ratio, a figure number; gate variants, Kimi ablation scale, DeepSeek-V3 group routing and scaling factor, PPO library normalization, mixed active-count conventions, others); about 40 other claims confirmed |

## 1.1.0

Dense decoder-only sizing, reference block with property tests, residual probe, components, stability, post-training (SFT and DPO), reference configs, sources. Verified against torch 2.13.0 and 2.14.1.
