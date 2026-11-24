# Changelog

## 2026-11-24, v1.1.0: gymnax 1.x, paper corpus, review fixes

Broken behaviour
- gymnax 1.0 (now the version uv resolves by default, with jax 0.11.2) reports the time limit through `is_truncated()` instead of OR-ing it into `step_env`'s `done`, and `step()` returns a 6-tuple. `mprl_common.AutoReset` never truncated or reset Pendulum episodes, and `throughput_bench.py` failed to unpack the step. The existing tests `test_autoreset_separates_truncation_from_termination` and `test_throughput_bench_runs` failed on gymnax 1.0.1 and pass after the fix; the suite passes on both versions (below).
- SKILL.md now has a `metadata.test-command` with pinned versions, since an unpinned install resolves to a different gymnax.

Wrong claims fixed
- EPO vs SAPG (`large-batch-on-policy.md`, `papers.md`): "35.8 vs SAPG 2.2" was wrong; EPO's own Table 1 gives SAPG 4.4 and 3.9 on M:TR (2.2 is PBT on M:ST). The 20% vs 38% seed-variance figure is not in the paper and was removed. The step count behind SAPG's published 28.58 differs between the two papers (5e10 vs 2e10), stated in place.
- Non-negotiable 1 listed gymnax among libraries that OR the time limit into `done`; that holds for gymnax < 1.0 only.
- "gymnax MinAtar has 1-2 start states" (SKILL.md, `scaling-laws.md`, `jax-stack.md`, the `AutoReset` docstring): Freeway's start state is random (2,000 distinct resets of 2,000 keys on gymnax 0.0.9 and 1.0.1); the claim now names Breakout, SpaceInvaders and Asterix.
- Qualifiers added from the citation audit: ABS adjusts rollout length (its PPO run uses 1 env, Atari uses 3 seeds), SGS results are at 1M envs, PQN's "up to 50x", FlashSAC's network size (12.5x the ~0.2M it attributes to FastTD3) and PPO's larger step budget.

Added (from a 206-paper corpus; every number re-grepped against the paper text, then independently reviewed)
- `references/reset-and-task-sampling.md`: reset banks and demo-state resets at scale, reset hygiene, UED controls and replay settings, task-sampling signals and their failure modes, per-task env allocation.
- `large-batch-on-policy.md`: staggered resets, CPO, the SONIC config, the scope of the fixed-minibatch recipe (dense-reward evidence) and the Rudin / Beukman / SGS disagreement on minibatch size, Mayor et al.
- `debugging.md`: plasticity and representation collapse, periodic return spikes from synchronous resets, GPU nondeterminism (GPUSimBench). `pytorch-gpu-sim.md`: RSL-RL-SAC, determinism. `jax-stack.md`: gymnax version split, recompilation cost of heterogeneous lanes. 20 new `papers.md` rows.

Review: an independent reviewer checked the diff (about 55 claim groups confirmed; 1 wrong, 2 miscited, 3 overstated, 3 missing a qualifier, 1 unsupported); all fixed except one PACE item whose cited numbers could not be found in the paper text, left as it was.

Not done: an optional `--stagger` warm-up in the scaffolds; a measured throughput table; evals and trigger evals (the skill has none).

Tests:
| Date and machine | Versions | Result |
|---|---|---|
| 2026-11-24, macOS arm64, Python 3.13.12 | jax 0.11.2, gymnax 1.0.1, flax 0.12.12, optax 0.2.8, distrax 0.1.9 | 25 passed (2 failed before the fix) |
| 2026-11-24, macOS arm64, Python 3.13.12 | jax 0.12.0.1, gymnax 0.0.9, same others | 25 passed |

## 2026-11-19, v1.0.1 and earlier
See git history (`git log -- massively-parallel-rl`). Earlier test record: 2026-11-03, macOS arm64, JAX 0.12.0.1, gymnax 0.0.9.
