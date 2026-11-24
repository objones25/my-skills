# Reset-free, goal-conditioned and unsupervised RL

Use this when the question is "what if I cannot reset the robot between episodes" or "I have no reward, only goals". Everything here is simulation evidence unless stated; for access-protocol theory see `resets-and-exploration.md`. Findings were read from the papers' text; method details beyond the lines cited were not re-read (unverified).

## Reset-free (autonomous) RL

- Leave No Trace (arXiv 1711.06782): train a forward policy and a reset policy; abort the forward episode early when the reset policy's Q-value (an ensemble; "optimistic aborts" abort only if all members are below Qmin, "realist" if the mean is) says the reset is unlikely to succeed; fall back to a hard (manual) reset only when needed. Optimistic aborts explore more at the cost of more hard resets [LNT lines 243-255].
- VaPRL (2107.12931) builds a start-state curriculum for the forward task. RISC (2405.01684) switches between forward and reset controllers on confidence; because each policy's value depends on the other's actions at a switch, it bootstraps through the switch (timeout-nonterminal loss) instead of treating it as terminal, and its ablation shows that matters [RISC line 258; ablation lines 621, from the findings report]. MoReFree (2408.09807) is the model-based variant (not re-read).
- Condition they share: reversible environments. VaPRL says its formulation "is suitable only for reversible environments" [2107.12931 line 200]; RISC lists irreversible states as a limitation [2405.01684 lines 638-642]; LNT's abstract frames non-reversible states as the thing to detect [1711.06782 lines 26-35].
- Reversal-Bench (2609.17745, Sep 2026, simulation, 8 manipulation settings in 5 engines): sweeping a reversibility parameter ρ from 0 to 1 shows a "reversibility cliff": reset-free agents, including safe-RL and dedicated reset-free methods, get absorbed into irrecoverable states as ρ rises while episodic agents keep learning [abstract; Fig. 2 caption line 228].
- Practical rule [inference]: reset-free methods when the task is reversible and manual resets are the cost; if failures can be irreversible (object off the table, spill), do not rely on a learned reset, add a recoverability monitor and budget for human resets.
- Truncation bootstrapping also governs controller switches (see `q-learning.md` on terminated vs truncated).

## Goal-conditioned and unsupervised RL

Use goal-conditioned or skill-discovery methods when there is no reward, the task is naturally goal reaching, you hold large unlabeled data, or you want a reusable skill prior. For a single reward-defined task with a fast simulator, reward-based PPO at scale stays the default [inference: the papers compare against other unsupervised methods, not against tuned PPO].
- Evidence on real robots is thin: Skew-Fit learned a real door-opening skill to a 95% success rate after about five and a half hours, image-based [1903.03698 lines 565-572]. TLDR was not tested on real robots and lists safety as an open challenge; it is wall-clock efficient but not sample efficient [2407.08464 lines 500-503].
- Assumptions to check: QRL's estimator is unbiased only for deterministic dynamics [2304.01203 line 178]; HIQL likewise assumes deterministic dynamics [2307.11949 line 626]; CRL looks only at goal-conditioned problems [2206.07568 line 686]; METRA's symmetric-distance assumption fails in asymmetric environments (from the findings report, not re-read).
