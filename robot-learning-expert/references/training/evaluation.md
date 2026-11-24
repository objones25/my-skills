# Evaluating and comparing RL runs

A learning curve from one seed is an anecdote: deep RL results vary a lot across seeds, and point estimates from 3 to 10 runs mislead (Henderson et al. 2018; Agarwal et al. 2021, both in `papers/canon-algorithms.md`). Fair comparison needs a fixed evaluator, the right x-axis, and interval estimates.

## Protocol

1. Decide the x-axis by what is scarce: env steps (sample efficiency), wall-clock on fixed hardware (practical speed, the G1 sweep's choice), or gradient steps. Say which, and time only what the method controls (`on-policy-at-scale.md` fairness rules).
2. Evaluate with a fixed protocol separate from training: same commands or tasks, seeds, horizon, number of eval envs, no training noise or pushes, deterministic (mean) actions unless the stochastic policy is the deliverable. Training reward includes exploration noise and curriculum stage, so it is not comparable across runs.
3. Seeds: at least 3 for a demo, 5 to 10 for a claim, with these qualifiers. Agarwal et al.'s stratified-bootstrap validation (percentile CIs good from N = 10 runs) is for aggregates across many tasks [2108.13264 line 437]. A single-task claim with long-tailed outcomes can need about 20 runs for the bootstrap CI and 30 for the mean [Patterson 2304.01315 lines 729-731, one DQN example]. Averaging over 3 or 5 runs is risky unless agent and environment are deterministic [Patterson line 2237]. With 3 seeds the IQM (25% trimmed on each side, as `run_stats.iqm` computes it) keeps all 3 runs and equals the mean, so show mean with min-max range instead (the G1 sweep's choice).
4. Aggregate across seeds (and tasks) with the interquartile mean (IQM, mean of the middle 50% of runs) and stratified bootstrap confidence intervals; for "is A better than B", report probability of improvement P(X_A > X_B). `scripts/run_stats.py` implements all three.
5. Report the batch arithmetic, compute, and every deviation from a default config.

Further rules from Patterson et al. (2304.01315) and a UED benchmark:
- IQM is "most usefully applied across collections of environments" and can hide catastrophic seeds [Patterson lines 753-758]: show min and max or per-seed outcomes next to it.
- Paired comparisons: share weight initialization, initial states and replay indices between the two conditions to cut variance [Patterson lines 2257-2262]. Do not drop or reseed diverged runs; find out why [lines 2265-2268]. Give baselines real tuning effort; a small baseline advantage is better than over-claiming [lines 1363-1366].
- Report held-out structured terrain next to in-distribution random levels; in a five-quadruped UED benchmark (MJX, blind actor, 5 seeds) the top method differed between them on three robots [2610.40364 lines 56-76, 277].
- For worst-quantile claims use cross-fitted CVaR: rank levels on one pass and score them on another, because ranking and scoring on the same episodes biases CVaR low (2000 levels of true success 0.5 and 4 episodes each gave same-pass CVaR25 0.19) [2610.40364 lines 249-253].
- Average success hides zero-success tasks: also report the fraction of tasks or variants with at least one success and the worst variant's success [practice; from the findings report, 2609.21650 not re-read (unverified)].

## Writing it up

Negative results are worth reporting with what you tried and why it failed. For a scaling sweep, report: batch arithmetic per run, the clock definition, curriculum and overhead disclosures, the evaluator definition, seeds and interval method, cost arithmetic, and an explanation of flat results (in the G1 case study, 16384 vs 4096 envs). Show the evaluator's definition next to the plot, and say which number is primary before looking at results.

## Robot-specific metrics `[practice]`

Locomotion: velocity tracking error (xy and yaw separately), fall-free fraction, time to fall, energy or cost of transport, foot slip. Manipulation: success rate over N trials with a binomial interval, time to success. Sim-to-real: the same metrics on hardware, with the number of trials, and sim numbers under the same randomization the policy trained on and outside it.
