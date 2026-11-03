# Evaluating and comparing RL runs

A learning curve from one seed is an anecdote: deep RL results vary a lot across seeds, and point estimates from 3 to 10 runs mislead (Henderson et al. 2018; Agarwal et al. 2021, both in `papers/canon-algorithms.md`). Fair comparison needs a fixed evaluator, the right x-axis, and interval estimates.

## Protocol

1. Decide the x-axis by what is scarce: env steps (sample efficiency), wall-clock on fixed hardware (practical speed, the G1 sweep's choice), or gradient steps. Say which, and time only what the method controls (`on-policy-at-scale.md` fairness rules).
2. Evaluate with a fixed protocol separate from training: same commands or tasks, seeds, horizon, number of eval envs, no training noise or pushes, deterministic (mean) actions unless the stochastic policy is the deliverable. Training reward includes exploration noise and curriculum stage, so it is not comparable across runs.
3. Seeds: at least 3 for a demo, 5 to 10 for a claim. With 3 seeds the IQM (25% trimmed on each side, as `run_stats.iqm` computes it) keeps all 3 runs and equals the mean, so show mean with min-max range instead (the G1 sweep's choice).
4. Aggregate across seeds (and tasks) with the interquartile mean (IQM, mean of the middle 50% of runs) and stratified bootstrap confidence intervals; for "is A better than B", report probability of improvement P(X_A > X_B). `scripts/run_stats.py` implements all three.
5. Report the batch arithmetic, compute, and every deviation from a default config.

## Writing it up

Negative results are worth reporting with what you tried and why it failed. For a scaling sweep, report: batch arithmetic per run, the clock definition, curriculum and overhead disclosures, the evaluator definition, seeds and interval method, cost arithmetic, and an explanation of flat results (in the G1 case study, 16384 vs 4096 envs). Show the evaluator's definition next to the plot, and say which number is primary before looking at results.

## Robot-specific metrics `[practice]`

Locomotion: velocity tracking error (xy and yaw separately), fall-free fraction, time to fall, energy or cost of transport, foot slip. Manipulation: success rate over N trials with a binomial interval, time to success. Sim-to-real: the same metrics on hardware, with the number of trials, and sim numbers under the same randomization the policy trained on and outside it.
