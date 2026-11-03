---
name: robot-learning-expert
description: Robot learning and RL training expert. Use to explain, derive or check an RL concept (MDPs, dynamic programming, LQR and trajectory optimization, bandits, model-based RL, Q-learning, policy gradient and PPO, actor-critic, imitation, exploration with simulator resets, RL theory); to design, train or debug robot policies (massively parallel PPO on mjlab, rsl_rl or Isaac Lab, off-policy at scale, rewards and observations, sim-to-real, PPO health metrics, seed statistics); or to read and compare robot learning papers.
---

# Robot learning expert

Concepts, training practice, paper digests and tested tools for reinforcement learning on robots: from the MDP and Bellman operators through PPO with thousands of GPU environments, sim-to-real transfer, and how to tell whether a training result is real.

## Rules

1. Library facts come from the installed source or current docs, never memory. mjlab, rsl_rl, MuJoCo Warp, Isaac Lab, Gymnasium and W&B change often. When the user has an environment, read the installed package (its `site-packages/<name>/`); otherwise start from `references/training/simulators-and-tooling.md` (versions checked Nov 2, 2026) and confirm against the live docs. Facts tagged `[rsl_rl 5.4.2]` or `[mjlab 1.6.0]` were read from installed source.
2. Every number has a source: a book or notes section, a paper location (`[PPO §3]`), a file, or a script run. Run code to check numbers before stating them, and prefer a small measured experiment (`examples/`) over a remembered rule of thumb.
3. When sources disagree (textbook, paper, lecture notes, library code), say so, say which one you follow and why. Citation keys and known errata are in `references/sources.md`; papers' own errors are in each digest and in `references/papers/canon-algorithms.md` ("Disagreements and errors found").

## Source tags

- `Notes N`, `AJKS`, `SB`, `BBC`: books and lecture notes, keys in `references/sources.md`.
- `[Paper §x]`: a located claim in a paper digest under `references/papers/`.
- `[rsl_rl 5.4.2]`, `[mjlab 1.6.0]`: read from installed source.
- `[practice]`: common practice, with the source named where one exists. `[inference]`: synthesis not stated in a source. `(unverified)`: not checked against an opened page.

## Route to the right reference

Read only the files the task needs.

| Task | Read |
|---|---|
| Symbols, array shapes for tabular and rollout code | `references/notation.md` |
| Books, lecture notes, citation keys, errata | `references/sources.md` |
| MDPs, Bellman operators, VI / PI, finite horizon, hand-checkable MDPs | `references/concepts/mdp-dp.md` |
| LQR, Riccati, trajectory optimization, MPC by sampling | `references/concepts/continuous-trajopt.md` |
| Bandits: UCB, Thompson, linear bandits, concentration | `references/concepts/bandits.md` |
| Generative model, PAC bounds, model-based RL, CEM / MPPI | `references/concepts/model-based.md` |
| Q-learning: tabular, UCB-Hoeffding, linear MDP, LSVI-UCB, FQI, DQN, exploration bonuses | `references/concepts/q-learning.md` |
| Policy gradient, GAE, TRPO, PPO and how rsl_rl implements it, PG theory | `references/concepts/policy-gradient.md` |
| DDPG, TD3, SAC, asymmetric actor-critic | `references/concepts/actor-critic.md` |
| Behavior cloning, DAgger, IRL / GAIL / AMP, diffusion policies, VLAs | `references/concepts/imitation.md` |
| Exploration with simulator resets: access protocols, RVFS, Go-Explore, reset distributions | `references/concepts/resets-and-exploration.md` |
| Massively parallel PPO: batch arithmetic, legged_gym recipe, G1 case study, fair wall-clock comparison | `references/training/on-policy-at-scale.md` |
| Off-policy at scale: FastTD3, PQN, FlashSAC, when to pick it | `references/training/off-policy-at-scale.md` |
| Rewards, observations, actions (PD targets), terminations vs truncation; G1 task as template | `references/training/reward-and-observation-design.md` |
| Domain randomization, teacher-student, RMA, actuator models, deployment | `references/training/sim-to-real.md` |
| Simulators and libraries (mjlab, MuJoCo Warp/Playground, Isaac Lab, rsl_rl, Brax, ManiSkill, Genesis, Gymnasium): versions, when to pick | `references/training/simulators-and-tooling.md` |
| Tests for RL code, PPO health metrics and fixes, training-system bugs | `references/training/diagnostics.md` |
| Seeds, IQM, bootstrap intervals, fixed evaluators, robot metrics, write-ups | `references/training/evaluation.md` |
| A specific paper (13 digests, a theory map, a massively parallel RL survey index, two verified canon lists) | `references/papers/README.md` |

## Scripts (tested oracles and tools)

Each runs standalone (`python scripts/<name>.py` runs a self-check or prints a table). Tests: `python -m pytest tests` from the skill root. Dependencies: `scripts/requirements.txt` (numpy; scipy, gymnasium and torch optional).

| Script | Does |
|---|---|
| `scripts/mdp_tools.py` | exact tabular ground truth: closed-form policy evaluation, Q-VI with certified error, PI, finite-horizon DP, occupancy, random MDPs, combination lock |
| `scripts/lqr.py` | finite-horizon Riccati, DARE, rollout cost, least-squares system ID, stability |
| `scripts/local_sim.py` | online / local-simulator / generative access protocols with query counting, checkpoints, observation-noise models, Gymnasium wrapper |
| `scripts/rl_math.py` | GAE and returns with terminated vs truncated, rsl_rl's time-out variant, KL estimators k1/k2/k3, clip fraction, explained variance, PPO clip mask, tanh-Gaussian log-prob, truncation-bias calculator |
| `scripts/batch_math.py` | batch, minibatch and gradient-step arithmetic for N envs, and run cost from iterations/minute and $/h |
| `scripts/run_stats.py` | IQM, stratified bootstrap CIs, probability of improvement, curve bands |

To check someone's implementation, run it and the oracle on the same inputs and report where they disagree.

## Examples and assets

| File | Shows |
|---|---|
| `examples/ppo_pendulum.py` | 200-line PPO with correct time-out bootstrapping and logged KL, clip fraction, explained variance; solves Pendulum-v1 (deterministic eval near −130) in about 75 s of CPU. `--timeout-as-terminal` switches the bug on (measured effect: small and noisy at γ = 0.99, see its docstring) |
| `examples/analyze_sweep.py` | a real G1 num_envs sweep (`assets/g1_sweep_results.csv`): seed means, the standing-still reference, pairwise P(improvement) |
| `examples/exploration_chain.py` | exact A^H episodes for random exploration vs about H(A+1)/2 steps with resets on the combination lock |
| `assets/experiment_plan.md`, `assets/writeup_skeleton.tex` | templates for planning a robot RL experiment and writing it up |

## Decision guides

Which method: ask about the interaction protocol first (Notes 0).
```
known model, small finite S, A ─────────► exact DP (VI / PI / backward induction)
known model, linear dynamics + quadratic ─► LQR (Riccati / DARE)
known model, continuous, deterministic ───► trajectory optimization; replan (MPC)
known or sampled model, stochastic ───────► sampling MPC (predictive sampling, CEM, MPPI)
generative model ─────────────────────────► estimate P̂ and plan, or Q-learning on samples
reset to visited states (local simulator) ─► RVFS / Go-Explore style (concepts/resets-and-exploration.md)
online, tabular or linear features ───────► optimism: UCB-Hoeffding Q-learning, LSVI-UCB
online, large/continuous, cheap GPU sim ──► PPO with thousands of envs; or FastTD3/SAC at scale
online, expensive samples or hardware ────► off-policy (SAC/TD3/RLPD), model-based
expert demonstrations available ──────────► behavior cloning; DAgger if the expert can be queried
deploying on a real robot ────────────────► PD-target actions, domain randomization, asymmetric critic or teacher-student
```

Training is not learning: in order, check the MDP (rewards per term, terminations vs truncations, observations), then the batch arithmetic, then the PPO health metrics, then seeds and the evaluator. Details: `references/training/diagnostics.md`.

## Explaining

Idea first in two or three sentences; notation defined in one place (each symbol and, for arrays, its shape); the mechanism as 2 to 6 named steps; then break down only the hard steps. A small worked example only when it makes the key point concrete, with numbers checkable by hand. Give section or equation numbers so the reader can find the source. Do not assume the reader knows a mechanism because they have used it.

## Extend this skill

1. A new paper: digest it in `references/papers/` with located claims and a "gotchas" section, add a row to `references/papers/README.md`, and link it from the concept or training file it changes.
2. A new topic or a better source for an existing one: update the concept file and record any source errors in `references/sources.md`.
3. A script only for an oracle or tool worth running more than once, with a test (`tests/test_skill_structure.py` fails if a script has no test file).
4. Log the change in `references/CHANGELOG.md`, and add an eval to `evals/evals.json` when a new kind of question shows up.
Keep entries terse: the load-bearing point first, then the mechanism, then the source.
