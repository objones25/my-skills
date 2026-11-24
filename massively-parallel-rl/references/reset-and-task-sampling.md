# Reset distributions and task samplers for many environments

With thousands of envs the reset rule decides what the batch contains. This file covers
what to sample at reset (a task or level, or a start state), the documented recipes with
their conditions, and the checks that tell you the sampler is the problem. Papers were
read from their text; each claim carries its source and setting. Concepts (access
protocols, start-state families, bandit teachers, evaluation) live in robot-learning-expert
`references/`.

## Contents
1. Two different things called a reset curriculum
2. Why uniform sampling wastes the batch
3. State-level reset banks at scale
4. Task and level samplers at scale (SGS, UED, learnability, teachers)
5. Reset hygiene and checks
6. What no paper here tests

## 1. Two different things

- **Task or level sampling.** Choose which task, terrain or level each env gets. Success
  Guided Sampling, PLR, SFL and teacher signals are this (`large-batch-on-policy.md`
  section 3 for SGS).
- **State-level reset banks.** Choose the start state inside one task (demo frames, visited
  states, validated grasp or contact states). OmniReset, X-Reset and CL-Buffer are this.

They compose, but evidence for one is not evidence for the other. Reset-free training is
a third, unrelated regime: simulated envs reset for free, so reset-free methods matter for
hardware, not here.

## 2. Why uniform sampling wastes the batch

Batch dilution (Minecraft curriculum paper, 2106.14876): under uniform sampling over N_tasks
tasks a given task gets 1/N_tasks of the batch, so getting H samples of it per batch needs
N_tasks x H samples, and more when some tasks are unlearnable and add only noise. That is the
quantitative reason sampling matters more as the batch grows. The paper is a single-agent
Minecraft setting, not a GPU-simulator one.

Wide start-state or level distributions can also break plain PPO outright: with sparse binary
reward, AlphaNIST (2610.13354, assembly, 1-256K envs) reports success rising from 38.6% at 1K
envs to 78.9% at 4K and 86.9% at 8K, and 97.4% at 256K with its beta reset sampling against
84.3% with uniform sampling; with dense shaping the same task stayed at 11.8-12.0% over 1K-4K
envs. Dex-One2Many (2610.12470) reports over 90% success with 49,152 envs against below 25%
with 24,576 or 12,288 in the same number of PPO updates under its diverse reset distribution.
Both runs hold updates, not samples, constant, so more envs also means more samples; read them
as "diverse resets plus large N", not as per-sample efficiency [inference].

## 3. State-level reset banks at scale

| Method | Reset rule | Scale and conditions | Reported result |
| --- | --- | --- | --- |
| OmniReset (2603.15789, ICLR 2026) | Offline: sample start states from four families, reject invalid ones by collision check plus a short simulator settling rollout, cache the four validated sets on the GPU, sample uniformly from their union | Dexterous manipulation, PPO, fixed hyperparameters, no curricula; Fig. 8 ablates 2,048 to 65,536 envs | Small N makes progress from near-goal states, but the full multi-stage task needs many envs; reward weights "largely insensitive" (App. A.1). The paper attributes saturation of plain RL to revisiting "the same narrow regions of state space" |
| X-Reset (2609.35715) | With probability p sample a retargeted human-demo hand-object state, else the default reset | Object-centric manipulation, PPO with 8,192 envs, minibatch 32,768, adaptive LR with KL target 0.01, rollout 32 (appendix table); default p = 0.9 | Ablation on UR7e+Sharpa: return 55.45-61.09 across p, against 8.28 with p = 0 (§5.3). Not specific to p = 0.9 |
| CL-Buffer (2309.09752, ICRA 2024) | Per rollout phase add N = 256 states, cluster the buffer with K = 64 in observation space, start episodes from the buffer with p = 0.8 | Quadruped locomotion, 1024 envs, 2000 iterations | Table I, validation reward over 5 seeds: overall 10.90 +- 0.15 for CL-Buffer vs 9.21 +- 0.43 vanilla, 9.58 random buffer, 10.40 observation-space buffer. The buffer excludes states under 15 steps from the episode start and states with negative accumulated reward |
| AlphaNIST (2610.13354) | Cached reset bank, success-weighted entry sampling with beta weights, per-GPU sampling from rolling success estimates | 4K envs per GPU, 32 steps per env, 5 epochs x 4 minibatches (scaling run 1-256K) | See section 2 |

Choosing: OmniReset-style offline banks need no demos but need per-task generators; X-Reset
needs demonstrations plus retargeting; CL-Buffer needs only the visited states. None of the
four compares itself with the others, and the settings are single-task manipulation or
locomotion benchmarks [inference].

## 4. Task and level samplers at scale

**Run the controls first.** Gilewicz et al. (2610.40364) compared fourteen curricula on one
PPO student across Go1, Go2, Barkour, W01-TEK and Spot terrain levels (MJX Playground robots).
No method was in the top group on every robot and level set. Methods with easy or
low-difficulty training levels, curated or not, filled the top groups on in-distribution
random levels for Go1, Barkour and (except PAIRED+BiBC) W01-TEK, and a success-driven terrain
curriculum of the Rudin et al. type was in a top group most often (13 of 20). Buffer methods
spent a fifth to a quarter of their steps on scoring iterations that do not update the
student. So before adopting PLR or ACCEL, run easy-start domain randomization and a
success-driven row curriculum on the same student. This qualifies, not contradicts, SGS: it
shows that the best sampler depends on robot and level set.

**Regret scores track success rate.** Rutherford et al. (2408.15099) report that the usual
UED scores (PVL, MaxMC) correlate with success rate rather than regret, and MaxMC shows no
correlation with learnability; PVL has high variance and keeps prioritizing solved maps.
Do not describe PLR or ACCEL as regret-maximizing without that qualifier. If reward is binary,
their SFL method scores levels by learnability p(1-p) (2408.15099); SGS and the Craftax teacher
papers use moderate-success bands for the same reason.

**Replay on Craftax.** PACE (2605.01358, Table 5, 20 Craftax evaluation levels, rliable IQM
of episodic reward): DR 0.603, PLR 0.484, PLR-perp 0.409, ACCEL 0.364, PACE 0.750. The
learned-score replay methods lost to plain DR there.

**Recipes with conditions.**
- PLR-perp in the quadruped benchmark (2610.40364): buffer of 4000 levels, replay with
  probability 0.8 once 256 levels are stored, 256 levels per iteration with 32 envs on each
  (256 x 32 = 8,192 envs), rank temperature 0.3, staleness weight 0.3.
- CL4AD (2608.22549, autonomous driving, scenario sets of 1K to 80K, PLR): buffer as large as
  the dataset; score temperature 2 or 4 instead of the usual 0.1-1.0, because a low temperature
  concentrates sampling on a few levels as the buffer grows; staleness coefficient 0.1-0.3 as in
  earlier work; among regret scores MaxMC was the most reliable across scales because it does
  not depend on value accuracy.
- Do not scale these numbers with N. The largest env count among the sampler papers read for
  this section is 8,192 (the quadruped benchmark) [inference from the papers read].

**Teacher signals and their blind spots** (small-scale evidence):
- Success band p(1-p): blind to tasks that are far from solved but flat, and to forgetting.
- Learning progress (slope of recent performance): in DRATS (2605.14350, 20 seeds on MuJoCo6
  and MetaWorld-MT10) it learned Humanoid more slowly than uniform sampling because its small
  slope deprioritized a task that was far from solved (Uniform and Learning Progress gave it at
  most 17% of samples), and it did not prioritize the hard push and pick-place tasks in MT10
  until late. Add a floor: DRATS enforces a minimum sampling probability of 1/(4k) for k tasks in every
  baseline to prevent forgetting.
- Per-task advantage normalization: DRATS normalizes advantages per task for all methods
  because tasks with larger mean return otherwise make other tasks' advantages look negative
  (their Fig. 13, in the appendix, not read in the text extraction).
- Mapping a teacher onto envs (2610.04604, Craftax, one setup): sample 64 goals with
  replacement per curriculum decision, assign each to 8 parallel envs (512 total), and fix
  four of the 64 slots to the target task. Reserve slots for the target; do not let a teacher
  drive it to zero.

## 5. Reset hygiene and checks

- Filter reset states offline before caching them: collision check plus a short settling
  rollout (OmniReset); X-Reset rejects states that terminate, give non-finite values or break
  speed gates, and reports reset rejection of 6.82% with clean poses and 16.71% with its
  high-noise condition (2609.35715 §5.4).
- A buffer of visited states fills with failure and transition states unless filtered
  (CL-Buffer's two exclusions above).
- Log success per reset family or per level bucket, not only from the default start, so you
  see which stage is stuck (OmniReset evaluates from each family).
- Level samplers add cost: PLR-style scoring iterations took a fifth to a quarter of steps in
  2610.40364; heterogeneous task code inside one vmapped JAX loop recompiles (`jax-stack.md`
  section 3).
- Uniform sampling over levels of mixed difficulty is not always the control to beat; add an
  easy-start control (section 4).

## 6. What no paper here tests

- Whether any level sampler's hyperparameters carry from 8K to 1M envs.
- Teachers at N of 1,024 envs or more beyond the 512-env Craftax case, in the papers read.
- Interaction between reset banks and the minibatch recipe in `large-batch-on-policy.md`.
- Combining a level sampler with SGS-style moderate-success weighting and an easy-start
  control in one run.
