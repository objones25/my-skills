# On-policy RL with very many environments

What happens to PPO and A2C as the number of parallel environments N grows from
thousands to a million, and what to do about it. Every claim cites the paper it comes
from; the papers were read in full or in their arXiv HTML.

## Contents
1. The plateau and its two explanations
2. Recipe: scaling N without stalling
3. Fixes that change what the batch contains
4. Choosing among them

Reset and task samplers (success-weighted levels, reset banks, UED, teachers) have their own
file: `reset-and-task-sampling.md`.

## 1. The plateau

Plain PPO stops improving well below the optimum once N is large, and adding
environments past that point does not help. Two diagnoses exist and they suggest
different fixes.

**Duplicate data** ([SAPG, Singla et al., ICML 2024](https://arxiv.org/abs/2407.20230)).
A Gaussian policy samples actions near its mean, so tens of thousands of environments
largely repeat each other. A bigger batch adds samples but little information.
Evidence: PPO barely learns the hard AllegroKuka manipulation tasks at 24,576 envs.

**Step size too large for the noise** ([Beukman et al., RLJ 2026](https://arxiv.org/abs/2603.06009);
all four Zheng et al. authors are co-authors). Treat PPO's outer loop (collect,
update, repeat) as noisy stochastic optimization. If each policy change is large
relative to the noise in its estimate, the policy bounces around an optimum instead
of settling. The plateau is under-regularization, not a lack of exploration or
capacity. Two levers: shrink the outer step (stronger trust region, fewer epochs) or
collect more data per update (more envs). Tuning the learning rate cannot fix a bad
outer step: the two act through different mechanisms.

The diagnoses are compatible: duplicates make the effective sample count lower than
N x T, which is the noise side of the second account.

## 2. Recipe: scaling N without stalling

From Beukman et al. 2026, who ran over 1M environments on 128 GPUs:

1. Hold minibatch size and learning rate fixed. Increase the number of minibatches
   per update (num_minibatches = N x T / minibatch_size), so more environments mean
   more optimizer steps on lower-noise data.
2. Increase minibatch size, with a scaled learning rate, only when hardware
   utilization is the bottleneck. Larger minibatches sometimes destabilized training
   and lowered the plateau.
3. As the total interaction budget grows, the best outer step size shrinks, so the
   ratio of data collected to policy change allowed per update should rise.
4. Weaker regularization plateaus sooner as epochs increase; stronger regularization
   partly offsets it.

Results: standard PPO configs plateau before about 10B interactions; scaled past 1M
envs, PPO improved monotonically to 1 trillion steps. On Isaac Gym tasks, reverting
the minibatch from 96k to 16k improved PPO everywhere and narrowed its gap to SAPG.
For Kinetix they used a compromise for wall-clock speed: 32x the minibatches, each 16x
larger, with a 4x larger learning rate. For the small Kinetix distribution, 65k envs
was enough.

Scope of this recipe. Beukman et al. state that "our analysis focuses solely on dense reward
tasks" and that sparse-reward and exploration-heavy tasks need further work (2603.06009,
limitations). Their unifying quantity is the Data to Divergence Ratio (DDR), data points per
unit of KL between successive policies: low DDR plateaus early, high DDR learns slowly, and the
DDR should rise as the training budget grows (2603.06009 §4.4, Fig. 8). They use PPO-EWMA
(the proximal policy is an exponential moving average of the weights) as an analysis tool
that decouples the regularization target from the behaviour policy. Their Isaac Gym evidence
is one change to SAPG's code, reverting the minibatch from 96k to 16k (Fig. 10), plus
preliminary results in their Appendix G; the main evidence is Kinetix.

Other sources disagree on the minibatch. Rudin et al. (2109.11978 §2.2) report that
"mini-batches of tens of thousands of samples" stabilized learning without raising training
time. Success Guided Sampling holds the number of minibatches constant, so the minibatch
grows with N (2610.12465 §4 scaling experiment). None of these is a matched-compute comparison against the
Beukman recipe; measure with the three-run triage in `debugging.md` instead of picking one.

On the many-envs-versus-longer-rollouts axis, Mayor et al. (2506.03404, ICML 2025) hold the
data budget fixed on Atari-10: raising `N_envs` while shortening the rollout (for example 128
envs x 8 steps instead of 8 x 128) gave stabler training than the reverse, with lower weight
norm, fewer dormant neurons and lower gradient kurtosis. Scaling the learning rate up or down by
the same factor as N_envs (only a 2x change was tested) did not beat the default for PPO (their
Fig. 11, PPO "robust to learning rate changes"). Conditions: Atari-10, PPO and PQN, N_envs of order 10^2, 5 seeds; the PQN
gain is marginal (their Fig. 6 caption). It is not evidence about thousands of envs.

Tension with the scaling-law paper: Zheng et al. 2026 grow the minibatch with compute
(Craftax "batch size" is observations per minibatch). Under the Beukman recipe the
same budget would instead add minibatches. Nobody has compared the two at matched
compute; when a user's batch-size law is driving a minibatch size up, mention that
the alternative exists.

`scripts/ppo_jax.py` implements the recipe: it takes `--minibatch-size` and derives
the number of minibatches from N x T.

## 3. Fixes that change what the batch contains

| Method | Where it acts | Mechanism | Reported gain |
| --- | --- | --- | --- |
| [SAPG](https://arxiv.org/abs/2407.20230) (ICML 2024) | who collects data | Split N envs into M blocks, each with its own policy (shared backbone, small per-policy latent). A leader trains on its own data plus importance-weighted, subsampled follower data; followers get different entropy bonuses to stay diverse. 24,576 envs, M = 6. | Beats DexPBT by 12% to 66% on three AllegroKuka tasks after 2e10 samples; about 2x PBT on two-arm reorientation (28.6 vs 14.5 successes) |
| [EPO](https://arxiv.org/abs/2503.19037) (Wang, Su, Gupta, Pathak 2025) | who collects data | Population of agents sharing one actor-critic, each conditioned on a latent "gene"; only the genes evolve (elites kept, crossover by averaging, Gaussian mutation). Master agent aggregates population data SAPG-style. 64 agents x 384 envs. | Two-arm reorientation: 35.8 episode successes at 1e10 steps in the text (Table 1, row M:TR: 37.0) vs SAPG 4.4 (64 agents) and 3.9 (6 agents) in EPO's own Table 1. The text also compares with SAPG's published 28.58, which EPO says SAPG reached after 5e10 steps; SAPG's own Table 1 caption says 2e10 samples (the two papers disagree). EPO reports lower seed variance qualitatively; the paper gives no percentage. A third source (CPO, 2603.01741 Table 1, 5 seeds, 2e10 steps) also reproduces SAPG far below its own number on this task: 5.11 vs CPO 35.30 [inference: SAPG's 28.58 may not reproduce outside its own setup] |
| [Success Guided Sampling](https://arxiv.org/abs/2610.12465) (Zhang et al., Oct 2026) | which tasks start | Uniform resets waste a growing share of the batch on configurations already solved or not yet attemptable. Track per-configuration success over a recent window; sample moderate-success configurations more (beta-shaped weights), never zero. Up to 2^20 envs. Settings: window of the last 100 outcomes per configuration, T = 2, t = 0.66 and kappa = 5 for locomotion (N = 104,000), t = 0.5 and kappa = 1 for manipulation (N = 32,768); the paper reports low sensitivity to most of these (§3, App. B.2). Moderate-success weighting assumes the configuration is feasible; an infeasible one can sit at a moderate success rate and soak envs, so keep an explicit exploration share [inference]. | At 1M envs, Anymal-D 13-terrain multi-task locomotion: 73% vs 54% for Prioritized Level Replay; Franka Panda nut-and-bolt: 70% vs 6% uniform and 5% PLR. Means over 3 seeds per scale, PPO with the minibatch count held fixed (2610.12465 §4, App. B). Also helps at 4,096 envs, where uniform and other baselines fail to learn on many terrains; on the easy rod-in-hole task all methods reach about 98% at 256K |
| [Adaptive Batch Scaling](https://arxiv.org/abs/2605.21557) (Park 2026) | data per update | Grow the batch (via rollout length) as the policy stabilizes, measured by the share of up to 2,048 states whose greedy action changed since the last check. Forward passes only, unlike gradient-noise-scale methods. | On PQN (Atari-10, 3 seeds, PQN-L and PQN-XL), a fixed 16,384 batch hurt larger models while scheduling 2,048 to 16,384 helped on aggregate; per-game results are mixed (Table 2: BattleZone PQN-L 0.68 default vs 0.05 with ABS). PPO gains shown only as curves on MuJoCo with Num. Environments = 1 (Table 8), so this is not evidence for the many-environment regime |

### What the ABS code actually does (daisophila/ABS 660b0c1)

PyTorch on CleanRL: PQN + ABS on Atari via envpool, a PQN + gradient-noise-scale
baseline, and PPO + ABS on MuJoCo; no BTR code. The metric is called `policy_change`.
- PQN: every 50 iterations, the fraction of greedy actions that flip on up to 2,048 states
  resampled from the latest rollout (not a fixed reference set). That rate maps
  log-linearly onto rollout length 16 to 64 between thresholds 0.05 and 0.95, smoothed
  with a 0.9/0.1 moving average; more change gives a shorter rollout; epochs are rescaled
  with the rollout (algos/pqn_atari_envpool_ours.py:84-92, :392, :436-440).
- PPO: the same scheme with Gaussian KL, thresholds 0.01 and 0.1, rollout 1,024 to 8,192,
  every 10 iterations (algos/ppo_continuous_action_ours.py:83-91, :177, :219-234).
- As published, the PQN path builds its snapshot network with three arguments
  (`QNetwork(self.envs, self.args, self.device)`, line 415) while the class takes one
  (line 361), so it should fail at the first adaptation step. The repo has no license
  file, so treat the code as reference only.

### A built-in outer-step controller: rsl_rl's adaptive KL

rsl_rl, the PPO used by Isaac Lab and mjlab, already adjusts the learning rate every
minibatch to hold KL near a target: above 2 x 0.01 it divides lr by 1.5, below 0.005 it
multiplies by 1.5, within [1e-5, 1e-2] (rsl_rl ppo.py:240-266). It acts on the step-size
side of Beukman et al.'s account, though the paper notes the learning rate and the outer
step act through different mechanisms. Its default minibatch is large: 4096 envs x 24
steps / 4 minibatches = 24,576.

### Staggered resets: fix the batch's phase composition

Source: Bharthulwar and Tao, 2511.21011 (ManiSkill3 PPO, 10 seeds; also run with SAPG).
If all N envs reset together and the rollout length K is much shorter than the task horizon H,
each update sees the same slice of the horizon and the slice cycles with period H/K updates
("cyclical batch nonstationarity"). Staggering starts each env at a different offset in the
horizon. Recipe (§3.3, App. D.1): split the N envs into NB groups with offsets {0, S, 2S, ...},
S = K, NB about H/K; run the offset steps before training (with random actions or the
initial policy, App. B.1); afterwards reset each env at its own episode end as usual.
- Reported effect: naive PPO's wall-clock gain from more envs saturates beyond about 1,024 envs on
  StackCube-v1 and degrades sharply at 6,144 on Unitree G1 Transport Box (§5.4, Fig. 5); the
  staggered runs did not. For StackCube-v1 with K = 8 the best NB was about 10-13, near H/K.
- Exception stated by the authors: on locomotion (MS-HumanoidWalk) the two schemes were
  comparable, because falls already desynchronize the envs.
- Related shipped behaviour: rsl_rl randomizes episode lengths once at the start of training
  (`init_at_random_ep_len`, `pytorch-gpu-sim.md` section 2), which staggers the first episodes
  but not later resets. Isaac Lab and mjlab runs on fixed-horizon manipulation tasks may benefit
  from the same check (symptom in `debugging.md`) [inference].
- Not shipped in `scripts/` (follow-up: an optional `--stagger` flag).

### Diagnosing SAPG: effective sample size (CPO)

Shitanda et al., 2603.01741: SAPG's leader reuses follower data through importance weights,
and the weights can degenerate. At 5e9 steps on ShadowHand the SAPG leader's effective-sample-size
rate was 0.0223 and mean IS ratio deviation 0.889; CPO, which adds a KL constraint tying followers
to the leader, gave 0.941 and 0.187 for the CPO(0.05) setting (Table 2). CPO reached SAPG's final
performance with about half the environment steps on many tasks, with no significant gain
over SAPG on AllegroKuka Regrasping and Throw (results text). Cost: about 24% more wall-clock with the
KL term, 52% with the adversarial reward as well, at N = 24,576. If you run SAPG, log the leader's
ESS and IS-ratio deviation.

### A very-large-batch PPO config that is documented to work (SONIC)

SONIC (2511.07820, Isaac Lab humanoid motion tracking, Table S2): 4,096 envs per GPU on 128
GPUs, 24 steps per env, 5 epochs x 4 minibatches, gamma 0.99, GAE 0.95, clip 0.2, actor
learning rate 2e-5 (adaptive within [1e-5, 2e-4]) and critic 1e-3, max gradient norm 0.1,
desired KL 0.01, initial action std 0.05. The authors report that more GPUs at the same
iteration count gave a better asymptote for the tracker, whereas OpenHomie's velocity tracking
plateaued beyond 8 GPUs (Fig. S6). The plateau therefore depends on data diversity as well
as N [inference]. For a related saturation of simulator PPO from 4 to 8 GPUs, see
robot-learning-expert's references/papers/supermimic.md.

## 4. Choosing among them

- Plateau with high data duplication (continuous control, narrow initial-state
  distribution): SAPG or EPO; both need an off-policy correction and more code. A wider
  reset distribution is a first-class alternative here (`reset-and-task-sampling.md`
  section 3: OmniReset attributes saturation to revisiting narrow regions of state space).
- Plateau with a wide task or level distribution (procedural levels, terrains, assembly
  variants): change the sampler, but run an easy-start control and a success-driven row
  curriculum first; the winner depends on robot and level set (`reset-and-task-sampling.md`
  section 4). Samplers compose with any learner.
- Plateau that moves when you add minibatches: the Beukman recipe; cheapest to try,
  no new machinery. Its evidence is dense-reward tasks.
- Same-phase batches (fixed-horizon task, short rollout, returns spike every H/K updates,
  or scaling N makes it worse): staggered resets; cheapest to try, composes with any learner.
- Early training unstable, late training noisy: a growing batch schedule.

Before any of these, check that the plateau is not a bug: truncation treated as
termination, stale observation normalization, or advantage normalization over the
wrong axis all produce plateaus too (see `debugging.md`).
