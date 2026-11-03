# An Empirical Study of Reinforcement Learning Scaling Laws

Zeyu Zheng, Clare Lyle, Khimya Khetarpal, Will Dabney (Google DeepMind). arXiv preprint, no venue printed. arXiv:2610.33420 (v1, 24 Oct 2026).

## The idea

Treat an RL training run like an LLM pretraining run: from small grid searches, fit power laws in total training compute C (FLOPs) for the best batch size, the best learning rate and the compute-optimal model size, then check the predictions at 10× and 100× more compute [Scaling §1, §3]. The algorithm is A2C (advantage actor-critic) on Starpilot (Procgen) and Craftax, and batch size is changed only by changing the number of parallel environments at a fixed rollout length of 16 [Scaling §3.1, App. A.2]. Performance is measured with a single-agent Elo score, which extrapolates log-linearly in C while mean episode return does not [Scaling §3.3, Fig. 4]. A compute-optimal Craftax run at 1e19 FLOPs reaches 54.0% versus a previous best of 20.6% [Scaling §4.3].

## Notation

| Symbol | Meaning |
|---|---|
| C | Total training compute in FLOPs [Scaling §3.2] |
| M | Model size measured as FLOPs-per-frame (FPF): FLOPs spent on each observation during training [§3.2] |
| D | Number of frames (environment observations) trained on; C = M·D [§3.2] |
| B*(C) = a_B C^{d_B} | Average competitive batch size at compute C [Eq. 1] |
| α*(C) = a_α C^{d_α} | Average competitive learning rate at compute C [Eq. 2] |
| M*(C) = a_M C^{d_M} | Compute-optimal FPF [Eq. 3] |
| Elo*(C) = d_Elo log₁₀C + a_Elo | Best achievable Elo at compute C [Eq. 4] |
| a_·, d_· | Fitted coefficient and exponent of each law [Tab. 1] |
| IQM | Interquartile mean: mean of the middle 50% of seed results [§3.1] |
| Competitive configuration | The best (batch, LR) pair in one bootstrap resample of the seeds [§3.1] |
| isoFLOPs profile | At a fixed C, train several model sizes and find the best one (Hoffmann et al. 2022) [§3.2] |
| Elo | Rating from a Bradley-Terry model fit to pairwise win rates between agents [§3.3] |
| V-trace | IMPALA's importance-weighted correction for slightly off-policy data, used in Craftax [App. A.2] |
| Ray interference | Learning one reward source suppresses learning from another early in training (Schaul et al. 2019) [§4.4] |

## The method

### 1. Hyperparameter laws [Scaling §3.1, §4.1]
A preliminary search found the best entropy coefficient constant across budgets, so only batch size and learning rate are studied [§3.1]. Per budget: 7 learning rates × 7 batch sizes × 4 model sizes, several seeds, aggregated by IQM [§4.1]. Resample seeds 1000 times (bootstrap); each resample's best pair is a competitive configuration; fit Eqs. 1-2 to all of them, not to the single empirical best [§3.1, §4.1]. Validation fits on 1e14 to 1e16 (Craftax from 3e14) and tests at 1e17; the laws used later are refit on everything up to 1e17 [§4.1].

### 2. Allocation law (model size) [Scaling §3.2, §4.2]
Budgets 1e15, 3e15, 1e16, 3e16, 1e17, 1e18. Depth fixed, width varied; batch and LR from step 1, batch rounded to the nearest power of 2 and LR adjusted by the square-root rule [§4.2, App. B]. 10 seeds per configuration, IQM of Elo. Fit a parabola of Elo against log₁₀M by weighted least squares (weights = inverse seed variance), take its peak as M*, fit Eq. 3 on 1e15 to 1e17, test at 1e18. D* follows from C/M*.

### 3. Elo performance metric [Scaling §3.3]
League = the final checkpoint of every run plus a uniform random policy. On 1024 held-out environment instances [App. A.2], each pair of agents plays one "game" per instance: higher return wins, a tie counts 0.5. Fit Bradley-Terry to the win-rate matrix, then subtract the random policy's Elo. Fit Eq. 4 on the parabola peaks.

### 4. Validation at scale [Scaling §3, §4.3]
10× (1e18) compares predictions with the empirical optimum. 100× (1e19) is a "hero run" at the predicted model size, 10 seeds, compared with the predicted Elo.

### Fitted laws [Scaling Tab. 1]

| Env | a_B | d_B | a_α | d_α | a_M | d_M | a_Elo | d_Elo |
|---|---|---|---|---|---|---|---|---|
| Starpilot | 7.14e−4 | 0.39 | 4.08e−3 | −0.023 | 5.3 | 0.44 | −3186 | 240.97 |
| Craftax | 0.28 | 0.26 | 1.57e−2 | −0.082 | 0.53 | 0.43 | −1987 | 223.86 |

Per 10× compute, from Tab. 1 [inference, arithmetic]: batch ×2.45 (Starpilot) / ×1.82 (Craftax); LR ×0.95 / ×0.83; FPF ×2.75 / ×2.69; frames D* = C/M* ×3.63 / ×3.72; number of batches D*/B* ×1.48 / ×2.04. So at higher budgets both the batch and the number of updates grow, and data grows faster than model size.

Configurations actually run [Scaling Tab. 6, Tab. 7]:

| C | Starpilot batch, LR | Craftax batch, LR |
|---|---|---|
| 1e15 | 2^9, 2^−9.12 | 2^11, 2^−10.17 |
| 1e16 | 2^10, 2^−9.35 | 2^12, 2^−10.38 |
| 1e17 | 2^12, 2^−9.07 | 2^13, 2^−10.58 |
| 1e18 | 2^13, 2^−9.30 | 2^14, 2^−10.79 |
| 1e19 (hero) | 2^14, 2^−9.53, width 43 | 2^15, 2^−11.00, width 58 |

### Training details [Scaling App. A]
Both: A2C, γ = 0.999, rollout length 16, value-loss weight 0.5, LR decayed linearly to 0, global grad-norm clip 0.5, Adam ε = 1e−8. Starpilot: IMPALA ResNet with shared policy/value torso and LayerNorm, width = first-layer channels, entropy 0.01, Adam β = (0.9, 0.999). Craftax: symbolic observations, 1×1 conv with E channels, linear to H = 16E, inventory linear to H, LayerNorm each then add, LSTM with H units, GELU, entropy 0.001, Adam β = (0.9, 0.99); each rollout batch is split into 4 minibatches (4 gradient steps), with V-trace for the resulting off-policyness, and "batch size" means observations per minibatch.

## Results that matter

- Hyperparameters at 1e17 [Scaling §4.1, Fig. 2]: predicted config returns 32.02 vs optimum 32.3 (Starpilot) and 63.46 vs 64.76 (Craftax); predictions sit near the center of the high-return region.
- Model size at 1e18 [Tab. 2]: predicted FPF 4.85e8 vs optimal 3.69e8 (+31%, Elo 1102.6 vs 1110.7, win-rate error −1.17%) for Starpilot; 2.73e7 vs 2.65e7 (+3%, Elo 2062.8 vs 2087.5, −3.55%) for Craftax.
- Elo at 1e18 [Tab. 3]: predicted 1151.5 vs optimal 1110.7 (+5.84% win rate) Starpilot; 2043.0 vs 2087.5 (−6.37%) Craftax.
- Hero runs at 1e19 [Tab. 3, §4.3]: predicted 1392.5 vs actual 1312.3 (Starpilot), 2266.8 vs 2185.9 (Craftax); win-rate errors +11.34% and +11.44%.
- Mean return as the metric [Fig. 4 right]: Starpilot fits but extrapolates poorly; Craftax does not even fit. At 1e19 the Craftax hero return is about 122 against a fitted line near 93; Starpilot about 43 against about 63 (read from Fig. 4, approximate).
- Craftax state of the art [§4.3]: 54.0% vs 20.6% (Hamon 2024), using 135 billion frames vs 4 billion; the authors say the target is compute optimality, not data efficiency.
- Inverse scaling in Craftax [§4.4, Fig. 5]: FPF 1.49e6 beats FPF 3.95e5 at 1e14, loses at 3e14 (about 25.4 vs 27.8 mean return), wins at 1e15 (about 36 vs 34.5) (read from Fig. 5, approximate). Achievement curves show the agent switching from fighting enemies to entering the dungeon around 3e14; the smaller model sees more frames for the same C.

## Use it for

- G1 num_envs sweep: the paper's batch knob is the sweep's knob (more envs at fixed steps per env), and its main finding is that the best batch is a function of the compute budget, not a constant [§3.1, Tab. 1]. So "best num_envs" only means something at a stated budget; compare runs at matched compute or matched env steps and expect the optimum to move with the budget [inference].
- Methodology to copy: several seeds per num_envs, IQM, bootstrap over seeds to get a distribution of the best setting, a power-law fit of best num_envs against budget, held-out larger budget to test it [§3.1, §4.1; transfer is inference]. This matches the IQM/bootstrap hygiene in `../training/evaluation.md`.
- Learning rate barely moved (exponents −0.023 and −0.082) [Tab. 1]. The usual "scale LR with batch" rules were used only to adjust for power-of-2 rounding [App. B].
- Compute accounting differs for robots: C = M·D counts network FLOPs only. In legged_gym-style training physics dominates step cost (`legged-gym.md`, App. A.1), so a FLOPs budget is a poor proxy for A100 wall-clock; use env steps or GPU-seconds as the budget axis [inference].
- Shaped locomotion rewards are sums of weighted terms; if mean return is noisy across runs, a league/Elo comparison on held-out commands or seeds is an option [§3.3; application is inference].
- Topics: policy gradient and actor-critic (A2C, V-trace importance weighting); evaluation methodology.
- Reset-based exploration: little direct bearing. Ray interference (one skill crowding out another) is a reminder that what the batch contains matters, which is also the argument for reset distributions [§4.4; link is inference].

## Gotchas / ambiguities

- Learning-rate sign: §4.1 says the LR exponent is slightly negative in Starpilot and slightly positive in Craftax and that this "contradicts the common wisdom that it should decrease". Tab. 1 gives −0.082 for Craftax, and Tab. 7's LRs fall from 2^−10.17 (1e15) to 2^−11.00 (1e19), so both decrease slightly; the sentence looks wrong (also flagged in the survey's survey doc).
- "Batch size" means two things: rollout batch (envs × 16) for Starpilot, observations per minibatch (a quarter of the rollout batch) for Craftax [App. A.2]. Craftax's num_envs is therefore 4B/16 = B/4, e.g. 8192 at the 1e19 setting [inference, arithmetic].
- §4.1 says 7 budgets from 1e14 to 1e17 and that Craftax drops 1e14; Tab. 5 (Craftax grid) has only 6 rows starting at 3e14.
- "Deviations are +80.2 and +80.9" Elo at 1e19 [§4.3]: actual is below predicted in both, so the hero runs fell short of the law by about 80 Elo.
- The 54.0% Craftax metric is not defined in the paper.
- The Bradley-Terry scale is not given, so Elo-to-win-rate conversions cannot be reproduced from the text.
- Fig. 2's heatmaps do not say which of the 4 model sizes they show.
- Fitted exponents have no reported uncertainty, though bootstrapping was done [Tab. 1].
- Laws are for A2C only; the authors argue against universal RL scaling laws because of environment-specific effects like Ray interference [§5].
