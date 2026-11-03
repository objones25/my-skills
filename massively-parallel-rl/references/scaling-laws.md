# Compute scaling laws for RL

How to predict, from cheap small runs, the best hyperparameters, model size and
performance of a large RL run. The method is Zheng et al. 2026
([arXiv 2610.33420](https://arxiv.org/abs/2610.33420)), itself adapted from DeepSeek LLM
(Bi et al. 2024) and Chinchilla's isoFLOPs profiles (Hoffmann et al. 2022). Numbers below
are from that paper unless another source is named.

## Contents
1. Vocabulary
2. The two-stage protocol, step by step
3. The Elo metric and why it replaces mean return
4. Published coefficients
5. Pitfalls
6. Other RL scaling results and how they differ
7. Running it with the bundled scripts

## 1. Vocabulary

- **C**: total training compute in FLOPs.
- **FPF (M)**: FLOPs per frame, the FLOPs spent on one observation during training.
  Used as the model-size unit because C = 6ND (N params, D tokens) only holds for
  dense transformers. Then **C = M x D**, D = frames trained on.
  `scripts/flops_per_frame.py` measures it and states its accounting assumption
  (the paper does not give its exact formula).
- **Batch size B**: observations per gradient step. Zheng et al. fix the rollout length
  T = 16 and set B through the number of parallel environments N. In their Craftax
  setup each rollout batch is split into 4 minibatches, and "batch size" means
  observations per minibatch (Appendix A.2).
- **IQM**: interquartile mean, the mean of the middle 50% of seeds. Robust to the
  occasional collapsed or lucky seed that RL produces.
- **Competitive configuration**: the best (B, lr) in one bootstrap resample of seeds.
  Fitting the cloud of competitive configs instead of the single empirical argmax
  absorbs the noise in "which cell won".
- **isoFLOPs profile**: at one fixed budget C, train several model sizes (each gets
  D = C / M frames) and see which M scores best.

## 2. The two-stage protocol

Stage 1 sets hyperparameters as functions of C. Stage 2 uses those functions to
train every model in the isoFLOPs sweep, then reads off model size and performance
laws. Stage 1 must come first: an isoFLOPs sweep with mistuned hyperparameters
measures the tuning error, not the model-size trade-off.

### Stage 1: hyperparameter laws

1. Pick budgets spanning about 3 orders of magnitude. Paper: 7 budgets, 1e14 to 1e17.
2. At each budget, grid over batch size and learning rate for a few model sizes.
   Paper: 7 batch sizes (powers of 2) x 7 learning rates (half-octave steps of
   2^-0.5) x 4 widths, several seeds each (Figure 2 aggregates 5). The grids shift
   upward with C (Appendix Tables 4 and 5).
3. Fix the other hyperparameters first with a coarse low-compute search. The paper
   found the A2C entropy coefficient optimal at the same value at every budget, so it
   was not scaled.
4. Bootstrap: resample seeds, take the IQM of each config, keep the argmax (B, lr).
   Paper: 1000 resamples per budget.
5. Fit B*(C) = aB C^dB and lr*(C) = a_lr C^d_lr by least squares in log-log space over
   all competitive configs.
6. Validate at a held-out larger budget (paper: fit 1e14 to 1e16, test 1e17). Success
   means the predicted cell sits inside the high-performance region of the held-out
   grid. Paper: Starpilot predicted config 32.02 vs grid optimum 32.3 return; Craftax
   63.46 vs 64.76.
7. Refit on all budgets before stage 2.

### Stage 2: model size and performance laws

1. Pick budgets (paper: 1e15, 3e15, 1e16, 3e16, 1e17, 1e18).
2. At each, train 4 to 11 widths at fixed depth (Appendix Tables 6 and 7), each with
   the stage 1 hyperparameters for that C, batch rounded to the nearest power of 2
   and lr rescaled by the square-root rule lr x sqrt(B_rounded / B) (Malladi et al.
   2022). 10 seeds per config.
3. Put the final checkpoint of every run, plus a uniform random policy, into one Elo
   league (section 3). IQM over seeds per config.
4. Per budget, fit a parabola Elo ~ log10 M by weighted least squares, weight = 1 /
   variance across seeds. Vertex = (M*, Elo*). The tested widths must bracket the
   vertex; an extrapolated vertex is a guess.
5. Fit M*(C) = aM C^dM and Elo*(C) = dElo log10 C + aElo. Elo is already a log scale,
   so a straight line in log10 C, not a power law.
6. Validate at a held-out budget (paper: 1e18), then spend the big run (paper: 1e19,
   10 seeds).

Paper's validation results:

| Check | Starpilot | Craftax |
| --- | --- | --- |
| Predicted vs optimal FPF at 1e18 | +31% (4.85e8 vs 3.69e8) | +3% (2.73e7 vs 2.65e7) |
| Elo cost of using predicted FPF | -8.1 (-1.17% win rate) | -24.7 (-3.55%) |
| Predicted vs optimal Elo at 1e18 | +40.8 (+5.84%) | -44.5 (-6.37%) |
| Predicted vs achieved Elo at 1e19 | +80.2 (+11.34%) | +80.9 (+11.44%) |

"Win rate error" is the win probability of a player rated the predicted Elo against
one rated the actual Elo, minus 50%: 1 / (1 + 10^(-dElo / 400)) - 0.5. All eight
numbers above reproduce exactly with that formula
(`python scripts/elo_league.py --winrate-from-diff 40.8` prints +5.84%).

Note the 31% FPF miss in Starpilot cost only 8 Elo: the isoFLOPs parabola is flat near
its top, so model-size predictions can be loose without hurting much.

## 3. The Elo metric

Mean episode return need not move smoothly with compute because reward magnitudes
are arbitrary and a new skill can jump return discontinuously. Hilton et al. 2023 hit
the same wall. In the paper, the return law fit Starpilot's fitting range but
extrapolated poorly, and in Craftax it did not even fit the fitting range. Elo
extrapolated within about 11% win rate at 100x the largest fitted budget.

Procedure (Section 3.3):

1. Reserve a fixed evaluation set of environment instances (paper: 1024) and
   evaluate every agent on the same instances.
2. For each pair of agents, each instance is a game: higher return wins, a tie is half
   a win each.
3. Build the win-rate matrix and fit a Bradley-Terry model to get one rating per
   agent. Use the 400-point Elo scale (the paper's win-rate numbers only reproduce on
   that scale).
4. Subtract the random policy's rating so random = 0.

Check that the instances really differ. gymnax MinAtar games have 1 or 2 start
states (Breakout 2; SpaceInvaders and Asterix 1) and their step ignores the random key,
so a greedy policy replays the same one or two games on all 1024 instances; the league
then rests on 2 games while looking like 1024. The original MinAtar adds randomness with
sticky actions (`sticky_action_prob=0.1` by default,
[minatar/environment.py](https://github.com/kenjyoung/MinAtar/blob/master/minatar/environment.py));
gymnax drops it. Pass `--sticky-prob 0.1` to the scaffolds. Both scaffolds warn when the
eval set is degenerate, and `elo_league.py` warns when few instance columns differ.
Procgen and Craftax generate a new level per seed, so they do not have this problem.

Limits the paper states: Elo is relative to the league, so it has no direct task
meaning. Where reward has semantic meaning (a success rate, a cost), report the raw
metric beside Elo. Keep every agent you will compare in one league; ratings from two
separately fit leagues are not comparable.

## 4. Published coefficients (Table 1)

| Environment | aB | dB | a_lr | d_lr | aM | dM | aElo | dElo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Starpilot | 7.14e-4 | 0.39 | 4.08e-3 | -0.023 | 5.3 | 0.44 | -3186 | 240.97 |
| Craftax | 0.28 | 0.26 | 1.57e-2 | -0.082 | 0.53 | 0.43 | -1987 | 223.86 |

Per 10x compute: batch x2.45 (Starpilot) or x1.82 (Craftax); learning rate x0.95 or
x0.83; FPF x2.75 or x2.69. So data D = C / M grows only about x3.6 to x3.7 per 10x compute,
and most extra compute goes to model size and batch.

These coefficients are for A2C with the paper's networks (IMPALA ResNet for
Starpilot; 1x1 conv + linear + LSTM for symbolic Craftax). Do not reuse them for
another algorithm or architecture; reuse the method.

Running `scripts/fit_scaling_laws.py predict` on these coefficients reproduces 13 of
the 14 batch sizes in Appendix Tables 6 and 7 and every learning rate to within 0.06
in log2 units (Table 1 is rounded to 2 to 3 significant figures).

## 5. Pitfalls

- **Text vs table.** Section 4.1 says the Craftax learning-rate exponent is slightly
  positive; Table 1 says -0.082 and Table 7's learning rates fall with C. Trust the
  table. Both exponents are near zero: the learning rate barely moves with compute,
  which matches Beukman et al. 2026 (hold lr fixed as N grows).
- **The learning-rate exponent is weakly identified.** With a half-octave learning-rate
  grid and a true exponent of -0.082, the optimum moves less than one grid step across
  three decades of compute. On six synthetic datasets built that way
  (`tests/test_scripts.py`), the fitted lr exponent ranged from -0.055 to -0.12 while the
  batch exponent stayed within 0.025 of truth. Report a fitted lr exponent within about
  0.05 of zero as "flat", and use finer lr steps near the optimum if it matters.
- **Store coefficients at full precision.** At C = 1e18 an exponent change of 0.005
  moves the prediction by 10^(0.005 x 18) = 1.23x. Table 1's two-decimal dM explains
  most of the gap between recomputed and printed FPFs.
- **Small-compute regime can be non-monotonic.** In Craftax the larger of two models
  won at 1e14, lost at 3e14 and won again at 1e15 (Figure 5), so 1e14 was dropped from
  the fit. Plot raw isoFLOPs curves before fitting; start the fit above any crossover.
  The paper ties this to Ray interference (Schaul et al. 2019): the agent abandons the
  battling skill once it learns the more rewarding dungeon entry, so a small model
  that switches skills beats a large one that would have kept both.
- **No universal law.** The paper argues against a single law for arbitrary RL
  environments because the dynamics above depend on the environment's reward
  structure.
- **Data-hungry by design.** The Craftax record (54.0% vs 20.6% for the previous best)
  used 135B frames against 4B. Compute-optimal is not data-efficient; say which one
  the user's constraint is.
- **Seeds.** RL evaluation variance is high (Henderson et al. 2018). Below about 5
  seeds per config the bootstrap has little to resample.

## 6. Other RL scaling results

| Work | Algorithm | What scales | Key result |
| --- | --- | --- | --- |
| [Hilton et al. 2023](https://arxiv.org/abs/2301.13442) | single-agent RL | model size, compute | "Intrinsic performance" (compute needed to match an agent) makes allocation laws fit; cannot answer the performance question |
| [Jones 2021](https://arxiv.org/abs/2104.03113), [Neumann & Gros 2023](https://arxiv.org/abs/2210.00849) | AlphaZero | compute | Elo rises predictably with compute in two-player games |
| [Rybkin et al. 2025](https://proceedings.mlr.press/v267/rybkin25a.html) | SAC, BRO, PQL | data vs compute, UTD | Data and compute to reach a target return lie on a frontier set by the update-to-data ratio (UTD) |
| [Fu et al. 2025](https://arxiv.org/abs/2508.14881) | value-based (BRO, SimbaV2) | batch, model size, UTD | Best batch grows with model size and shrinks with UTD; small critics overfit big batches ("TD-overfitting") |
| [Wang et al. 2025](https://arxiv.org/abs/2503.14858) | contrastive RL | depth to 1024 layers | Deeper nets use bigger batches; gains jump at critical depths |

Two consequences for a new study: a law in C alone hides the model-size dependence of
batch size that Fu et al. found for value-based methods, and Zheng et al. scale width
only, so a depth-scaled study may find different exponents.

## 7. Scripts

```bash
python scripts/fit_scaling_laws.py hparams grid.csv --boot 1000 --holdout 1e17 --json hp.json
python scripts/fit_scaling_laws.py hparams grid.csv --boot 1000 --json hp.json
python scripts/elo_league.py final_returns.csv --reference random --bootstrap 200
python scripts/fit_scaling_laws.py isoflops iso.csv --holdout 1e18 --json iso.json
python scripts/fit_scaling_laws.py predict --hp hp.json --iso iso.json --flops 1e19
```

CSV formats are in each script's docstring. `scripts/ppo_jax.py --results-csv` appends
rows in the `hparams`/`isoflops` format and `--eval-csv` writes the per-instance
returns `elo_league.py` reads.
