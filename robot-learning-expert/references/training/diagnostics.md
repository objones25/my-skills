# Testing and debugging RL code

RL code is numerical and randomized, so a wrong implementation often runs, prints plausible numbers, and fails only after hours of training or on someone else's test. Test degenerate cases locally (one state, actions with no effect, uniform rewards, episode length 1); the rest is general practice `[practice]`.

## Test ladder for anything tabular or model-based

1. Shapes and invariants: P rows sum to 1, π rows sum to 1, V within [0, 1/(1−γ)] for rewards in [0,1], one-hot greedy policies. `mdp_tools.check_mdp` fails fast with the offending (s, a).
2. Hand-computed MDPs (`mdp-dp.md`, last table): uniform, single state, H = 1, enticing trap, the 2-state stay/switch MDP. Assert exact numbers after 1 and 2 steps from zero as well as at the fixed point; the first step catches missing γ, the second catches wrong axis in `P @ V`.
3. Two independent routes to one number: iterative policy evaluation vs `(I − γP^π)^{-1} r`; value iteration vs policy iteration; finite-horizon DP vs brute-force enumeration of A^H sequences for tiny H.
4. Random MDPs across regimes: Dirichlet α = 0.1 (near-deterministic), 1, 10 (near-uniform), several seeds, S ≠ A so transposed axes crash instead of passing silently.
5. Properties: contraction ‖T f − T g‖_∞ ≤ γ‖f − g‖_∞ on random pairs; monotonicity f ≥ g ⇒ Tf ≥ Tg; Lemma 15 (policy iteration never decreases Q pointwise).
6. Seeds: same seed gives identical output, different seeds differ. Build one `np.random.default_rng(seed)` and draw sub-seeds from it; never call the global `np.random.*`.
7. Statistical tests for samplers: compare empirical frequencies of `generate_samples` to P with a tolerance like 4·sqrt(p(1−p)/n), not exact equality.

## Common bugs

| Symptom | Usual cause |
|---|---|
| Values right for V, wrong for Q (or the reverse) | expectation over a' ~ π(s') applied at s instead of s' |
| Works when S = A, crashes or silently wrong otherwise | P axes in math order P(s'\|s,a) instead of code order P[s, a, s'] |
| Policy iteration never terminates | argmax ties flip between equal actions; keep the current action when tied |
| Finite-horizon answer off by one step | mixing t = 1..H (notes) with 0..H−1 (code); forgetting V_{H+1} = 0 |
| Value iteration "converges" to the wrong thing | in-place updates mixing old and new values (that is Gauss-Seidel, a different algorithm), or starting from nonzero when tests expect zero |
| Same random numbers every call | reseeding inside a loop with the same seed, or a restored simulator that also restored its RNG (`local_sim.py` docstring) |
| Trajectory optimizer stuck at start | variance too small for the action scale, forgot to project to the action box, or rewards summed over the wrong axis (H vs K) |
| Model-based error does not shrink with data | counts normalized by the wrong total, or evaluating π̂ under P̂ where the task asks for the true P |

## Deep RL debugging ladder `[practice]`

1. Known answer first: LQR (`scripts/lqr.py` gives the optimal cost), a tabular MDP through a one-hot encoder (compare to `mdp_tools` Q⋆), Pendulum or CartPole. `examples/ppo_pendulum.py` is a 200-line PPO that reaches a deterministic eval return near −130 on Pendulum-v1 in about 75 s of CPU (measured over 3 seeds: −122 to −135); if yours does not, compare against it.
2. Overfit a tiny problem (one state, one episode) to confirm the update can drive its loss to zero.
3. Unit-test the math that silently biases results: GAE with terminated vs truncated, returns, KL estimators, explained variance (`scripts/rl_math.py` and its tests are the reference).
4. Change one thing at a time; several seeds before believing a difference (`evaluation.md`).
5. If a paper's number does not reproduce, read its "gotchas" section in `papers/` before suspecting your algorithm.

## PPO health metrics and what to do

| Metric | Healthy (typical) | Too low means | Too high means |
|---|---|---|---|
| approx KL per update (k3) | around desired_kl (0.01 in rsl_rl) | lr too small or advantages near zero; learning stalled | lr too large, too many epochs, minibatches too small; rsl_rl's adaptive lr should cut lr by 1.5× |
| clip fraction | roughly 0.05 to 0.3 | updates barely move the policy | policy moving far each update; same fixes as high KL |
| explained variance of V | rising toward 0.8 to 1 | critic not learning: reward scale, missing obs normalization, value lr, too few epochs | (cannot be too high; near 1 with flat return can mean rewards are trivially predictable) |
| policy std / entropy | decays slowly | collapsed exploration: entropy_coef up, init_std up | policy ignoring the critic; check advantage sign and normalization |
| value loss | falls then tracks | | rising for many iterations: reward scale change, curriculum jump, or bootstrapping bug |
| grad norm (pre-clip) | below max_grad_norm most of the time | | always clipped: lr too high or exploding activations |
| Episode length | rises to the time limit | (falls early) termination too strict or reward pays for dying | all envs hit the limit on the same iteration: episode lengths not randomized after reset or resume |
| per-term Episode_Reward | tracking terms rise | | a penalty term dominates: policy learned to stand still |

Numbers in the "healthy" column are rules of thumb, not sourced thresholds (the verified sources give only rsl_rl's desired_kl of 0.01 and the 37-details blog's approx_kl below 0.02, `papers/canon-algorithms.md`); calibrate them against your own known-good run, e.g. `examples/ppo_pendulum.py`, which logs all of them.

## Training-system bugs that look like RL bugs

| Symptom | Usual cause |
|---|---|
| Reward oscillates between full and near zero every episode length | all envs synchronized on time-outs (episode lengths not re-randomized after resume; G1 case study, Oct 30) |
| W&B metrics stop appearing after some step | a writer logged at a larger step first and W&B silently drops lower steps (rsl_rl 5.4.2 writes `*/time` at step = total seconds; the G1 case study, drops those keys) |
| Throughput collapses at large N | per-step `.item()` / `.cpu()` syncs in the rollout, or a CPU-side env wrapper |
| Resume gives a jump in curves | optimizer state, obs normalizer, lr schedule or curriculum state not saved in the checkpoint |
| Big-N run is worse per sample and no better per minute | same 20 gradient steps per iteration but fewer iterations per minute, so extra data buys smoother gradients, not more updates (G1: 16384 used about 2× the env steps of 4096 for the same 45-min error); see `on-policy-at-scale.md` |
| Values right in sim eval, policy falls on hardware | sim-to-real gap, not a training bug (`sim-to-real.md`) |
