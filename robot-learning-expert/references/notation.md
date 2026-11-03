# Notation and conventions

Symbols follow AJKS (see `sources.md`). This file fixes them once and maps them to the array layout the scripts use.

## Math symbols

| Symbol | Meaning |
|---|---|
| Y^X | set of all functions X → Y. So R^S is "all value-like functions", A^S "all deterministic stationary policies". |
| △(X) | probability distributions on X |
| [N] | {1, ..., N} |
| x_{m:n} | the tuple (x_m, ..., x_n) |
| (S, A, P, r, μ) | MDP: state space, action space, transition map P: S×A → △(S), reward r: S×A → R (deterministic, usually in [0,1]), initial distribution μ |
| P(·) vs P(·) | sans-serif P is "probability of" under an already-specified process; italic P is the transition map |
| E[· \| π] | expectation over trajectories with s_0 ~ μ, a_t ~ π(s_t), s_{t+1} ~ P(·\|s_t, a_t) |
| V^π, Q^π | discounted value and action-value of π; Q^π(s,a) = value of taking a once, then following π |
| V^⋆, Q^⋆ | per-state maxima over all (history-dependent) policies |
| T^π, T^⋆ | policy and optimal Bellman operators (on V or Q, overloaded) |
| P^π | closed-loop state-action kernel P^π(s',a'\|s,a) = P(s'\|s,a) π(a'\|s') |
| π_V, π_Q | a greedy policy with respect to V or Q (Def. 14) |
| ‖f‖_∞ | sup_x \|f(x)\| |
| H, h | horizon and step index; finite-horizon sums run t = 1..H (the discounted sum starts at t = 0) |
| Δ_t(s) | action gap: V⋆(s) minus the best non-optimal Q⋆ (Notes 2 §3.3) |
| N_h(s,a) | visit count of (s, a) at step h (Notes 5 §3) |
| b_h(s,a), ι | exploration bonus; ι = log(SAT/δ) is the log factor (notes write ℓ) |
| α_t | Q-learning step size; Robbins-Monro: Σα = ∞, Σα² < ∞ (Notes 5 eq. 1) |
| T̂⋆_N | empirical optimal Bellman operator averaging N sampled next states (Notes 5 §2) |
| φ(s,a) ∈ R^d, θ⋆, μ_i | linear MDP features, reward weights, transition measures (Notes 6 Def. 1) |
| Λ_h | ridge Gram matrix λI + Σ φφᵀ at step h (Notes 6 Alg. 2) |
| F = {f_θ} | parameterized Q class; Bellman complete if T⋆F ⊆ F (Notes 7 Def. 1) |
| T̃⋆_D | least-squares (FQI) backup on dataset D (Notes 7 eq. 1) |
| d^π_μ(s,a) | discounted visitation distribution (Notes 7 Def. 2) |
| C (concentrability) | sup over π, s, a of d^π_μ(s,a)/ν(s,a) (Notes 7 §3) |

Bandits use capital A_t, X_t for the random action and reward and n for the number of rounds (Notes 3 §2.1). LQR reuses Q and R for cost matrices and writes cost c = −r (Notes 2 §2.1.2).

## Array conventions for tabular code (used by `scripts/mdp_tools.py` and `scripts/local_sim.py`)

| Object | Shape | Indexing |
|---|---|---|
| P | (S, A, S) | P[s, a, s'] = P(s'\|s, a). Note the order differs from the math P(s'\|s,a). |
| r | (S, A) | r[s, a] |
| π | (S, A) | π[s, a] = π(a\|s); greedy policies are one-hot rows |
| V | (S,) | |
| Q | (S, A) | |
| generative samples | (S, A, k) ints | k next-state draws per (s, a) |
| rollout arrays (`scripts/rl_math.py`) | (T, N) | time first, then parallel env, like rsl_rl's rollout storage |

Shape reasoning for the tabular operators:
- `P @ V` contracts the last axis: (S, A, S) @ (S,) → (S, A), giving E_{s'}[V(s')] for every (s, a).
- `(pi * Q).sum(axis=1)` gives V^π from Q^π.
- `Q.max(axis=1)` is (S,); `P @ Q.max(axis=1)` is the expected next-state max.
- Use one NumPy `Generator` seeded from a `seed` argument; same seed must reproduce, different seeds must differ.
