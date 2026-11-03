# Known MDPs: Bellman operators and exact dynamic programming

Sources: Notes 1, AJKS ch. 1. Notes 1 §1 says much of what comes later "can be understood as an attempt to approximate the behavior of these algorithms" with reasonable memory and computation, so most RL bugs and most RL theory trace back to a statement here.

## Setting

- Protocols (§2.3): fixed-length episodic (horizon H, then reset to μ); variable-length episodic (terminal states); continual with resets (learner chooses when to reset to μ); single trajectory (no resets, hardest; the "enticing trap" MDP of Fig. 1 shows why: one bad action traps you forever).
- Markov property P(s_{t+1} | s_t, a_t) = P(s_{t+1} | whole history). Read it as "the state is rich enough", not as a restriction; time-varying dynamics become Markov by adding t to the state.
- Policy classes: history-dependent ⊃ Markov time-varying S×N → △(A) ⊃ stationary S → △(A); deterministic if point masses.
- POMDP: observe o_t ~ O(s_t, a_t) instead of s_t. Treated in `../training/reward-and-observation-design.md` (history, latents).

## Discounted objective

V^π(s) = E[Σ_{t≥0} γ^t r(s_t, a_t) | π, s_0 = s], γ ∈ [0,1). With r ∈ [0,1], 0 ≤ V^π ≤ 1/(1−γ) by the geometric series. γ = 1 is degenerate except special cases like LQR; γ = 0 ignores dynamics.

Theorem 4: there is an optimal policy that is history-independent, stationary and deterministic, realizing V⋆ and Q⋆ at every state simultaneously: π(s) ∈ argmax_a E[r(s,a) + γ V⋆(s_1)]. Finite horizon breaks stationarity (see below).

## Operators and their facts

| Operator | Definition | Fixed point | Facts |
|---|---|---|---|
| T^π on V (Def. 6) | (T^π V)(s) = E_{a~π(s), s'~P}[r(s,a) + γ V(s')] | V^π (policy Bellman equation) | γ-contraction in ‖·‖_∞ (Thm 11); monotone |
| T^π on Q (Def. 12) | (T^π Q)(s,a) = r(s,a) + γ E_{s', a'~π(s')}[Q(s',a')] = r + γ P^π Q | Q^π | same (Exercise 2) |
| T⋆ on V (Def. 16) | max_a [r(s,a) + γ E V(s')] | V⋆ (Thm 20) | γ-contraction (Thm 19, via \|max f − max g\| ≤ max\|f − g\|, Lemma 18); monotone (Prop. 17) |
| T⋆ on Q (Def. 23) | r(s,a) + γ E_{s'} max_{a'} Q(s',a') | Q⋆ (Thm 24) | γ-contraction |

Banach fixed point (Thm 10): a γ-contraction on a complete normed space has a unique fixed point and iterates converge as ‖x_n − x⋆‖ ≤ γ^n ‖x_0 − x⋆‖. Contraction depends on the norm; here it is always ‖·‖_∞.

## Algorithms

| Algorithm | Update | Converges to | Rate / cost |
|---|---|---|---|
| Policy evaluation (Alg. 1) | V ← T^π V | V^π | ‖V_k − V^π‖_∞ ≤ γ^k ‖V_0 − V^π‖_∞ |
| Closed-form evaluation (§4.5) | Q^π = (I − γ P^π)^{-1} r | Q^π exactly | one SA×SA solve; dead end for continuous S |
| Value iteration (Alg. 3) | V ← T⋆ V | V⋆ | ‖V_k − V⋆‖_∞ ≤ γ^k/(1−γ) from V_0 = 0 (Prop. 21) |
| Q-value iteration (§6.2) | Q ← T⋆ Q | Q⋆ | same rate; O(SA) memory but O(A) action selection |
| Policy iteration (Alg. 2) | Q_i = Q^{π_i}; π_{i+1} = greedy(Q_i) | π⋆ | ‖Q⋆ − Q^{π_{i+1}}‖ ≤ γ‖Q⋆ − Q^{π_i}‖ (Thm 26); finite termination for finite S, A with exact evaluation |
| Generalized PI (§7.1) | N evaluation sweeps per improvement | | N = 1 is exactly Q-value iteration (eq. 15) |

Iterations for ε accuracy with r ∈ [0,1]: k ≥ (log(1/ε) + log(1/(1−γ))) / (1−γ), using 1−γ ≤ −log γ (§4.3). The 1/(1−γ) "effective horizon" factor keeps reappearing.

Greedy-policy loss (the bound that matters for control):
- From a value-like f: V^{π_f}(s) ≥ V⋆(s) − 2γ/(1−γ) ‖f − V⋆‖_∞ (Thm 22, tight).
- From a Q-like f: V^{π_f}(s) ≥ V⋆(s) − 2/(1−γ) ‖f − Q⋆‖_∞ (Thm 25).
So a small value error can still mean a 1/(1−γ)-amplified policy loss, one reason to measure true suboptimality (E⋆ in `model-based.md`) separately from model error (E^P) [inference].

Lemma 15 (policy improvement): Q^{π_{i+1}} ≥ Q^{π_i} pointwise. Weak on its own (allows no progress); Thm 26 combines it with T⋆ to get the γ rate.

## Finite horizon (§8)

Objective E[Σ_{t=1}^H r_t(s_t, a_t)], no discount. Optimal V, Q, π are time-indexed even for stationary P, r (enticing trap: stay for H−1 steps, then grab the 1). Backward induction: V_{H+1} = 0, Q_h = r_h + P_h V_{h+1}, V_h = max_a Q_h. Exact in H sweeps instead of approximately in O(1/(1−γ)), at H times the memory. Time-varying P_h, r_h cost nothing extra.

## Why tabular stops working (§9)

Every operator hides sweeps: max over A, expectation over s', and the outer loop over all s (or s, a). Discretizing d torque joints into 1000 levels gives |A| = 1000^d. Tabular methods also ignore that nearby bit patterns are nearby states. Two missing pieces define the rest of the course: function approximation (continuous S, A) and unknown P, r (learning).

## Array forms (for checking, not for pasting into graded files)

`scripts/mdp_tools.py` implements exact evaluation, Q-value iteration with a certified stopping rule (‖Q_k − Q⋆‖ ≤ γ/(1−γ) ‖Q_k − Q_{k−1}‖), policy iteration with exact evaluation, backward induction, discounted occupancy, and a combination-lock MDP. Use them as oracles: compare your iterative result against the closed-form one on random MDPs with Dirichlet α ∈ {0.1, 1, 10}.

## Degenerate MDPs with hand-computable answers

| MDP | Answer |
|---|---|
| Actions have no effect, reward 1 everywhere | V⋆ = 1/(1−γ) everywhere |
| Action picks the next state | move to the best-reward state, then stay |
| Enticing trap, γ = 0.9 (Fig. 1) | stay: 0.5/(1−γ) = 5 > go: 1; T^π from 0 gives 0.5, then 0.95 |
| 2 states, stay/switch, reward only r(1, stay) = 1, γ = 0.5 | V⋆ = (1, 2), Q⋆(0) = (0.5, 1), Q⋆(1) = (2, 0.5) |
| Single state | V = r/(1−γ) for the best action |
| H = 1 | Q_1 = r, policy is greedy on r |
