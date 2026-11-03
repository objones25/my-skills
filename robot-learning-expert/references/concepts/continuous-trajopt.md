# Known MDPs with infinite state spaces: LQR and trajectory optimization

Sources: Notes 2, BBC, Howell et al. 2022. Two escapes from the tabular sweep problem: special structure where Bellman operators stay finite-dimensional (LQR), and giving up on whole policies to optimize one trajectory locally.

## LQR (§2.1)

Dynamics s_{t+1} = A s_t + B a_t + w_t with S = R^n, A = R^m; cost c(s,a) = s'Qs + a'Ra, Q ⪰ 0 (positive semidefinite: x'Qx ≥ 0 for all x), R ≻ 0 (positive definite: x'Rx > 0 for x ≠ 0). Stabilizability (every mode is controllable or decays on its own) is assumed.

Finite horizon (Thm 2): V⋆_h(s) = s' P_h s with P_h ⪰ 0, by backward induction. Base case a_H = 0, P_H = Q. The step:
1. Substitute s' = As + Ba into s'Qs + a'Ra + s''P's'. The result J(a) is a convex quadratic in a.
2. Set ∇J = 0: a = K s with K = −(R + B'P'B)^{-1} B'P'A (invertible because R ≻ 0). The optimal action is linear in the state.
3. Substitute back: P = Q + K'RK + (A + BK)'P'(A + BK), again PSD.
Time-varying A, B, Q, R cost nothing extra. With Gaussian noise the policy is the same; the value picks up a constant.

Infinite horizon, undiscounted (§2.1.4): guess V(s) = s'Ps, apply one backup, require a fixed point. This gives the DARE (discrete-time algebraic Riccati equation):
P = Q + A'PA − A'PB (R + B'PB)^{-1} B'PA.
It has a solution unless (A, B) is not stabilizable; no closed form, but standard solvers handle large n, m. An optimization over an infinite-dimensional function space collapses to a matrix equation.

Limits (§2.1.5): box action limits break linearity of the optimal policy (it is not even the projection of Ks). LQR needs no strategic exploration, so ideas that work on LQR may not transfer. Use it as a sanity check: an RL method that cannot learn LQR is probably broken. `scripts/lqr.py` gives the exact P, K, the DARE residual, rollout costs, and least-squares system ID.

Sign convention trap: the notes write a = Ks with the minus inside K; AJKS ch. 14 writes π⋆(x) = −K⋆x.

## Trajectory optimization (§3)

Problem 3 (action-space, stochastic): argmax over a_1..a_H of E[Σ r_t(s_t, a_t)] from a fixed s_1. Open-loop plans cannot react to realized randomness: in the 7-state Fig. 1 example no fixed action sequence gets more than 1/2 but a feedback policy gets 1. Planning on a wrong model is a second failure mode. In practice: "plan, then track" in predictable settings (welding robots), continual replanning elsewhere (cars), i.e. MPC.

Deterministic dynamics s_{t+1} = p_t(s_t, a_t):

| Formulation | Variables | Pros | Cons |
|---|---|---|---|
| Problem 4 (action-space; often called single shooting) | a_1..a_H; states are nested functions p(p(p(s,a),a),a) | small, no constraints | early actions dominate, badly conditioned, deep recursion |
| Problem 5 ((state, action)-space; often called direct transcription or simultaneous) | a_1..a_H and s_2..s_H with s_{t+1} = p_t(s_t, a_t) as equality constraints | well conditioned | bigger problem, needs a constrained solver |

Assumption 6 (convex trajectory optimization): S = R^n, convex A, linear dynamics, reward concave in (s, a). Then both problems are convex, and convex state/action constraints, slew limits and time-varying A are easy. Usually inadequate for robots because rotations and contact make dynamics nonlinear and state spaces nonconvex (a kinematic car needs SO(2) or a wrapped angle). General nonconvex trajectory optimization is NP-hard (Exercise 3 encodes TSP), so aim for local optima from good initial guesses.

## Discrete actions: why search blows up and when it does not (§3.3)

With finite S, many of the |A|^H sequences hit the same states (pigeonholing) and costs add over time (optimal substructure), so DP is polynomial. With continuous S every sequence can reach a distinct state, and the "needle in a haystack" MDP forces 2^H work for any search.

k-QVI-solvable (Def. 7, Laidlaw et al. 2023): applying k steps of optimal Bellman backup to Q^{π_rand} (the random policy's Q) already yields optimal actions at every (h, s). Laidlaw et al. found many benchmark MDPs, including Atari games, are k-QVI-solvable for small k; over half were 1-QVI-solvable. The needle-in-a-haystack MDP is also 1-QVI-solvable, so this alone is a weak notion of easiness.

GORP (Alg. 1, Greedy Over Random Policy): at each step i, exhaustively try all A^k continuations, estimate the return of each by m random rollouts, commit to the first action of the best. Effective horizon (Def. 8): H_k = k + log_{|A|} m_k, where m_k is the fewest rollouts for which GORP(k, m) returns the optimal sequence with probability ≥ 1/2; the effective horizon is min_k H_k ≤ H, and GORP's cost |A|^k m_k = |A|^{H_k} equals exhaustive search at that horizon. m_k is governed by the action gap Δ_t(s). Laidlaw et al. found it tracks empirical RL sample complexity better than other measures. Takeaway in the notes: "learning and function approximation can be useful or necessary for purely computational reasons", even with a known model.

## Predictive sampling

Predictive sampling (Howell et al. 2022) is a batched random search:
1. Sample K perturbations ε^k_t ~ N(0, σ²I) of the current plan a_{1:H}.
2. Project each perturbed action onto the action box (Π_A clips each coordinate).
3. Roll out all K sequences from s_1 through the model.
4. Sum rewards per rollout; keep the single best sequence as the new plan.
Repeat N times. Worked example: a unicycle with (x, y, θ) state and (v, ω) action, v ∈ [0, v_max], ω ∈ [−ω_max, ω_max], reward −[(x−x⋆)² + (y−y⋆)² + λ(v² + ω²)] in a grid maze where blocked moves leave the state unchanged.

Variations worth trying: decaying variance, reaching the cell center precisely, other mazes to see the limitations of sampling-based trajectory optimization.
