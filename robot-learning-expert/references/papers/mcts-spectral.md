# Monte Carlo Tree Search with Spectral Expansion for Planning with Dynamical Systems

Benjamin Riviere, John Lathrop, Soon-Jo Chung (Caltech). Science Robotics, Vol. 9, Issue 97, 2024 (accepted version, as stated on the arXiv PDF). arXiv:2412.11270 (v1, Dec 2024).

## The idea

MCTS needs a discrete tree, but robots have continuous states and actions, and uniform discretization makes trees too wide and too deep. SETS (Spectral Expansion Tree Search) builds each node's children from the eigenvectors of the locally linearized controllability Gramian: 2n branches (n = state dimension), each a length-H trajectory that tracks one "natural motion" with LQR feedback [MCTS Methods]. The paper proves this discrete tree's optimal value is close to the continuous optimum (Thm 1), that its MCTS variant converges at rate ℓ^{−1/2} without knowing the gap (Thm 2), and combines them (Thm 3).

## Notation

[MCTS Methods, Problem formulation; Supplement Defs 1 to 4]. Warning: H here is the branch length (steps per tree edge), not the RL episode horizon. The episode horizon is K.

- Deterministic planning MDP <X, U, F, R, D, Ω, K, γ>: compact state space X ⊆ R^n, compact action space U ⊆ R^m, discrete-time dynamics x_k = F(x_{k−1}, u_k), stage reward R ∈ [0, 1], terminal reward D ≥ 0, unsafe set Ω ⊆ X, horizon K (timesteps), discount γ ∈ [0, 1).
- Objective (Eq. 1): maximize Σ_{k=1}^K γ^k R(x_k, u_{k+1}) + γ^K D(x_K) subject to dynamics, x_k ∉ Ω, u_k ∈ U. V*(x_0) is the optimal value.
- Δt: duration of one discrete timestep (continuous-time systems are discretized).
- H: number of timesteps per tree edge. The tree has depth ⌈K/H⌉ and branching factor at most 2n.
- ℓ: MCTS iteration (rollout) count. For a node i: C(i) children, r(i) reward along its edge, T(i, ℓ) visit count, Ṽ(i, ℓ) total backed-up value, V(i, ℓ) = Ṽ/T average value.
- Linearization L_k(x, u) = F(x̄_k, ū_{k+1}) + ∇_x F (x − x̄_k) + ∇_u F (u − ū_{k+1}) along a nominal trajectory [Supp Def 4]; written z_{k+1} = A_k z_k + B_k u_{k+1} + c_k.
- Controllability matrix C = [(Π_{k=1}^{H−1} A_k) B_0 S, ..., A_{H−1} B_{H−2} S, B_{H−1} S], with S = diag(2/(ū_j − u_j)) rescaling each input by its control range. It maps an input sequence to the terminal state. The Gramian is CC^T.
- Reachable set R_F(x_0, U) = {F(x_0, u) : u ∈ U} [Supp Def 2]; Hausdorff distance d_S(M, N) = max{sup_{m∈M} d(m, N), sup_{n∈N} d(M, n)} [Supp Def 3].
- κ_0, ..., κ_8: problem-specific constants (Lipschitz constants, linearization error, σ_max(C), etc.).

## Core definitions

Spectral expansion [MCTS Alg 1, lines 15 to 28; Methods "Spectral Expansion Operator"]. At a node with state x_0:
1. Linearize F over H steps around a nominal input (default ū = 0, time-varying linearization).
2. Eigendecompose CC^T = Σ λ_j v_j v_j^T.
3. For branch i, target terminal state z_H = ±sqrt(λ_{i//2}) v_{i//2} + (free response of the linear system from z_0). This gives 2n targets, plus and minus each eigenvector.
4. Reference input u_ref = clip(C^† z_H, U), the minimum-energy input to that target, clipped to limits (C^† is the pseudoinverse).
5. Track the linear reference with the nonlinear system using a time-varying LQR gain from the discrete algebraic Riccati equation, clipping inputs; roll out F for H steps.
Plain meaning: for a linear system the states reachable with unit control energy form an ellipsoid whose axes are the Gramian eigenvectors scaled by sqrt(eigenvalue); the 2n axis endpoints are a small set that covers the reachable set, and each becomes one child edge. The paper reads this as automatically generated options (temporally extended actions) [MCTS Discussion].

Tree policy with polynomial exploration [MCTS Alg 1, line 6]. Pick child i* = argmax_{i ∈ C(parent)} V(i, ℓ) + c_1 T(parent, ℓ)^{c_3} / T(i, ℓ)^{c_2}; unvisited children are chosen uniformly at random. Plain meaning: a UCT-style rule whose bonus is polynomial instead of logarithmic, which is what makes the gap-free analysis go through. Constants: the main text says c_1 = 1, c_2 = 0.5, c_3 = 1, but the supplement's analysis requires c_1 = 1, c_3 ∈ [1/4, 1/2), c_2 = 2c_3 and says experiments used c_3 near 1/2, c_2 near 1. The two are inconsistent; treat the supplement's range as the one the theorem needs.

Assumption 1 [MCTS Theoretical Results]. F is twice continuously differentiable, R is Lipschitz and depends only on the state, U is a box. Plain meaning: smooth deterministic dynamics and a smooth state reward, so linearization error and reachable-set distances translate into value error.

## Main results

| Result | Statement | Assumptions |
|---|---|---|
| Supp Lemma 1 | ||V*_1 − V*_2||_∞ ≤ (L_R + γ L_V)/(1 − γ) · max_x d_S(R_F(x, U_1), R_F(x, U_2)) for two MDPs differing only in action sets | Assumption 1 |
| Thm 1 (spectral expansion) | |V*(x_0) − V*_SETS(x_0)| ≤ (κ_0 + γ^H)/(1 − γ^H) · (κ_1 (1 + κ_2 Δt)^H + κ_3) | Assumption 1 |
| Thm 2 (MCTS convergence) | At any node i: |V*(i) − E[V(i, ℓ) | T(i, ℓ) = τ]| ≤ κ_4 / τ^{κ_5}, and P(|E[V(i,ℓ)] − V(i,ℓ)| ≥ z/τ^{κ_6} | T = τ) ≤ κ_7 / z^{κ_8}; the analysis gives κ_5 = κ_6 = 1/2 | Finite action set (the SETS tree), deterministic rewards, the polynomial exploration law |
| Thm 3 (SETS) | |V*(x_0) − E[V(x_0, ℓ)]| ≤ κ_4/sqrt(ℓ) + (κ_0 + γ^H)/(1 − γ^H) · (κ_1 (1 + κ_2 Δt)^H + κ_3) | Thms 1 and 2 via the triangle inequality |

Thm 1 proof route: value gap ≤ constant × Hausdorff distance between reachable sets (Lemma 1), then chain nonlinear reachable set → linear reachable set (Lemma 2) → finite set of linear modes (Lemma 3) → nonlinear tracked trajectories (Lemma 4, with tracking-controller stability in Lemma 14). Thm 2 is a backward induction over tree depth using non-stationary bandit arguments, following Shah, Xie, Xu (2020) "Non-asymptotic analysis of MCTS", but without needing the suboptimality gap and with depth-independent constants [MCTS Theoretical Results].

Trade-off in Thm 3: larger H means fewer decisions (depth ⌈K/H⌉), so κ_4 (which grows polynomially in the number of action sequences) shrinks and convergence is faster, but the steady-state error grows like (1 + κ_2 Δt)^H. Confirmed on a 2D double integrator (Fig. 7) and the glider (Fig. 5C, best H = 1000).

## Algorithm: SETS [MCTS Alg 1]

1. Root i_0 = Node(x_0).
2. For ℓ = 1, 2, ... (anytime):
   a. Path p = [i_0]. For d = 1 to ⌈K/H⌉: choose i* by the polynomial tree policy; if i* is unexpanded, run spectral expansion on it; stop if i* ∈ Ω; append i*.
   b. Backup: for each node on p, add the discounted sum of edge rewards from that depth onward (γ^{tH} weights) to Ṽ and increment T.
3. Return the maximum-valued trajectory found (allowed because in a deterministic setting the max is at least the average).
Run in receding-horizon (MPC) fashion: e.g. quadrotor plans 10 s every 5 s; tracked vehicle plans 1.6 s every 0.1 s.

Experiments: quadrotor with a learned DNN wind residual in Caltech's wind tunnel, a tracked vehicle doing driver assist under actuator degradation, two tethered spacecraft redirecting a third on an air-bearing floor, and a simulated 6-DOF glider using a thermal. Baseline study on the glider: spectral expansion with predictive sampling (SE-PS) is the only strong baseline, suggesting representation matters more than the search rule [MCTS Results].

## Relations to the other papers

- Access model. SETS is planning with a known deterministic model. It only simulates forward from states already in its tree, which starts at the current state, so it uses the model the way a local simulator is used in Power of Resets: reset to a previously reached state and roll forward [inference]. Resets itself says any computationally efficient local-simulator algorithm yields an efficient planner when the model is known [Resets §2.1 and App. A].
- RVFS vs SETS. Resets presents RVFS as "a principled, provable counterpart" to recursive search with value function approximation such as MCTS [Resets abstract], and says MCTS and AlphaZero "perform exploration via simple bandit-style heuristics, and are not explicitly designed to handle distribution shifts" [Resets §4.3]. SETS is an MCTS with a non-asymptotic guarantee, but for deterministic dynamics and with no function approximation, so it sidesteps the stochasticity and generalization issues RVFS targets [inference].
- Spectral expansion as a coverage device. Thm 1's proof bounds the Hausdorff distance between what any input sequence can reach and what 2n chosen trajectories reach. That is a covering argument on reachable sets, loosely parallel to coverability, where a single distribution covers what any policy can reach [inference, an analogy only; the quantities differ: geometric distance for deterministic systems vs density ratios for stochastic MDPs].
- Optimism. The polynomial bonus is optimism in the face of uncertainty at each tree node, the same principle as GOLF's global optimism but applied locally per node [inference].
- No statistical learning: the dynamics are known (or learned separately, e.g. the DNN wind model), so there is no sample-complexity notion comparable to C_cov or BE dimension. The "samples" are model rollouts, and the guarantee is computation vs accuracy.

## Connections

Planning with a known model (model-based RL, trajectory optimization): combines linearization and LQR from the control side with UCB-style bandit search (bandits) over a tree; a robotics-scale example of exact DP being replaced by sampling-based search.
