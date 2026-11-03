# Q-learning: tabular, linear, deep

Q-learning approximates Q⋆ directly, never estimating P. Organize it as three rows of one table (Notes 7 §1): tabular (a table, count bonuses), linear MDP (weights on known features, covariance bonuses), general (neural network, no clean bonus). Each row answers the same three questions: how to represent Q, which (s, a) to query with a generative model, and what exploration bonus to use in the online protocol. Sources: Notes 5 to 7, Jin et al. 2018 and 2020, AJKS ch. 3 to 4, SB ch. 6, 9 to 11.

Symbols used throughout: S and A are the numbers of states and actions (or the sets themselves), H the episode horizon, γ the discount, K the number of episodes, T = KH the total number of steps, δ a failure probability ("with probability 1 − δ"), c or C an unspecified absolute constant.

## Why model-free at all (Notes 5 §1)

Memory: estimating P̂ costs O(S²A), a Q table O(SA), a deterministic policy O(S). For LQR, dynamics cost O(n² + nm) versus a gain matrix O(mn), and n ≥ m in most robots. Practical reasons the notes leave as an exercise: no model bias, and no planner needed at run time.

## 1. Tabular, generative model (Notes 5 §2)

The empirical Bellman operator replaces E over s' with N samples:
(T̂⋆_N Q)(s, a) = r(s, a) + (γ/N) Σ_i max_{a'} Q(s'_i, a'), s'_i ~ P(·|s, a).

- Iterating T̂⋆_N alone does not converge: it "chatters". Fig. 1 example: A goes to B or C with probability 1/2, B pays 1, C pays 0, both go to absorbing D. With N = 1, Q(A) jumps between γ and 0 forever. Even Q = Q⋆ is moved by one sampled backup.
- Fix: average with a step size α, Q ← Q + α(T̂⋆_N Q − Q). With N = 1 this is the familiar update Q(s,a) ← (1 − α)Q(s,a) + α[r + γ max_{a'} Q(s', a')]. N = 1 is the norm: bigger N averages transition noise but does not move information back in time faster.
- Step sizes must satisfy Robbins-Monro, Σ α_i = ∞ and Σ α_i² < ∞ (eq. 1). Typical α_i = 1/(i+1)^ω with ω ∈ (1/2, 1].
- Rates: ω = 1 decays too fast and gives exponential sample complexity (Even-Dar and Mansour 2003). ω ∈ (1/2, 1) is polynomial, best Õ(SA/((1−γ)⁵ε²)), a 1/(1−γ)² factor worse than model-based. Variance-reduced Q-learning with N > 1 matches model-based at Õ(SA/((1−γ)³ε²)) (Sidford et al. 2018, skipped in class).

## 2. Tabular, online protocol (Notes 5 §3)

Regret over n episodes: Reg_n = Σ_i [V⋆_1(s^i_1) − V^{π_i}_1(s^i_1)], policy fixed within an episode, start state possibly adversarial.

ε-greedy fails in the worst case (Fig. 2, a chain where only the action a_1 at every step reaches the one reward at s_{H+1}): with uniform tie-breaking the chance of H correct actions in a row is (1/A)^H, so order A^H episodes. The visit counts N_h(s, a) pile up near s_0; that pile-up is the signal "something is going wrong". Reproduce it with `mdp_tools.combination_lock`.

Optimism fixes it. Q-learning with UCB-Hoeffding (Jin et al. 2018, Notes 5 Alg. 2):
1. Initialize Q_h(s, a) = H (optimistic: unvisited pairs look at least as good as visited ones), N_h = 0.
2. Act greedily: a_h = argmax_{a'} Q_h(s_h, a').
3. Count t = N_h(s_h, a_h) + 1, bonus b_t = c sqrt(H³ ι / t) with ι = log(SAT/δ).
4. Update Q_h(s_h, a_h) ← (1 − α_t) Q_h + α_t [r_h + V_{h+1}(s_{h+1}) + b_t], with α_t = (H+1)/(H+t).
5. V_h(s_h) = min{H, max_{a'} Q_h(s_h, a')}.

Why α_t = (H+1)/(H+t) and not 1/t: the step size must depend on the per-pair count t because visits are uneven, and this choice keeps roughly the last 1/H fraction of the backups in the estimate so errors do not compound across the H layers (Jin et al., quoted in Notes 5 §3.2). Regret Õ(√(H⁴SAT)) for the Hoeffding bonus and Õ(√(H³SAT)) with a variance-aware (Bernstein) bonus; model-based UCB-VI gets Õ(√(H²SAT)) (Azar et al. 2017). In Jin et al., T = KH is the total number of steps, not episodes (arXiv 1807.03765 abstract; the notes say episodes). Takeaway the notes stress (Remark 3): count-based 1/sqrt(N) bonuses are "the right thing" for tabular RL, and the exact bonus and step-size forms came from theory, not trial and error.

## 3. Linear MDP (Notes 6)

Definition 1: known features φ: S×A → R^d with r(s,a) = ⟨φ(s,a), θ⋆⟩ and P(s'|s,a) = Σ_i φ(s,a)_i μ_i(s') for unknown (signed) measures μ_i. "Linear" means linear in features, not linear dynamics as in LQR. Tabular MDPs are linear MDPs with one-hot φ, d = SA.

Key property (Thm 3): for any f: S → [0, H], the backup r(s,a) + E_{s'}[f(s')] = ⟨φ(s,a), w⟩ with w = θ⋆ + F, F_i = ∫ f(s') μ_i(s') ds'. So the class of linear Q functions is closed under the Bellman operator (Bellman complete), and Q⋆_h is linear for every h (Cor. 4, backward induction). Finite A is needed: φ is arbitrary, so max over a continuous A has no structure.

Generative model: Least-Squares Value Iteration (LSVI, Alg. 1). Backward over h, regress the sampled backup onto φ:
θ_h = argmin_θ Σ_{(s,a,r,s')∈D_h} (r + max_{a'} θ_{h+1}ᵀφ(s', a') − θᵀφ(s, a))².
Which (s, a) to query: the data matrix AᵀA = Σ_{(s,a)∈D} φφᵀ defines an ellipsoid; noise in a direction with eigenvalue κ² gets amplified by 1/κ, so the data must "cover" every feature direction. Worked example in §4 (the notes write δ for κ): two samples φ_1 = (1,0), φ_2 = (0,κ) with targets y_i = ⟨θ, φ_i⟩ + ε_i, ε_i zero-mean sampling noise, give θ̂_2 = θ_2 + ε_2/κ. The D-optimal (equivalently G-optimal, Kiefer-Wolfowitz) design is a distribution ρ over (s, a) with support on at most d(d+1)/2 pairs; with Σ_ρ = Σ ρ(s,a)φφᵀ it guarantees ‖φ(s,a)‖²_{Σ_ρ⁻¹} ≤ d for every (s, a). Sampling ⌈ρ(s,a)N⌉ next states per support point gives an ε-optimal policy with probability 1 − δ using O(H⁷d² log(1/δ)/ε²) samples (AJKS, cited in Notes 6 §4).

Online: LSVI-UCB (Jin et al. 2020, Notes 6 Alg. 2). Each episode, backward over h: ridge-regress with Λ_h = λI + Σ_τ φ_τφ_τᵀ, define bonus b_h(s,a) = β sqrt(φ(s,a)ᵀΛ_h⁻¹φ(s,a)) and Q_h = min{θ_hᵀφ + b_h, H}; then act greedily forward. Eigen view: write Λ_h = U D Uᵀ with U orthogonal (eigenvectors) and D diagonal (eigenvalues, how much data lies along each eigenvector); then b = β‖D^{-1/2}Uᵀφ‖, largest along directions with little data. With one-hot features Λ is diagonal with entries λ + N_h(s,a), and as λ → 0 the bonus becomes β/sqrt(N_h(s,a)), the tabular bonus. Paper constants: ridge λ = 1, β = c·d·H·sqrt(ι), ι = log(2dT/p) with p the failure probability, regret O(√(d³H³Tι²)) with T = KH total steps (arXiv 1907.05388 Thm 3.1). Why β carries a factor d (= SA for one-hot features) where the tabular bonus has none: the analysis bounds the regression error uniformly over the whole class of value functions the algorithm might produce, and that class's covering number grows with d (Jin et al. 2020 §4, Lemma B.3); the tabular analysis avoids this union bound. Misspecification by ζ (in TV distance and reward) adds a term linear in T, at most O(ζdHT√ι) (Thm 3.2); the paper argues linear-in-T is unavoidable.

## 4. General function approximation (Notes 7)

Infinite horizon is the default here: learning a separate neural Q⋆_h per step is too costly.

Bellman completeness (Def. 1): F is complete if T⋆F ⊆ F. Neural classes give no such guarantee; theory assumes bounded Bellman error (for all f ∈ F some f' ∈ F has ‖f' − T⋆f‖_∞ ≤ ε, uniform in f) and pays poly(ε, H or 1/(1−γ)) extra suboptimality. Practice ignores the issue.

From the ideal backup to DQN, one approximation per step:
1. Ideal: min_θ max_{s,a} |r + γ E max f(s', ·) − f_θ(s,a)| (a min-max game, computationally hard).
2. Replace max over (s, a) and E over s' with a dataset D.
3. Replace worst-case (L∞) error with squared (L2) error, assuming roughly equal data per (s, a). Repeating this least-squares backup is Fitted Q-Iteration (FQI, eq. 1): T̃⋆_D(θ) = argmin_{θ'} Σ (r + γ max_{a'} f_θ(s', a') − f_{θ'}(s, a))².
4. Solve each regression approximately with N steps of minibatch SGD started from the target θ (Alg. 1). The frozen θ in the target is the target network. DQN fits this template with N = C = 10,000 SGD steps between target refreshes, one SGD step per 4 environment steps, and data growing online under ε-greedy (Alg. 2; DQN Ext. Data Tab. 1 in `papers/canon-algorithms.md`).

What makes D good: the discounted visitation distribution d^π_μ(s, a) = (1−γ) Σ_t γ^t P(s_t = s, a_t = a | π, s_0 ~ μ) (Def. 2). FQI's analysis needs concentrability: D sampled i.i.d. from ν with d^π_μ(s,a)/ν(s,a) ≤ C for every policy π. Strict: ν must cover everything any policy can reach.

Exploration in unstructured spaces (§3.2):
- Noisy TV problem: a corridor with reward at one end and a TV with a "new random image" button at the other. Novelty alone traps the agent at the TV; the bonus has to measure what is learnable, the way LSVI-UCB's covariance does.
- Pseudo-counts (Bellemare et al. 2016): from a density model ρ, ask for a pseudo-count N̂(s) and pseudo-total n̂ that behave like counts, ρ(s|s_{1:n}) = N̂/n̂ and ρ(s|s_{1:n}, s) = (N̂+1)/(n̂+1); solving gives N̂(s) = ρ(1 − ρ')/(ρ' − ρ), with ρ' the density after adding s. Needs ρ' > ρ.
- Random Network Distillation (Burda et al. 2019): fixed random network g, trained predictor ĝ_ψ, bonus w‖g(s) − ĝ_ψ(s)‖² with weight w > 0 (the notes call it λ), one SGD step on ψ per visit. Generalizes counts through continuity. Predicting a_t from (s_t, s_{t+1}) instead of s_{t+1} from (s_t, a_t) ignores uncontrollable noise like the TV.
- Use the simulator: benchmarks are simulators, so throwing away resets is a choice. Go-Explore (Ecoffet et al. 2019) and Power of Resets (Mhammedi et al. 2024, `papers/power-of-resets.md`). The notes call this under-researched and one of the most promising directions in empirical RL; see `resets-and-exploration.md`.

Action selection (§4): finite A: one network S → R^{|A|}. Continuous A: argmax_a f(s, a) is a nonconvex optimization; solve it by sampling (CEM-style, as in QT-Opt, Kalashnikov et al. 2018, slow) or learn an actor S → A, which leads to actor-critic (`actor-critic.md`).

Other tricks (§5): Rainbow (Hessel et al. 2018) combines many DQN extensions; distributional RL models the whole distribution of r + max Q(s', ·) with a categorical over value bins.

## Deep Q-learning in practice `[practice]`

The notes stop at the FQI view. What a robot learner adds, with sources in `papers/canon-algorithms.md` and `papers/fasttd3.md`:
- Double DQN target y = r + γ Q_target(s', argmax_{a'} Q_online(s', a')) cuts maximization bias (one network picks, the other evaluates).
- Bootstrap through time-limit truncations, not through true terminations: y = r + γ(1 − terminated) max Q(s', ·). Treating a time-out as terminal teaches the critic that the world ends at step H (`training/diagnostics.md`).
- n-step returns propagate reward faster and reduce bootstrap bias, at the cost of more variance and, off-policy, bias from the behavior policy's actions. FastTD3 stays stable with very large batches and many envs; PQN instead drops replay and target networks and relies on LayerNorm and λ-returns over vectorized envs (`papers/canon-algorithms.md`).
- Q magnitudes above max|r|/(1−γ) mean divergence (deadly triad: function approximation + bootstrapping + off-policy data, SB §11.3).

Errata for the Notes 5 to 7 derivations above (including where the notes differ from Jin et al.): `sources.md`.
