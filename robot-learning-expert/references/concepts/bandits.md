# Bandits and strategic exploration

Sources: Notes 3, Lattimore and Szepesvári *Bandit Algorithms*. Bandits supply the exploration ideas (optimism, posterior sampling) that model-based RL (Notes 4 §6), UCB-style RL theory and the reset papers reuse.

## Setting

k-armed stochastic bandit: arms A = [k], each with a fixed unknown reward distribution R_a with mean μ_a; μ⋆ = max_a μ_a. Each round t ∈ [n] the learner picks A_t, sees X_t ~ R_{A_t}. No state links rounds.

Regret Reg_n = n μ⋆ − E[Σ_t X_t] = Σ_a Δ_a E[T_a(n)], where Δ_a = μ⋆ − μ_a is arm a's gap (the notes print the sign backwards) and T_a(n) counts pulls. Goal: sublinear regret, and finite-sample rates (log n, √n, n^{2/3}, ...).

Baselines mentioned in class: explore-then-commit, ε-greedy.

## UCB (optimism in the face of uncertainty)

Treat an uncertain arm as if it were as good as it plausibly could be; giving up on the best arm after unlucky draws costs linear regret.
1. Confidence width from Hoeffding (bounded rewards in [0,1]): P(μ − μ̂ > z) ≤ exp(−2nz²) ≤ δ when z ≥ sqrt(log(1/δ)/(2n)).
2. Index UCB_a(t) = ∞ if T_a(t) = 0, else μ̂_a(t) + sqrt(2 log(1/δ) / T_a(t)) (the notes' constant; a valid, looser bound).
3. Play A_t = argmax_a UCB_a(t−1).
Smaller δ means more optimism, which is the conservative choice. Pick δ a bit below 1/n.

Regret sketch: a suboptimal arm is played only if the optimal arm is underestimated (rare by choice of δ) or the suboptimal arm is overestimated (its interval shrinks like 1/sqrt(T_a)). Results: Reg_n = O(Σ_{a: Δ_a > 0} log(n)/Δ_a) (gap-dependent; useless when gaps are tiny) and O(sqrt(k n log n)) (gap-free).

## Thompson sampling (posterior sampling)

Keep a belief over each arm's parameter; each round, sample a parameter per arm, act greedily on the sample, update by Bayes' rule.
- Bernoulli arms with a Beta(α, β) prior: reward 1 gives Beta(α+1, β), reward 0 gives Beta(α, β+1). Beta(1,1) is uniform. Mean α/(α+β); the posterior concentrates as α+β grows.
- Needs a conjugate prior (a family closed under the Bayes update) to stay tractable; Gaussian mean with known variance has a Gaussian conjugate, and so on.
- Computational edge: UCB solves max over θ in the confidence set and over actions (for shortest paths, a search over all plausible graphs); TS samples one graph and runs Dijkstra.
- Bayesian regret E_{problem~Q}[Reg_n] ≤ C sqrt(k n log n) when the prior is the true Q, matching UCB's gap-free rate. TS also makes priors easy to inject.

## Contextual and linear bandits

Context C_t each round, reward r(C_t, A_t) + noise, regret Σ_t [max_a r(C_t, a)] − Σ X_t (max inside the sum). With few contexts and no structure, run one bandit per context; it gets interesting only with shared structure.

Linear bandit: r(c, a) = ⟨θ⋆, φ(c, a)⟩ with known features. Ridge regression θ̂_t = V_t^{-1} Σ A_s X_s with V_t = λI + Σ A_s A_s'. Confidence ellipsoid {θ : ‖θ − θ̂_{t−1}‖²_{V_{t−1}} ≤ β_t}, axes along V_{t−1}'s eigenvectors with lengths proportional to 1/sqrt(eigenvalue), so they shrink as data accumulates in that direction (the notes say "reciprocal of the eigenvalues", which drops the square root). LinUCB: Reg_n = O(d sqrt(n) log(nL)), L bounding action norms.

## Concentration toolkit (App. A)

| Result | Statement | Needs |
|---|---|---|
| Markov | P(X ≥ t) ≤ E[X]/t | X ≥ 0; tight |
| Subgaussian (Def. 15) | E[e^{λX}] ≤ e^{σ²λ²/2} | tails no heavier than a Gaussian's |
| Sums (Thm 16, 17) | independent σ_1², σ_2² subgaussian sum is (σ_1²+σ_2²)-subgaussian; aX is a²σ²-subgaussian | |
| Hoeffding's lemma (Thm 18) | X ∈ [a, b] zero-mean is subgaussian with σ = (b−a)/2 | bounded |
| Chernoff (Thm 19) | P(X > t) ≤ exp(−t²/(2σ²)) | Markov on e^{sX}, optimize s = t/σ² |
| Hoeffding (Thm 20) | P(X̄ > t) ≤ exp(−n t²/(2σ²)) for i.i.d.-σ² case; n ≥ 2σ² log(1/δ)/t² samples for confidence 1−δ | independent subgaussian |

Unlike the CLT, these are finite-sample statements, which is what lets you compute how many samples or rollouts an algorithm needs (e.g. GORP's m, RVFS's test sizes).

## Where this shows up next

- Model-based RL with exploration: optimistic confidence sets over MDPs or posterior sampling over MDPs (Notes 4 §6; `model-based.md`).
- AJKS 2026 draft ch. 5 (bandits), ch. 6 (UCB-VI for tabular MDPs), ch. 7 (linear MDPs, Lin-UCBVI).
- Project papers: coverage-based exploration, GOLF, seed sampling in concurrent RL (`papers/`).
