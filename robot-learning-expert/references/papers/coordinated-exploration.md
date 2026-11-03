# Coordinated Exploration in Concurrent Reinforcement Learning

Maria Dimakopoulou, Benjamin Van Roy (Stanford). ICML 2018 (venue from memory; the arXiv v1 PDF does not state it). arXiv:1802.01282 (v1, Feb 2018). The v1 is an 18-page version with no regret theorems; check the ICML version if you need more.

## The idea

When K agents explore the same unknown MDP in parallel and share data, efficient exploration needs three things: adapt to everyone's new data, commit to multi-step probing plans, and spread agents across different things to learn [CE §3]. Optimism (UCRL) is deterministic, so agents with the same data do the same thing (no diversity); Thompson sampling that resamples every step loses commitment. Seed sampling fixes a random seed per agent and maps it through the current posterior, which gives all three [CE §3.2].

## Notation

[CE §2]

- Single-episode, time-homogeneous MDP M = (S, A, R, P, ρ, H): finite states S, finite actions A, reward model R, transition model P, initial distribution ρ, horizon H (actions per agent).
- K agents. Agent k starts at s_{k,0} and acts at the arrival times t_{k,1} < ... < t_{k,H} of its own rate-1 Poisson process. At t_{k,m} it takes a_{k,m}, moves s_{k,m−1} → s_{k,m}, sees reward r_{k,m}.
- All agents share a prior over (P, R) and update a common posterior in real time. F_0 is the prior, F_t the posterior given history H_{t−1} (all agents' observations before time t).
- M_{k,m}: the MDP agent k "believes in" when choosing its m-th action; π_{k,m}: the optimal policy of M_{k,m}.
- Bayesian regret per agent: BayesRegret(K) = (1/K) Σ_k E[Σ_{m=1}^H (R* − r_{k,m})], with R* the optimal reward (here, the optimal per-episode total). The study object is how this falls as K grows.

## Core definitions

Property 1, Adaptivity [CE §3]. Use new data from all agents as it arrives. Plain meaning: do not ignore peers' discoveries mid-episode.

Property 2, Commitment. Keep carrying out a probing action sequence that spans several steps. Plain meaning: do not change your mind just because of fresh random noise.

Property 3, Diversity. Different agents pursue different learning opportunities. Plain meaning: divide and conquer.

Thompson resampling [CE §3.1.1]. At every action time, sample a fresh M_{k,m} ~ F_{t_{k,m}}. Diverse and adaptive, not committed.

Concurrent UCRL [CE §3.1.2]. At every action time, pick the most optimistic MDP in a confidence set built from the shared posterior. Adaptive and committed, not diverse: same data implies same choice.

Seed sampling [CE §3.2]. Let Z be a seed distribution and G_0 a deterministic map with G_0(z) ~ prior when z ~ Z. At time t the map G_t is built from G_0 and the shared history. Each agent draws z_k once at the start and uses M_{k,m} = G_{t_{k,m}}(z_k). Plain meaning: each agent's sample is always distributed as the current posterior, but the randomness that makes agent k "optimistic about chain 3" stays fixed for its whole episode.

## Algorithms

Generic loop (shared by all methods; only step 3 differs) [CE §3]:
1. Each agent k: if seed sampling, draw z_k ~ Z once.
2. At each action time t_{k,m} for agent k:
3. Form M_{k,m} (Thompson: fresh posterior sample; UCRL: optimistic MDP; seed: G_{t_{k,m}}(z_k)).
4. Solve for π_{k,m}, act a_{k,m} = π_{k,m}(s_{k,m−1}), observe (s_{k,m}, r_{k,m}).
5. Update the shared posterior with (s_{k,m−1}, a_{k,m}, s_{k,m}, r_{k,m}).

Three seed maps:

| Variant | Uncertainty | Seed | Map G_t(z) |
|---|---|---|---|
| Exponential-Dirichlet [CE §3.2.1] | Transitions, Dirichlet posterior with counts α_t(s,a,s') | i.i.d. Exp(1) sequences z_{s,a,s',i} for every (s,a,s') | p_{s,a}(s') = Σ_{i ≤ α_t(s,a,s')} z_{s,a,s',i} / Σ_{s̃} Σ_{i ≤ α_t(s,a,s̃)} z_{s,a,s̃,i} |
| Standard-Gaussian [CE §3.2.2] | Reward parameters θ ∈ R^{|S||A|}, Gaussian or lognormal posterior N(μ_t, Σ_t) | z ~ N(0, I) | μ_t + D_t z with D_t^T D_t = Σ_t (lognormal: exp of that) |
| Martingalean-Gaussian [CE §3.2.3] | Same | z = (θ̂_0, w), θ̂_0 ~ N(μ_0, Σ_0), w_j ~ N(0, σ²) i.i.d. (lognormal: N(−σ²/2, σ²)) | argmin_ρ (ρ − θ̂_0)^T Σ_0^{-1} (ρ − θ̂_0) + (1/σ²) Σ_j (o_j^T ρ − r_j − w_j)²; closed form (O^T O + σ² Σ_0^{-1})^{-1} (O^T (R + W) + σ² Σ_0^{-1} θ̂_0) |

The exponential-Dirichlet map works because a Gamma(a) variable is a sum of a Exp(1) variables and normalized Gammas are Dirichlet [CE §3.2.1]; this needs integer counts α_t [inference]. The martingalean map is a randomized least squares fit to perturbed data plus a prior sample, the same idea as randomized least-squares value iteration and ensemble sampling [inference, consistent with the citation of Lu and Van Roy 2017].

## Main results

There are no regret bounds in this version. The formal statements are:

- [CE Prop 1] Conditioned on the information before t_{k,m}, the martingalean-Gaussian sample θ̂_{k,m} is distributed as the posterior of θ (from Lu and Van Roy 2017, Lemma 4). So seed sampling keeps Property 1.
- [CE Prop 2] Under a measure where θ̂_{k,0} is fixed and θ ~ N(θ̂_{k,0}, 2Σ_0), the sequence θ̂_{k,m} is a martingale. This is the source of the name and the formal sense of commitment.

Simulations [CE §4]:

| Problem | Setup | Finding |
|---|---|---|
| Bipolar chain (Ex 2) | N = 100 vertices, H = 150, one end worth +N, other −N, 50/50 prior | Thompson resampling oscillates and per-agent regret does not fall with K; seed sampling and UCRL improve with K |
| Parallel chains (Ex 3) | C = 10 chains, H = 5, θ_c ~ N(0, 100 + c), reward noise N(θ_c, 1) | UCRL sends every agent to the highest-variance chain; it needs 100,000 agents to match the per-agent regret seed sampling and Thompson reach with 100 |
| Maximum reward path (Ex 1) | Erdős-Rényi graph, N = 100, p = 2 ln N / N, ln θ_e ~ N(0, 4), H = 10, 10,000 agents | Cumulative regret relative to seed sampling: UCRL +49.9%, Thompson resampling +191% |

## Relations to the other papers

- Access model. This is concurrent online RL: many episodes run at once and share a posterior, and nobody can reset to a chosen state. In Power of Resets, the learner can reset to any previously seen state; here an agent who has not started yet effectively begins from the start state with more information [inference]. Neither model contains the other.
- Optimism vs randomization. GOLF (bellman-eluder.md, coverage.md) and SimGolf (Power of Resets) are global-optimism methods; this paper's argument says that if you ran them concurrently with shared data, every agent would select the same optimistic f, so there would be no diversity [inference]. A seeded randomized version, such as each agent fitting its own perturbed regression (martingalean-Gaussian style) inside the GOLF confidence set, is a natural concurrent variant; nothing in these papers proves it works [inference].
- Diversity and coverability. Coverability counts how much total "reach mass" policies have (Σ_{(x,a)} sup_π d^π_h(x,a)) [Coverage Lemma 3]. When that sum is large, a team that spreads out covers more of it per episode than a team that copies one policy; this is the intuitive reason diversity should matter more on high-C_cov problems [inference, not a theorem in any paper here].
- Commitment and deep exploration. Commitment is what lets Thompson-style methods do multi-step (deep) probing; the Coverage and BE analyses get the same effect from global optimism over whole value functions, not per-step choices [inference].
- No function approximation and no complexity measure: this is tabular or linear-Gaussian Bayesian RL. The paper's closing remarks list "seeding schemes that operate with practical scalable RL algorithms" as open [CE §5].

## Connections

A multi-agent extension of the bandits material's Thompson sampling vs UCB comparison, applied to model-based RL with a Bayesian model (Dirichlet transitions, Gaussian rewards); relevant to parallel-simulation training (many robots or simulator instances sharing data).
