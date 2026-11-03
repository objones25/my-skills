# Theory map: coverage, resets, and exploration across the reset and exploration papers

Files: coverage.md (Xie et al., ICLR 2023), bellman-eluder.md (Jin et al., NeurIPS 2021), coordinated-exploration.md (Dimakopoulou and Van Roy, 2018), mcts-spectral.md (Riviere et al., Science Robotics 2024), icrl-transformers.md (Liang and Lai, 2026 preprint), plus the project's anchor paper, Power of Resets (Mhammedi, Foster, Rakhlin, NeurIPS 2024, arXiv:2404.15417; details in power-of-resets.md).

## Access models

How the learner gets data, from least to most powerful (AJKS §1.4 for episodic and generative; [Resets §2.1, App. A] for local simulators):

1. Online episodic: run a policy from the start state (or start distribution), see the trajectory, repeat.
2. Concurrent online: K agents run episodes at the same time and share data in real time. Same per-agent power as (1), but parallel.
3. Local simulator (RLLS, "resets"): online episodic, plus you may reset to any layer h and any state x_h you have already observed and continue from there.
4. Generative model (global simulator): query any (s, a) you like and get a sampled next state and reward.
5. Known model (planning): the dynamics are given; the only cost is computation.

## Comparison table

| Paper and algorithm | Access | Structural measure | Representation assumption | Guarantee (main form) |
|---|---|---|---|---|
| BE dimension: GOLF [BE Thm 15, Cor 16] | Online episodic | BE dimension d (contains Bellman rank, Eluder dimension) | Realizability + generalized completeness | Reg ≤ O(H sqrt(d K β)); Õ(H² d log N_{F∪G} / ε²) episodes |
| BE dimension: OLIVE [BE Thm 18] | Online episodic | BE dimension, D_F family only | Realizability only | O(H³ d² log N_F · ι / ε²) episodes, no regret bound |
| Coverage: GOLF [Coverage Thm 1, Thm 12] | Online episodic | Coverability C_cov; more generally SEC | Completeness | Reg ≤ O(H sqrt(C_cov T log(TH|F|/δ) log T)) |
| Resets: SimGolf [Resets Thm 3.1] | Local simulator | Coverability C_cov | Q*-realizability only | Õ(H^5 C_cov² log(|Q|/δ) / ε^4) samples; computationally inefficient |
| Resets: RVFS [Resets Thm 4.1] | Local simulator | Pushforward coverability C_push | (V*, π*)-realizability + gap Δ, or V^π and π-realizability | Õ(C_push^8 H^23 A ε^{−13}) samples; reduces to regression (efficient) |
| Seed sampling [CE §3.2, §4] | Concurrent online | None (Bayesian prior over a tabular or Gaussian model) | Correct prior | No bound; Props 1, 2 (posterior-correct, martingale); simulations |
| SETS [MCTS Thms 1 to 3] | Known deterministic model, forward rollouts from tree nodes | Gramian spectrum: branching ≤ 2n, depth ⌈K/H⌉ | Smooth dynamics, Lipschitz state reward | Value error ≤ κ_4/sqrt(ℓ) + discretization error in H, Δt |
| ICRL transformer [ICRL Thms 3.1 to 3.3] | Single stream of on-policy windows, many training MDPs | None for exploration; excitation conditions for training | Linear features | Exact SARSA and actor-critic implementation; local exponential convergence of training |

## How the measures relate

Bellman rank and Eluder dimension are both contained in BE dimension [BE Props 11, 12, 13]. BE dimension and coverability are incomparable in the useful direction: there are instances with C_cov = O(1) and BE dimension Ω(d) [Coverage Prop 10], and SEC upper bounds both, so GOLF's regret is really a SEC bound [Coverage Thm 12, Props 13, 14]. Pushforward coverability is stronger than coverability: C_cov ≤ C_push · |A| and not conversely [Resets §4.1]. Examples: block MDPs have C_cov ≤ |S||A| and C_push ≤ |S|; low-rank MDPs have C_cov ≤ d|A| and C_push ≤ d; exogenous block MDPs have C_cov ≤ |S||A| while BE dimension and Bellman rank can be arbitrarily large [Coverage Prop 5; Resets §3, §4.1].

## The thread through reset-based exploration

GOLF is the common algorithm. Jin et al. introduce it and analyze it with BE dimension. Xie et al. reanalyze the same algorithm with coverability and show that the key step is a change of measure that needs squared Bellman error, which is why average-error methods like OLIVE fail under coverability [Coverage §3.2, Prop 18]. GOLF's squared-error confidence sets need completeness to remove double-sampling bias. Power of Resets removes that need with resets: at each visited (x_h, a_h) it draws fresh next states from the local simulator and estimates the Bellman backup directly (SimGolf), so coverability plus Q*-realizability suffices, and it conjectures this is impossible without resets [Resets §3]. RVFS trades the weaker coverability assumption for pushforward coverability to get computational efficiency, and is framed as a provable MCTS or AlphaZero [Resets §4.3].

The other three papers attach at the edges. SETS is a concrete MCTS with guarantees for deterministic continuous systems; it rolls out only from states already in its tree, which is local-simulator style use of a known model [inference], and it covers reachable sets geometrically instead of covering occupancy measures [inference]. Seed sampling is the randomized alternative to optimism when many agents explore at once: deterministic optimism (GOLF, SimGolf, UCRL) gives every agent the same plan, so a parallel version of these algorithms would need per-agent randomness [inference]. The ICRL transformer paper says a single attention layer can compute a SARSA or actor-critic step from a window of data, which concerns how an update is computed, not which data to collect; it has no exploration or coverage guarantee.

Tags: everything marked [inference] is the reference author's synthesis, not a claim in the papers.
