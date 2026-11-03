# Bellman Eluder Dimension: New Rich Classes of RL Problems, and Sample-Efficient Algorithms

Chi Jin, Qinghua Liu, Sobhan Miryoosefi. NeurIPS 2021 (per citations in Coverage and Power of Resets). arXiv:2102.00815 (v4, Jul 2021).

## The idea

Two older tractability conditions, low Bellman rank and low Eluder dimension, each cover examples the other misses. The Bellman-Eluder (BE) dimension is the Eluder dimension of the class of Bellman residuals, measured under a family of distributions, and it contains both [BE §3]. The paper gives a new algorithm, GOLF (optimistic fitted Q-iteration), with sqrt(K) regret scaling with BE dimension, and re-analyzes OLIVE under BE dimension [BE §4, §5].

## Notation

[BE §2] AJKS uses the same letters except where noted.

- Episodic MDP (S, A, H, P, r): states S, actions A, horizon H, transitions P_h(s' | s, a), deterministic rewards r_h(s, a) ≥ 0 with Σ_h r_h ≤ 1 on every trajectory, fixed start state s_1.
- V^π_h, Q^π_h: value and Q-functions; V*, Q*: optimal ones. Bellman operator (T_h Q)(s, a) = r_h(s, a) + E_{s' ~ P_h(·|s,a)}[max_a' Q(s', a')].
- Regret over K episodes: Reg(K) = Σ_k [V*_1(s_1) − V^{π^k}_1(s_1)].
- F = F_1 × ... × F_H, F_h ⊆ (S × A → [0,1]): candidate Q-functions; f_{H+1} = 0. G = G_1 × ... × G_H: an auxiliary class used only in the regression constraint.
- π_f: greedy policy of f.
- N_F(ε): ε-covering number of F in the metric max_h ||f_h − g_h||_∞ (smallest number of ε-balls that cover F) [BE Def 3]. For finite classes it is at most |F|.
- (I − T_h)F = {f_h − T_h f_{h+1} : f ∈ F}: Bellman residuals at step h.
- E(f, π, h) = E_π[(f_h − T_h f_{h+1})(s_h, a_h)]: average Bellman error of f under roll-in policy π.

## Core definitions

Realizability [BE Assumption 1]. Q*_h ∈ F_h for all h. Plain meaning: the class contains the right answer.

Completeness [BE Assumption 2]. T_h F_{h+1} ⊆ F_h. Generalized completeness [BE Assumption 14]: T_h F_{h+1} ⊆ G_h. Plain meaning: Bellman backups of candidates stay inside a class you can regress onto.

ε-independence between points and Eluder dimension [BE Defs 4, 5], from Russo and Van Roy. Point z is ε-independent of x_1..x_n w.r.t. G if some g_1, g_2 ∈ G have sqrt(Σ_i (g_1(x_i) − g_2(x_i))²) ≤ ε but g_1(z) − g_2(z) > ε. dim_E(G, ε) is the length of the longest sequence where each point is ε'-independent of its predecessors for some ε' ≥ ε. Plain meaning: how many times can you be surprised by a new point after agreeing on all earlier ones; a nonlinear, approximate version of linear dimension.

ε-independence between distributions and distributional Eluder (DE) dimension [BE Defs 6, 7]. Distribution ν is ε-independent of μ_1..μ_n w.r.t. G if some g ∈ G has sqrt(Σ_i (E_{μ_i}[g])²) ≤ ε but |E_ν[g]| > ε. dim_DE(G, Π, ε) is the longest such sequence drawn from the distribution family Π. Plain meaning: the same surprise count, but you only observe expectations under distributions, which is what RL data gives you.

Bellman-Eluder dimension [BE Def 8].
dim_BE(F, Π, ε) = max_h dim_DE((I − T_h)F, Π_h, ε).
Two families are used: D_F (state-action distributions at step h reachable by greedy policies π_f, f ∈ F) and D_Δ (point masses on single (s, a)). A problem has low BE dimension if the minimum over these two choices is small. Plain meaning: how many genuinely new roll-in distributions can reveal a large average Bellman error that earlier data could not predict.

Bellman rank, Q-type [BE Def 10]. Smallest d such that E(f, π_{f'}, h) = <φ_h(f), ψ_h(f')> for maps φ_h, ψ_h : F → R^d with ||φ_h(f)|| · ||ψ_h(f')|| ≤ ζ. Plain meaning: the matrix of average Bellman errors (rows: candidate f, columns: roll-in policy) has rank d. The original Jiang et al. 2017 version (AJKS Ch. 8) is V-type [BE Def 19]: the action at step h comes from π_f, not the roll-in policy. V-type BE dimension [BE Def 20] uses state-wise residuals s ↦ (f_h − T_h f_{h+1})(s, π_{f,h}(s)).

## Containments

| Result | Statement |
|---|---|
| Prop 11 | Q-type Bellman rank d (normalizer ζ) implies dim_BE(F, D_F, ε) ≤ O(1 + d log(1 + ζ/ε)) |
| Prop 12 | Under completeness, dim_BE(F, D_Δ, ε) ≤ max_h dim_E(F_h, ε) |
| Prop 13 | For any m there is an MDP and F with dim_BE ≤ 5 under both D_F and D_Δ, while both min_h dim_E(F_h, ε) and Bellman rank are ≥ m (strict containment) |
| Prop 21 | V-type analogue of Prop 11 |
| App B | Tabular, linear MDPs, generalized linear, kernel MDPs (Prop 31), kernel reactive POMDPs (Prop 38, V-type) have low BE dimension |

Not covered: model-based examples such as low witness rank, which Bilinear Classes (Du et al. 2021) handles [BE §1.1].

## Main results

| Result | Bound | Assumptions |
|---|---|---|
| Thm 15 (GOLF regret) | For all k ≤ K, Reg(k) ≤ O(H · sqrt(d · k · β)) w.p. ≥ 1 − δ, with β = c · log[N_{F∪G}(1/K) · KH/δ] and d = min_{Π ∈ {D_Δ, D_F}} dim_BE(F, Π, 1/sqrt(K)) | Realizability + generalized completeness |
| Cor 16 (GOLF sample complexity) | Output policy is O(ε)-optimal w.p. ≥ 1/2 once K ≥ Ω((H² d / ε²) · log(N_{F∪G}(ε²/(dH²)) · Hd/ε)), d = dim_BE(F, Π, ε/H); informally Õ(H² d log N / ε²) | Realizability + completeness |
| Thm 18 (OLIVE) | O(ε)-optimal policy w.p. ≥ 1 − δ within O(H³ d² log[N_F(ζ_elim/8)] · ι / ε²) episodes, d = dim_BE(F, D_F, ε/H), ι = c log(Hd/δ) | Realizability only; only the D_F family; no sublinear regret |
| Thm 22 (V-type GOLF) | O(ε)-optimal w.p. ≥ 0.99 once K ≥ Ω((H² d |A| / ε²) · log(...)), d = V-type BE dimension | Realizability + generalized completeness |
| Thm 23 (V-type OLIVE) | O(H³ d² |A| log|F| · ι / ε²) episodes, finite F | Realizability; matches Jiang et al. 2017 |

Specializations claimed in [BE §4.1]: linear class of dimension d_lin gives Õ(H d_lin sqrt(K)) regret, matching Zanette et al. 2020 up to logs; Eluder-dimension regret improves from linear to sqrt in d_E; Bellman-rank sample complexity improves from quadratic to linear in d.

## Algorithms

GOLF [BE Alg 1]: "Global Optimism based on Local Fitting".
1. Initialize D_h = ∅, B^0 = F.
2. Each episode k: f^k = argmax_{f ∈ B^{k−1}} f(s_1, π_f(s_1)); play π^k = π_{f^k} for one episode.
3. Add (s_h, a_h, r_h, s_{h+1}) to D_h.
4. B^k = {f ∈ F : L_{D_h}(f_h, f_{h+1}) ≤ inf_{g ∈ G_h} L_{D_h}(g, f_{h+1}) + β for all h}, with L_{D_h}(ξ, ζ) = Σ_{D_h} [ξ(s,a) − r − max_a' ζ(s', a')]².
5. Output a uniformly random π^k.
FQI would set f_h = argmin L_{D_h}(·, f_{h+1}); GOLF keeps every near-minimizer and then picks the most optimistic one. The optimistic step is not computationally efficient in general [BE §4].

Proof in three steps [BE §4.2]: (1) Q* ∈ B^k w.h.p., so optimism bounds regret by Σ_h Σ_k E_{π^k}[(f^k_h − T f^k_{h+1})] (Jiang et al. Lemma 1). (2) The confidence set gives Σ_{i<k} (E_{π^i}[f^k_h − T f^k_{h+1}])² ≤ O(β). (3) Lemma 17, a pigeonhole lemma for DE dimension, converts (2) into a bound on (1). Step (2) as stated in [BE Eq. (5)] is about the squared average error (E[δ])². The confidence set actually controls the stronger average squared error E[δ²] (the Coverage paper quotes this as [Coverage Lemma 15], citing [BE Lemmas 39, 40]), and (E[δ])² ≤ E[δ²] by Jensen. The BE analysis throws away that extra strength; the Coverage paper's coverability proof keeps it.

OLIVE [BE Alg 2], from Jiang et al. 2017: pick optimistic f^k; run π^k for n_act episodes; if Σ_h Ê(f^k, π^k, h) ≤ H ζ_act, output π^k; else pick a step t with Ê ≥ ζ_act, collect n_elim more episodes, and eliminate every f with Ê(f, π^k, t) > ζ_elim. Each elimination step at layer h adds a ζ_act-independent distribution, so at most d eliminations per layer [BE §5.2]. Optimistic planning in OLIVE is NP-hard in the worst case (Dann et al. 2018, cited in [BE §5]).

## Relations to the other papers

- Coverage (coverage.md): reuses GOLF unchanged and shows coverability is not captured by BE dimension: C_cov = O(1) but dim_BE = Ω(d) [Coverage Prop 10]; even squared BE dimension gives at best T^{2/3} regret on some low-coverability classes [Coverage Prop 11]. Its SEC upper bounds both BE dimension and coverability [Coverage Props 13, 14].
- Power of Resets: SimGolf is GOLF plus local-simulator Bellman backup estimates; the "minus inf over G_h" term of GOLF (which needs generalized completeness) is replaced by direct estimation of T_h[g_{h+1}] at visited states [Resets §3.1]. So the representation requirement drops from completeness (GOLF here) to Q*-realizability, while the structural requirement changes from BE dimension to coverability. OLIVE also needs only realizability, but Coverage shows average-error methods like OLIVE fail under coverability [Coverage Prop 18], which is why SimGolf keeps GOLF's squared-error confidence sets [Resets Remark 3.1].
- Resets notes BE dimension and Bellman rank can be arbitrarily large for Exogenous Block MDPs [Resets §3], citing Coverage.
- SETS (mcts-spectral.md) and seed sampling (coordinated-exploration.md) use optimism and posterior sampling respectively, but have no function-class complexity measure. Concurrent UCRL in the seed sampling paper is the deterministic-optimism analogue; that paper argues optimism alone gives no diversity across parallel agents [inference: the same would apply to running GOLF independently on parallel agents with shared data, since every agent would compute the same argmax].

## Connections

Generalizes the Bellman-rank chapter (AJKS Ch. 8) and linear MDPs (AJKS Ch. 7); GOLF is optimistic fitted Q-iteration, so it sits between the Q-learning material and the strategic-exploration (UCB) material from bandits.
