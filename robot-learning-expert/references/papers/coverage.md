# The Role of Coverage in Online Reinforcement Learning

Tengyang Xie, Dylan J. Foster, Yu Bai, Nan Jiang, Sham M. Kakade. ICLR 2023 (venue per the citation in Power of Resets; the arXiv v1 PDF does not state it). arXiv:2210.04157 (v1, Oct 2022).

## The idea

Offline RL works when the data distribution covers every policy's state-action distribution (concentrability). This paper asks what happens online if such a covering distribution merely exists and the learner does not know it. Answer: existence alone (coverability) makes online exploration sample-efficient, and the standard optimistic algorithm GOLF already exploits it, but the proof needs squared Bellman error, which existing complexity measures (Bellman rank, Bellman-Eluder dimension) do not capture well [Coverage §1.1].

## Notation

All symbols below are as in the paper [Coverage §2]. AJKS writes states as s and the data distribution as ν; this paper writes x and μ.

- Episodic MDP M = (X, A, P, R, H, x_1): state space X (possibly huge), action space A, horizon H, transitions P_h(· | x, a), rewards R_h(x, a) in [0,1], fixed start state x_1. Rewards are normalized so that the sum over an episode is in [0, 1].
- Policy π = (π_1, ..., π_H), with π_h : X → Δ(A) (Δ(·) means probability distributions over a set).
- J(π) = E^π[Σ_h r_h], the expected return. π* is the optimal policy, Q* its Q-function.
- Occupancy measure d^π_h(x, a) = P^π[x_h = x, a_h = a]: probability that π is at (x, a) at step h. This is a per-layer distribution, not the discounted occupancy of AJKS Ch. 4.
- Bellman operator [T_h f](x, a) = R_h(x, a) + E_{x' ~ P_h(x,a)}[max_a' f(x', a')].
- Value function class F = F_1 × ... × F_H with F_h ⊂ (X × A → [0,1]); f_{H+1} = 0. π_f is the greedy policy of f. Policy class Π = {π_f : f ∈ F} unless stated otherwise.
- Bellman residual of f at layer h: δ_h = f_h − T_h f_{h+1} (a "test function").
- Regret over T episodes: Reg = Σ_{t=1}^T [J(π*) − J(π^(t))].
- μ = (μ_1, ..., μ_H), μ_h ∈ Δ(X × A): a data distribution per layer.
- Õ hides polylog(T) factors.

## Core definitions

Concentrability [Coverage Def 1]. For a fixed data distribution μ and policy class Π,
C_conc(μ) = sup_{π ∈ Π, h ∈ [H]} || d^π_h / μ_h ||_∞.
Plain meaning: the worst-case ratio between where any policy goes and where the data is; small means the data sees everything any policy could visit. This is AJKS Assumption 4.1 in finite-horizon form.

Coverability [Coverage Def 2].
C_cov = inf_{μ_1, ..., μ_H ∈ Δ(X × A)} sup_{π ∈ Π, h ∈ [H]} || d^π_h / μ_h ||_∞.
Plain meaning: the best concentrability any data distribution could achieve, even one designed by someone who knows the MDP; it is a property of the MDP and Π, not of any data. Always C_cov ≤ |X||A|; it can be independent of |X| (block MDPs, low-rank MDPs) [Coverage §2.3].

Cumulative reachability [Coverage Lemma 3]. An equivalent form:
C_cov = max_h Σ_{(x,a)} sup_{π ∈ Π} d^π_h(x, a).
Plain meaning: for each (x, a), take the best probability any policy has of reaching it, then add these up; if policies overlap a lot, the sum is small. The minimizing μ_h is proportional to sup_π d^π_h(x, a) (proof of Lemma 3, App. A).

Completeness [Coverage Assumption 1]. T_h f_{h+1} ∈ F_h for all f_{h+1} ∈ F_{h+1}. Plain meaning: the class is closed under Bellman backups; this implies realizability (Q* ∈ F) and is strictly stronger.

Single-policy concentrability [Coverage Def 3]. C*_conc(μ) = || d^{π*}_h / μ_h ||_∞. Plain meaning: data only needs to cover the optimal policy. Its online analogue is trivial (take μ = d^{π*}, value 1), so it cannot be a useful online condition [Coverage §4].

Generalized concentrability and generalized coverability [Coverage Defs 4, 5]. C_conc(μ, F) is the smallest C with Σ_h E_{d^π_h}[(f_h − T_h f_{h+1})²] ≤ C Σ_h E_{μ_h}[(f_h − T_h f_{h+1})²] for all f ∈ F, π ∈ Π; C_cov(F) = inf_μ C_conc(μ, F). Plain meaning: coverage only measured on Bellman residuals of the class, summed over layers.

Bellman-Eluder dimension, Q-type [Coverage Def 6]. Restated from Jin et al.; see bellman-eluder.md. Uses average Bellman error E_{d^(i)}[δ].

Squared Bellman-Eluder dimension [Coverage Def 7]. Same as Def 6 but the "small on past distributions" condition uses Σ_{i<t} E_{d^(i)}[(δ^(t))²] instead of Σ_{i<t} (E_{d^(i)}[δ^(t)])². Plain meaning: a version of BE dimension that can see pointwise squared error.

Sequential Extrapolation Coefficient (SEC) [Coverage Defs 8, 9]. For test functions Ψ and distributions D,
SEC(Ψ, D, T) = sup over sequences ψ^(1..T) ⊆ Ψ and d^(1..T) ⊆ D of Σ_{t=1}^T E_{d^(t)}[ψ^(t)]² / (1 ∨ Σ_{i<t} E_{d^(i)}[(ψ^(t))²]).
For RL: SEC_RL(F, Π, T) = max_h SEC(F_h − T_h F_{h+1}, {d^π_h : π ∈ Π}, T). Plain meaning: how much on-policy average Bellman error can exceed what past in-sample squared error predicts, summed over time; the minimal ingredient an optimism proof needs.

## Main results

| Result | Bound | Assumptions |
|---|---|---|
| Thm 1 | Reg ≤ O(H · sqrt(C_cov · T · log(TH|F|/δ) · log T)) w.p. ≥ 1 − δ, GOLF with β = c·log(TH|F|/δ) | Completeness (Assumption 1), finite F, Π = greedy policies of F |
| Cor 2 (PAC) | J(π*) − J(π̄) ≤ O(H · sqrt(C_cov · log(TH|F|/δ) · log T / T)), π̄ = uniform mixture of π^(1..T) | Same |
| Prop 5 | Exogenous block MDP: C_cov ≤ |S||A| (S = endogenous latent states) | Decodable emissions |
| Cor 6 | Ex-BMDP regret O(H · sqrt(|S||A| T log(TH|F|/δ) log T)) | Completeness of F for that instance (a "luckiness" result) |
| Props 7, 8 | C_cov does not increase when adding rich (decodable) observations or exogenous noise | None beyond the constructions |
| Thm 9 (lower bound) | Generalized coverability C_cov(F) ≤ C with |A| = 2, completeness, log|F| ≤ H log(2|X|), yet any learner needs Ω(min{X, 2^{Ω(H)}, 2^{Ω(C)}}) trajectories for a 0.1-optimal policy w.p. 0.9 | Shows offline-sufficient coverage need not be online-sufficient |
| Prop 10 | An MDP with H = 2, |A| = 2, |Π| = |F| = d, completeness, C_cov = O(1), but dim_BE(F, Π, ε) = Ω(d) for ε ≤ 1/2 | Exponential gap: GOLF gets Õ(sqrt(T log d)) regret |
| Prop 18 (App C) | OLIVE (average Bellman error) needs Ω(d) trajectories on the same kind of instance | Average-error methods cannot exploit coverability |
| Prop 11 | Squared BE dimension can be ∝ 1/ε_T while C_cov = O(1); BE-style analysis cannot beat T^{2/3} regret on such classes | Scale insensitivity of BE |
| Thm 12 | Reg ≤ O(H · sqrt(SEC_RL(F, Π, T) · T · log(TH|F|/δ))) for GOLF | Completeness |
| Prop 13 | SEC_RL(F, Π, T) ≤ O(C_cov · log T) for any F | None |
| Prop 14 | SEC_RL(F, Π, T) ≤ O(inf_{ε>0} {ε²T + dim_BE(F, Π, ε)} · log T) | None |
| Thm 23 (App D) | Reward-free GOLF (explore with zero reward, then offline GOLF on target reward): suboptimality O(H · sqrt(SEC_RL,rf · log(TH|G|/δ) / T)) | Completeness plus reward-free completeness (Assumption 2) |

The constant c is an unspecified absolute constant in every statement.

## Algorithm: GOLF (as used here) [Coverage Alg 1]

Input F, confidence width β. Initialize F^(0) = F, datasets D_h = ∅.
For t = 1, ..., T:
1. Optimism: f^(t) = argmax_{f ∈ F^(t−1)} f_1(x_1, π_{f,1}(x_1)); play π^(t) = π_{f^(t)}.
2. Run one episode, add (x_h, a_h, r_h, x_{h+1}) to D_h for every h.
3. Confidence set: keep f if, for every h, L_h(f_h, f_{h+1}) − min_{f'_h ∈ F_h} L_h(f'_h, f_{h+1}) ≤ β, where L_h(f, f') = Σ_{D_h} (f(x,a) − r − max_a' f'(x', a'))².
Output (PAC only): uniform mixture of π^(1..T).

The "minus min" term is the debiasing that removes double-sampling bias in squared Bellman error; it is valid only because of completeness.

## Proof idea of Theorem 1 (the part Power of Resets reuses)

1. With high probability Q* ∈ F^(t) and Σ_{x,a} d̃^(t)_h(x,a) δ^(t)_h(x,a)² ≤ O(β), where d̃^(t)_h = Σ_{i<t} d^(i)_h is the unnormalized sum of past occupancies (Lemma 15, from Jin et al.).
2. Optimism gives Reg ≤ Σ_t Σ_h E_{d^(t)_h}[δ^(t)_h] (Lemma 16, the standard decomposition).
3. Burn-in: τ_h(x,a) = first t with d̃^(t)_h(x,a) ≥ C_cov μ*_h(x,a). Before τ, contribution is at most 2 C_cov in total, using |δ| ≤ 1.
4. After τ, Cauchy-Schwarz splits into extrapolation error (I) times in-sample squared error (II). (II) ≤ O(sqrt(βT)). (I) ≤ O(C_cov log T) by a per-state scalar elliptic potential lemma (Lemma 4) plus cumulative reachability (Lemma 3).
The non-negativity of squared error is what makes the change of measure work; average Bellman error can cancel and break it [Coverage §3.2, §5.1].

## Relations to the other papers

- Bellman-Eluder (bellman-eluder.md). GOLF is Jin et al.'s algorithm, here with G = F. Jin et al. bound GOLF by BE dimension; this paper shows BE dimension can be exponentially larger than coverability (Prop 10) and introduces SEC, which is bounded by both (Props 13, 14). So the right statement is "GOLF regret scales with SEC", and coverability and BE dimension are two ways to bound SEC.
- Power of Resets (Mhammedi, Foster, Rakhlin, NeurIPS 2024, arXiv:2404.15417; see power-of-resets.md). It takes coverability as its structural assumption [Resets Assumption 3.2] and cites this paper for C_cov ≤ SA in ExBMDPs. Its SimGolf algorithm is GOLF with one change: at each visited (x_h, a_h) it uses the local simulator to draw K fresh next states and rewards, and estimates T_h[g_{h+1}](x_h, a_h) by an empirical mean, so the confidence set no longer needs the completeness-based debiasing [Resets §3.1]. Result: only Q*-realizability plus coverability, at sample cost Õ(H^5 C_cov^2 log(|Q|/δ) / ε^4) [Resets Thm 3.1]. This directly answers the open question in [Coverage Thm 1 discussion] of whether completeness can be dropped, but only with local simulator access; Resets conjectures it cannot be dropped in plain online RL [Resets §3.2].
- Coverability vs pushforward coverability. Resets' RVFS needs C_push = max_h inf_{μ_h ∈ Δ(X)} sup_{(x_{h−1}, a_{h−1}, x_h)} P_{h−1}(x_h | x_{h−1}, a_{h−1}) / μ_h(x_h) [Resets Assumption 4.1]. Pushforward coverability bounds every one-step transition density instead of only policy occupancies, and C_cov ≤ C_push |A| with no converse in general [Resets §4.1]. This paper lists pushforward concentrability as an offline condition whose online role was open [Coverage §6.2]; Resets is one answer to that question under local simulator access [inference].
- Definition mismatch to watch: here Π = {π_f : f ∈ F}; Resets defines C_cov with sup over all policies Π_S, so for the same MDP the Resets coefficient is at least as large [inference: sup over a superset]. Resets also writes max_h inf_{μ_h}, while this paper writes inf_{μ_1..μ_H} sup_h; these are equal because each μ_h can be chosen separately [inference].
- Resets Remark 3.1 repeats this paper's point that average Bellman error (OLIVE style) is insufficient under coverability.
- Seed sampling, SETS, ICRL transformers: no direct technical link. Coverage-style conditions reappear loosely in the ICRL paper's excitation assumptions (see icrl-transformers.md) [inference].

## Connections

Bridges offline RL coverage (FQI and concentrability, AJKS Ch. 4) with strategic exploration (optimism, UCBVI in AJKS Ch. 6, Bellman rank in AJKS Ch. 8); GOLF is "optimistic fitted Q-iteration", so it is the theory counterpart of Q-learning with function approximation.
