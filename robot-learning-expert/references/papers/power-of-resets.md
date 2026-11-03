# The Power of Resets in Online Reinforcement Learning (Mhammedi, Foster, Rakhlin)

Source: arXiv:2404.15417v2 (26 Apr 2024), 69 pages. Published at NeurIPS 2024 (Advances in NeurIPS 37, pp. 12334 to 12407, per the citation in the UGTS paper). Tags like [PoR §4.2], [PoR Alg. 3, l. 14], [PoR Thm 4.1], [PoR Lem. H.1] point to the arXiv v2 numbering. [inference] marks statements that are mine, not the paper's.

The paper has no experiments. Every result is a theorem, and the authors list empirical evaluation of RVFS-inspired exploration as future work [PoR §5].

## 0. Summary

- With a local simulator (reset to any previously seen state), MDPs with low coverability are learnable with only Q⋆-realizability, via the computationally inefficient SimGolf [PoR Thm 3.1]. Consequence: general Exogenous Block MDPs are learnable [PoR Cor 3.1].
- RVFS (Recursive Value Function Search) is the computationally efficient algorithm. It needs a stronger structural condition (pushforward coverability) plus either V⋆-realizability with a gap, or V^π-realizability for all π [PoR Thm 4.1].
- RVFS builds per-layer core sets of state-action pairs, fits value functions by Monte Carlo regression on next states drawn from those pairs, and adds a new pair (then recurses) whenever a confidence-set test says the current value estimate cannot be trusted at a newly reached state [PoR §4.2].
- RVFS^exo adds randomized rounding of Bellman backups so RVFS works on weakly correlated ExBMDPs without a gap [PoR §4.4, Thm 4.2].
- The policies RVFS computes need the simulator at decision time; RVFS.bc distills them into an executable policy with behavior cloning [PoR Alg. 4, App. L].

## 1. Setting and notation

### 1.1 MDP notation [PoR §2, §2.2]

- MDP M = (X, A, T, R, H). X: state space, possibly huge or infinite. A: finite action set, A := |A|. H: horizon (number of layers, indexed h ∈ [H] = {1,…,H}).
- R_h : X × A → [0,1] is the reward at layer h. The algorithms sample r_h ~ R_h(x,a), so R_h is treated as a distribution over [0,1].
- T_h : X × A → Δ(X) is the transition kernel at layer h (Δ(X) = distributions over X). T_0(· | ∅) is the initial state distribution. The MDP is layer-indexed (non-stationary).
- Policy π = {π_h : X → Δ(A)}; Π_S is the set of all such (stochastic, Markov) policies. x_{H+1} is a terminal state with zero reward.
- J(π) := E^π[Σ_{h=1}^H r_h], expected return.
- V_h^π(x) := E^π[Σ_{h'=h}^H r_{h'} | x_h = x]; Q_h^π(x,a) := E^π[Σ_{h'=h}^H r_{h'} | x_h = x, a_h = a]. Values lie in [0,H].
- π⋆: optimal deterministic policy; Q⋆ := Q^{π⋆}, V⋆ := V^{π⋆}.
- Occupancy measure d_h^π(x,a) := P^π[x_h = x, a_h = a].
- Bellman backup for a Q-type function g : X × A → R: T_h[g](x,a) := E[r_h + max_{a'} g(x_{h+1}, a') | x_h = x, a_h = a].
- Bellman backup for a V-type function f : X → R: P_h[f](x,a) := E[r_h + f(x_{h+1}) | x_h = x, a_h = a]. RVFS uses only P_h.
- argmax ties broken by smallest action index [PoR §2.2].
- π_unif: uniform random policy. Õ(·) hides polylog factors.

### 1.2 The local simulator protocol (RLLS) [PoR §2.1]

Standard online RL: in each episode τ ∈ [N_episodes] the learner picks a policy π^(τ), runs it from x_1 ~ T_0(·|∅), and observes the trajectory {(x_h, a_h, r_h)}_{h=1}^H. At the end it outputs π̂ and is judged by E[J(π⋆) − J(π̂)].

RL with local simulator access (RLLS, also called local planning) changes one thing: at the start of each episode, instead of drawing x_1 ~ T_0, the learner may reset the MDP to any layer h ∈ [H] and any state x_h it has previously encountered, then continue the episode from there [PoR §2.1]. Goal: E[J(π⋆) − J(π̂)] ≤ ε with N_episodes = poly(C, 1/ε) for a problem parameter C.

What the learner may do:
- Reset to (h, x_h) for any x_h it has already observed at layer h, then take any action and see r_h ~ R_h(x_h, a), x_{h+1} ~ T_h(·|x_h,a), and continue.
- Repeat this as often as it likes; each reset starts a (partial) episode that counts toward sample complexity.

What it may not do:
- Query states it has never observed. Querying arbitrary (x,a) is a "global simulator", which the paper distinguishes from local access [PoR App. A]. (The usual name for that is a generative model [inference: terminology, the paper says "global simulator"].)
- Use the simulator at test time if the goal is an executable policy (see 1.4) [PoR §2.1].
- Read the transition probabilities. It only gets samples.

| Protocol | Start states available | Can query arbitrary (x,a)? | Source |
|---|---|---|---|
| Episodic online RL | x_1 ~ T_0 only | No | [PoR §2.1] |
| Local simulator (RLLS) | any previously observed (h, x_h) | No | [PoR §2.1] |
| Global simulator (generative model) | any (x,a) | Yes | [PoR App. A] |

Planning view: if a model is known or learned, any RLLS algorithm becomes a planning algorithm by simulating the environment with the model; a computationally efficient RLLS algorithm gives an efficient planner [PoR §2.1, App. A].

### 1.3 Executable versus non-executable policies [PoR §2.1, Def. 2.1]

A policy π is non-executable with sample complexity n if computing π(x) needs n local simulator queries [PoR Def. 2.1]. The greedy policy RVFS computes, π̂_h(x) ∈ argmax_a P̂_h[V̂_{h+1}](x,a), is non-executable because P̂ samples next states from x [PoR §4.2]. The paper's main results target executable policies; RVFS.bc and RVFS^exo.bc convert via behavior cloning [PoR §4.2, Alg. 4, Alg. 7].

### 1.4 Sample complexity accounting

Bounds count episodes, where every reset-and-sample (including each of the K or N_sim next-state draws inside a backup estimate) counts. SimGolf calls the simulator KH times per iteration [PoR App. D]. For RVFS, every action of π̂ costs N_sim simulator draws per candidate action, and these are counted [PoR App. H.6].

## 2. Assumptions, defined

Function classes are assumed finite; bounds scale with log|Q|, log|V|, log|Π| and extend to infinite classes by standard covering arguments [PoR §3, Lem. 3.1 footnote 4].

- Assumption 3.1, Q⋆-realizability: a class Q ⊂ (X × A × [H] → [0,H]) with Q⋆_h ∈ Q_h for all h [PoR §3].
- Assumption 3.2, coverability: C_cov := max_h inf_{μ_h ∈ Δ(X×A)} sup_{π ∈ Π_S} ‖d_h^π / μ_h‖_∞. Intuition: one distribution μ_h covers every policy's occupancy up to factor C_cov. Block MDPs and ExBMDPs have C_cov ≤ |S||A| (S = latent state set); Low-Rank MDPs have C_cov ≤ d|A| [PoR §3]. The algorithm does not need μ_h [PoR §3].
- Assumption 3.3, decoder realizability (ExBMDP only): a class Φ of maps X → S with ϕ⋆ ∈ Φ [PoR §3.3].
- Assumption 4.1, pushforward coverability: C_push := max_h inf_{μ_h ∈ Δ(X)} sup_{(x_{h−1}, a_{h−1}, x_h)} T_{h−1}(x_h | x_{h−1}, a_{h−1}) / μ_h(x_h). Intuition: one distribution μ_h covers every one-step next-state distribution from every single state-action pair. Block MDPs: C_push ≤ |S|; Low-Rank MDPs: C_push ≤ d; weakly correlated ExBMDPs (Assumption 4.7): C_push ≤ C_exo·S·A [PoR §4.1, Lem. J.3]. Always C_cov ≤ C_push·|A|; the converse fails in general [PoR §4.1]. In tree-search language the paper calls C_push a stochastic analogue of branching factor [PoR §4.3, footnote 7].
- Assumption 4.2, V⋆-realizability: V⋆_h ∈ V_h for all h, where V ⊂ (X × [H] → [0,H]) [PoR §4.1].
- Assumption 4.3, π⋆-realizability: a policy class Π ⊂ Π_S with π⋆ ∈ Π [PoR §4.1]. Needed only to output an executable policy.
- Assumption 4.4, Δ-gap: π⋆_h(x) is unique and Q⋆_h(x, π⋆_h(x)) > Q⋆_h(x,a) + Δ for all h, x, a ≠ π⋆_h(x) [PoR §4.1].
- Assumption 4.5, V^π-realizability: V_h^π ∈ V_h for all π ∈ Π_S and h [PoR §4.1]. Remark 4.1: only near-optimal π are needed; formalized as Assumption H.1 (relaxed V^π-realizability: V_h^π ∈ V for all π in the benchmark class Π_{ε_real}) [PoR §4.1, App. H.1].
- Assumption 4.6, π-realizability (policy completeness): for every π ∈ Π_S, x ↦ argmax_a P_h[V^π_{h+1}](x,a) is in Π [PoR §4.1]. Assumptions 4.5 and 4.6 both follow from Q^π-realizability for all π [PoR §4.1].
- Assumption 4.7, weak correlation (ExBMDP): there is C_exo ≥ 1 with P[ξ_h = ξ, ξ_{h+1} = ξ'] ≤ C_exo · P[ξ_h = ξ] · P[ξ_{h+1} = ξ'] for all h, ξ, ξ' [PoR §4.4]. The exogenous law does not depend on the policy [PoR §4.4, footnote 8].

Note on "linear" assumptions: none of the main theorems assume linear Q⋆ or linear V⋆. The paper only remarks that RVFS can be modified to recover known linear results (linear Q⋆ with gap, Li et al. 2021; Q^π-realizability, Yin et al. 2022) and leaves that for future work [PoR §4.3]. Linear-V⋆ and linear-Q⋆ local-simulator results are discussed only as related work [PoR App. A].

Note on SimGolf versus RVFS assumptions: V⋆-realizability and π⋆-realizability are each implied by Q⋆-realizability, but Setup I adds the gap, so Setup I and Assumption 3.1 are incomparable [PoR §4.1].

## 3. Results and the assumptions each uses

| Result | Algorithm | Assumptions | Guarantee | Sample complexity |
|---|---|---|---|---|
| Thm 3.1 | SimGolf (Alg. 1) | 3.1 (Q⋆-real.) + 3.2 (coverability) | J(π⋆) − E[J(π̂)] ≤ ε w.p. 1−δ | Õ(H⁵ C_cov² log(\|Q\|/δ) / ε⁴) |
| Cor 3.1 | SimGolf on ExBMDP | 3.3 (decoder real.), Q built from Φ as in Lem. 3.1 | J(π⋆) − J(π̂) ≤ ε w.p. 1−δ | Õ(H⁵ S³ A³ log\|Φ\| / ε⁴) |
| Thm 4.1, Setup I | RVFS.bc (Alg. 4) | 4.1 + 4.2 + 4.3 + 4.4, and ε ≤ 6HΔ | J(π⋆) − J(π̂) ≤ 2ε w.p. 1−δ | Õ(C_push⁸ H²³ A ε⁻¹³) |
| Thm 4.1, Setup II | RVFS.bc (Alg. 4) | 4.1 + 4.5 + 4.6 | same | same |
| Thm H.1 | RVFS_0 alone (non-executable) | 4.1 + H.1 with ε_real = 4ε | E^π̂[D_tv(π̂_h(x_h), π̃_h(x_h))] ≤ ε/(4H³C_push) for every h, w.p. 1−5δ | Õ(C_push⁸ H¹⁰ A ε⁻¹³) episodes |
| Thm H.2 | RVFS.bc | 4.1 + H.1 with ε_real = 1 + 4.3 | J(π⋆) − J(π̂) ≤ ε | Õ(C_push⁸ H²³ A ε⁻¹³) |
| Thm 4.2 | RVFS^exo.bc (Alg. 7) | 3.3 + 4.7 | J(π⋆) − J(π̂) ≤ ε w.p. 1−δ | Õ(C_exo⁸ S⁸ H³⁶ A⁹ ε⁻²⁶) |
| Lem. J.7 | RVFS^exo_0 alone, given good seeds ζ | 3.3 + 4.7 + ζ ∈ E_rand | P^π̂[π̂_h(x_h) ≠ π̄_h(x_h)] ≤ ε²/(4H³SAC_exo) | Õ(C_exo⁸ S⁸ H¹⁰ A⁹ ε⁻²⁶) |

Sources: [PoR Thm 3.1, Cor 3.1, Thm 4.1, Thm H.1, Thm H.2, Thm 4.2, Lem. J.7]. ExBMDPs also satisfy Setup I of Thm 4.1 when weakly correlated and gapped, since Lem. 3.1 gives V⋆-realizability and Lem. J.3 gives C_push ≤ C_exo·SA [PoR §4.4].

The paper conjectures that Q⋆-realizability plus coverability is not enough in ordinary online RL (no resets), which would make Thm 3.1 a separation result [PoR §3.2].

## 4. SimGolf (Algorithm 1)

What it is: GOLF (global optimism over a version space of Q-functions) where the squared Bellman error constraint is estimated with local-simulator samples instead of a debiasing trick, which removes the need for Bellman completeness [PoR §3.1].

```
Algorithm 1  SimGolf(Q, C_cov, ε, δ)                                  [PoR §3.1, Alg. 1]
 1: input: value function class Q, coverability C_cov > 0, suboptimality ε > 0, confidence δ > 0
 2: N ← Θ̃(H² C_cov β / ε²);  β_stat ← 16 log(2HN|Q|/δ);  β ← 2 β_stat;  K ← 8N / β_stat
 3: initialize Q^(1) ← Q
 4: for t = 1, 2, …, N do
 5:   g^(t) = argmax_{g ∈ Q^(t)} Σ_{s<t} max_{a∈A} g_1(x_1^(s), a)          (optimism)
 6:   for each h, x: π_h^(t)(x) ∈ argmax_a g_h^(t)(x, a)
 7:   execute π^(t) for one episode, observe τ^(t) = (x_1^(t), a_1^(t)), …, (x_H^(t), a_H^(t))
 8:   for h ∈ [H] do
 9:     draw K independent samples x_{h+1}^(t,k) ~ T_h(· | x_h^(t), a_h^(t)), r_h^(t,k) ~ R_h(x_h^(t), a_h^(t))   (resets)
10:   Q^(t+1) ← { g ∈ Q : Σ_{i≤t} ( g_h(x_h^(i), a_h^(i)) − (1/K) Σ_{k=1}^K ( r_h^(i,k) + max_a g_{h+1}(x_{h+1}^(i,k), a) ) )² ≤ β, ∀h ∈ [H] }
11: return π̂ = unif(π^(1), …, π^(T))
```

(Line 10's printed sum index is "h ≤ t"; Eq. (2) and App. C use the episode index i, so I read it as i ≤ t. Line 11 prints T where the loop runs to N.)

- How resets are used: at each visited (x_h^(t), a_h^(t)) the learner resets K times to draw next states, which gives a direct estimate of T_h[g_{h+1}](x,a) for all g at once and avoids the double-sampling bias of squared Bellman error [PoR §3.1].
- Why inefficient: line 5 is a global-optimism step, maximizing over a version space cut out by H data-dependent constraints. No efficient implementation of global optimism is known in general, and some variants have provable barriers [PoR §4]. [inference] For neural Q there is no tractable way to search that constrained set.
- Proof idea: show the simulator-based confidence set satisfies Eq. (2) (low squared Bellman error on past data) with high probability [PoR Lem. C.1], then reuse the coverability change-of-measure argument of Xie et al. (2023) [PoR §3.2, App. D].
- Output is a uniform mixture of greedy policies of Q-functions, so it is executable [PoR Alg. 1, l. 6, 11].

## 5. RVFS (Recursive Value Function Search)

### 5.1 Mechanism

State: value estimates V̂_1,…,V̂_H (V̂_h aims at V⋆_h, or at the value of a near-optimal benchmark policy), confidence sets 𝒱̂_h ⊂ V (all functions that fit the layer-h regression data about as well as V̂_h), and core sets C_1,…,C_H [PoR §4.2].

- Core set C_h: a small set of layer-(h−1) state-action pairs (x_{h−1}, a_{h−1}). V̂_h is trained on next states x_h ~ T_{h−1}(·|x_{h−1}, a_{h−1}) drawn from these pairs by resetting. Informally C_h holds the pairs where we want E[|V̂_h(x_h) − V⋆_h(x_h)| | x_{h−1}, a_{h−1}] ≤ ε [PoR §4.2, footnote 6].
- Policy: π̂_τ(x) ∈ argmax_a P̂_τ[V̂_{τ+1}](x,a), the "Bellman backup policy", where P̂ estimates the backup with N_sim simulator draws from (x,a) [PoR §4.2, Eq. 15, Alg. 5]. Executing these backup policies, instead of an argmax over a learned Q̂, is used crucially in the analysis [PoR §4.2].
- Test (consistency check): from each core pair at layer h, roll out with π̂ to deeper layers; at each reached x_{ℓ−1} and each action a, ask whether some f in the confidence set 𝒱̂_ℓ disagrees with V̂_ℓ on the backup at (x_{ℓ−1}, a) by more than ε(1 + β). If yes, the data in C_ℓ does not pin down V_ℓ at the pushforward of (x_{ℓ−1}, a): distribution shift happened [PoR §4.2].
- On failure: add (x_{ℓ−1}, a) to C_ℓ, recursively re-run RVFS_ℓ, RVFS_{ℓ−1}, …, RVFS_{h+1} (each refits its layer and re-tests deeper layers), then restart all tests [PoR Alg. 3, l. 15 to 18].
- On success at every layer below h: fit V̂_h by least squares on Monte Carlo returns of π̂_{h:H} from next states of C_h, and build 𝒱̂_h [PoR Alg. 3, l. 20 to 27].
- Termination: the number of failures at each layer is at most M = ⌈8ε⁻¹C_push H⌉, so |C_h| ≤ M [PoR Lem. H.1]. The proof uses a pushforward-coverability potential argument [PoR Lem. B.8].
- When RVFS_0 returns, every test passed, and the greedy backup policy of V̂_{1:H} is near optimal [PoR §4.2, Thm H.1].
- Inspiration: DMQ (Du et al. 2019b; Wang et al. 2021) for linear Q⋆; core sets from linear RLLS work (Li et al. 2021; Yin et al. 2022; Weisz et al. 2022), here generalized to nonlinear V [PoR §4.2]. The paper calls RVFS a provable counterpart to MCTS / AlphaZero and compares its core sets to Go-Explore's archive [PoR §4.3].

### 5.2 Algorithm 3, RVFS_h (full version) [PoR App. E.1, Alg. 3]

Notation: V̂_ℓ is a fitted function; 𝒱̂_ℓ is a set of functions (confidence set); P̂_{ℓ,ε,δ'} is Algorithm 5. Transcribed from the page image.

```
Algorithm 3  RVFS_h : Recursive Value Function Search
 1: parameters: value function class 𝒱, suboptimality ε ∈ (0,1), confidence δ ∈ (0,1)
 2: input:
      level h ∈ {0, …, H}
      value function estimates V̂_{h+1:H}, confidence sets 𝒱̂_{h+1:H},
      state-action collections (core sets) C_{h:H}, buffers B_{h:H}, counters t_{h:H}
    /* Initialize parameters. */
 3: M ← ⌈8 ε⁻¹ C_push H⌉
 4: N_test ← 2⁸ M² H ε⁻¹ log(8 M⁶ H⁸ ε⁻² δ⁻¹),   N_reg ← 2⁸ M² ε⁻¹ log(8 |𝒱|² H M² δ⁻¹)
 5: N_est(k) ← 2 N_reg² log(8 A N_reg H k³ / δ)   and   δ' ← δ / (8 M⁷ N_test² H⁸ |𝒱|)
 6: ε_reg² ← 9 M H² log(8 M² H |𝒱|² / δ) / N_reg  +  34 M H³ log(8 M⁶ N_test² H⁸ / δ) / N_test
 7: β(t) ← sqrt( 2 log_{1/δ'}( 8 A M |𝒱| t² / δ ) )
    /* Test the fit for the estimated value functions V̂_{h+1:H} at future layers. */
 8: for (x_{h−1}, a_{h−1}) ∈ C_h do
 9:   for layer ℓ = H, …, h+1 do
10:     for n = 1, …, N_test do
11:       Draw x_h ~ T_{h−1}(· | x_{h−1}, a_{h−1}), then draw x_{ℓ−1} by rolling out with π̂_{h:H}, where
              ∀τ ∈ [H]:  π̂_τ(·) ∈ argmax_{a∈A} P̂_{τ,ε,δ'}[V̂_{τ+1}](·, a),   with V̂_{H+1} ≡ 0        (15)
12:       for a_{ℓ−1} ∈ A do
              /* Number of times P̂_{ℓ−1,ε,δ'} (Algorithm 5) is called in the test on Line 14. */
13:         Update t_ℓ ← t_ℓ + 1
              /* Test fit; if test fails, re-fit value functions V̂_{h+1:ℓ} up to layer ℓ. */
14:         if sup_{f ∈ 𝒱̂_ℓ} | ( P̂_{ℓ−1,ε,δ'}[V̂_ℓ] − P̂_{ℓ−1,ε,δ'}[f_ℓ] )(x_{ℓ−1}, a_{ℓ−1}) | > ε + ε·β(t_ℓ) then
15:           C_ℓ ← C_ℓ ∪ {(x_{ℓ−1}, a_{ℓ−1})}  and  B_ℓ ← B_ℓ ∪ {(x_{ℓ−1}, a_{ℓ−1}, V̂_ℓ, 𝒱̂_ℓ, t_ℓ)}
16:           for τ = ℓ, …, h+1 do
17:             (V̂_{τ:H}, 𝒱̂_{τ:H}, C_{τ:H}, B_{τ:H}, t_{τ:H}) ← RVFS_τ(V̂_{τ+1:H}, 𝒱̂_{τ+1:H}, C_{τ:H}, B_{τ:H}, t_{τ:H}; 𝒱, ε, δ)
18:           go to line 8
19: if h = 0 then return (V̂_{1:H}, ·, ·, ·, ·)
    /* Re-fit V̂_h and build a new confidence set. */
20: for (x_{h−1}, a_{h−1}) ∈ C_h do
21:   Set D_h(x_{h−1}, a_{h−1}) ← ∅
22:   for i = 1, …, N_reg do
23:     Sample x_h ~ T_{h−1}(· | x_{h−1}, a_{h−1})
24:     Let V̂_h(x_h) be a Monte Carlo estimate for E^{π̂_{h:H}}[ Σ_{ℓ=h}^H r_ℓ | x_h ] computed by collecting
          N_est(|C_h|) trajectories starting from x_h and rolling out with π̂_{h:H}
25:     Update D_h(x_{h−1}, a_{h−1}) ← D_h(x_{h−1}, a_{h−1}) ∪ {(x_h, V̂_h(x_h))}
26: Let V̂_h := argmin_{f ∈ 𝒱̂} Σ_{(x_{h−1},a_{h−1}) ∈ C_h} Σ_{(x_h, v_h) ∈ D_h(x_{h−1},a_{h−1})} (f(x_h) − v_h)²
27: Compute value function confidence set
       𝒱̂_h := { f ∈ 𝒱 :  Σ_{(x_{h−1},a_{h−1}) ∈ C_h} (1/N_reg) Σ_{(x_h, −) ∈ D_h(x_{h−1},a_{h−1})} ( V̂_h(x_h) − f(x_h) )² ≤ ε_reg² }   (16)
28: return (V̂_{h:H}, 𝒱̂_{h:H}, C_{h:H}, B_{h:H}, t_{h:H})
```

Reading notes:
- Line 24 overloads V̂_h(x_h) to mean a Monte Carlo return at x_h (a number). Call it v̂(x_h) in code. Line 26 then defines the fitted function V̂_h.
- In line 11, x_{ℓ−1} is reached by applying π̂_h, …, π̂_{ℓ−2} from x_h; for ℓ = h+1 it is x_h itself [inference from the indexing].
- π̂_h depends only on V̂_{h+1}, so the Monte Carlo targets in line 24 use only deeper estimates, which passed their tests before line 20 runs [inference from Eq. 15 and the control flow].
- "go to line 8" restarts the whole test loop over C_h from the beginning after any failure [PoR Alg. 3, l. 18].
- Buffers B and counters t: B is used only in the analysis and need not be stored; t only tunes β(t) [PoR App. E]. Confidence sets need not be stored explicitly if a regression oracle is available [PoR App. E, §4.2].

Base call [PoR App. E]:

```
V̂_{1:H} ← RVFS_0( V̂_{1:H} = arbitrary, 𝒱̂_{1:H} = {𝒱}_{h=1}^H, C_{0:H} = {∅}_{h=0}^H, B_{0:H} = {∅}_{h=0}^H, t_{1:H} = {1}_{h=1}^H; … )
```

Recursive calls: (V̂_{h:H}, 𝒱̂_{h:H}, C_{h:H}, B_{h:H}, t_{h:H}) ← RVFS_h(V̂_{h+1:H}, 𝒱̂_{h+1:H}, C_{h:H}, B_{h:H}, t_{h:H}; 𝒱, ε, δ) [PoR App. E]. See §9 for the ambiguity in what C_0 contains.

### 5.3 Algorithm 5, backup estimator P̂_{h,ε,δ}[f] [PoR App. E.1, Alg. 5]

```
Algorithm 5  P̂_{h,ε,δ}[f] : estimate E[r_h + f(x_{h+1}) | x_h = ·, a_h = ·]
 1: parameters: layer h, suboptimality ε ∈ (0,1), confidence δ ∈ (0,1), target function f
 2: input: (x, a) ∈ X × A
 3: Set N_sim := 2 log(1/δ) / ε²
 4: Set D ← ∅
 5: for i = 1, …, N_sim do
 6:   Sample r_h ~ R_h(x, a) and x_{h+1} ~ T_h(· | x, a)          // uses local simulator access
 7:   Update D ← D ∪ {(r_h, x_{h+1})}
 8: return N_sim⁻¹ · Σ_{(r,x) ∈ D} (r + f(x))
```

Guarantee: by Hoeffding plus a union bound over a ∈ A and f ∈ V, |P̂_{h,ε,δ'}[f](x_i,a) − P_h[f](x_i,a)| ≤ ε·sqrt(2 log_{1/δ'}(2Ai²|V|/δ)) simultaneously for all i, f, a [PoR Lem. G.2]. The union bound over f means one batch of samples per (x,a) can serve every f, which is how SimGolf uses its K samples [inference from Lem. G.2 and PoR §3.1].

### 5.4 Algorithm 4, RVFS.bc, and Algorithm 8, BehaviorCloning [PoR App. E.1, App. L]

```
Algorithm 4  RVFS.bc : learn an executable policy with RVFS via behavior cloning
 1: input: value function class 𝒱, policy class Π, suboptimality ε ∈ (0,1), confidence δ ∈ (0,1)
    /* Set parameters for RVFS. */
 2: Set ε_RVFS ← ε H⁻¹ / 48
 3: Set V̂_{1:H} ← arbitrary, 𝒱̂_{1:H} ← 𝒱, C_{0:H} ← ∅, B_{0:H} ← ∅, and t_i ← 0 for all i ∈ [0..H]
    /* Set parameters for BehaviorCloning. */   (as printed; these are RVFS parameters)
 4: Set M ← ⌈8 ε_RVFS⁻¹ C_push H⌉ and N_test ← 2⁸ M² H ε_RVFS⁻¹ log(80 M⁶ H⁸ ε_RVFS⁻² δ⁻¹)
 5: N_reg ← 2⁸ M² ε_RVFS⁻¹ log(80 |𝒱|² H M² δ⁻¹) and δ' = δ / (40 M⁷ N² H⁸ |𝒱|)
    /* Get value functions from RVFS */
 6: (V̂_{1:H}, ·, ·, ·, ·) ← RVFS_0(V̂_{1:H}, 𝒱̂_{1:H}, C_{0:H}, B_{0:H}, t_{0:H}; 𝒱, N_reg, N_test, ε_RVFS, δ/10)
    /* Extract executable policy via BehaviorCloning algorithm for imitation learning. */
 7: Define π̂_h^RVFS(·) ∈ argmax_{a∈A} P̂_{h, ε_RVFS, δ'}[V̂_{h+1}](·, a)
 8: Compute π̂_{1:H} ← BehaviorCloning(Π, ε, π̂_{1:H}^RVFS, δ/2)
 9: return π̂_{1:H}
```

```
Algorithm 8  BehaviorCloning(Π, π̂_{1:H}, ε, δ)
 1: input: policy class Π ⊆ Π_S, expert policy π̂_{1:H}, suboptimality ε ∈ (0,1), confidence δ ∈ (0,1)
 2: Set N_bc = 16 H² log(|Π|/δ) / ε
 3: Set D ← ∅
 4: for i = 1, …, N_bc do
 5:   Generate trajectory τ = ((x_1, a_1), …, (x_H, a_H)) ~ P^π̂
 6:   Update D ← D ∪ {τ}
 7: Compute π^bc ∈ argmin_{π∈Π} Σ_{τ∈D} Σ_{h∈[H]} 1{π_h(x_h) ≠ a_h}
 8: Return π^bc
```

BC guarantee: if inf_{π∈Π} Σ_h P^π̂[π̂_h(x_h) ≠ π_h(x_h)] ≤ ε_mis, then J(π̂) − J(π^bc) ≤ 4H·ε_mis + ε/2 w.p. 1−δ [PoR Prop. L.1]. The expert π̂ can be non-executable [PoR App. L]. Generating expert trajectories costs simulator calls because each expert action calls P̂ [inference from Alg. 4, l. 7].

### 5.5 What each parameter controls

- ε: target accuracy. Sets the test threshold ε + ε·β(t), N_sim = 2 log(1/δ')/ε² for every backup, and the core-set cap M [PoR Alg. 3, Alg. 5].
- δ, δ': overall failure probability; δ' is the per-call failure probability of P̂, shrunk by M⁷N_test²H⁸|𝒱| for union bounds [PoR Alg. 3, l. 5].
- M = ⌈8ε⁻¹C_push H⌉: proven upper bound on |C_h|, the number of test failures per layer [PoR Lem. H.1]. Needs C_push as an input. [inference] In practice it only appears in sample sizes; you can drop the cap and log |C_h|.
- N_test: rollouts per (core pair, layer ℓ) in the test phase. Controls how rare a bad state can be and still be caught: after passing, P^π̂[sup_f max_a |P_{ℓ−1}[V̂_ℓ] − P_{ℓ−1}[f_ℓ]| > 3ε | core pair] ≤ 4 log(8M⁶N_test²H⁸/δ)/N_test [PoR Lem. H.2].
- N_reg: next states sampled per core pair for regression [PoR Alg. 3, l. 22].
- N_est(k): Monte Carlo rollouts per regression target, growing with |C_h| = k [PoR Alg. 3, l. 5, 24].
- ε_reg: radius of the confidence set (16) [PoR Alg. 3, l. 6, 27].
- β(t): Hoeffding width that absorbs P̂'s estimation error in the test; grows slowly with the call count t [PoR Alg. 3, l. 7, Lem. G.2].
- ε_RVFS = ε/(48H): accuracy RVFS runs at inside RVFS.bc, so policy-level error after BC and the performance-difference lemma is ε [PoR Alg. 4, App. I.1].

### 5.6 How resets are used in RVFS

Every one of these needs a reset to a stored state [PoR §4.2, Alg. 3, Alg. 5]:
1. Sampling x_h ~ T_{h−1}(·|x_{h−1}, a_{h−1}) repeatedly from core pairs (lines 11, 23).
2. Every action of π̂ at any state x: reset to x, try each a ∈ A, N_sim times, to compute P̂[V̂](x, a) (Eq. 15).
3. Monte Carlo return estimates: N_est rollouts from the same x_h (line 24).
4. The test at (x_{ℓ−1}, a_{ℓ−1}) (line 14).

### 5.7 Proof skeleton (Setup II, then Setup I as a special case)

- Benchmark class Π_ε: stochastic policies induced by random Q̃ with |Q̃_h − Q_h^π| ≤ ε pointwise; any π̃ ∈ Π_ε has V⋆_h ≤ V_h^π̃ + 3Hε [PoR Def. G.1, Lem. G.1]. This replaces optimism.
- Lem. H.1: at most M test failures per layer, so RVFS terminates [PoR Lem. H.1].
- Lem. H.2: passing tests implies the backup of V̂_ℓ matches every f in 𝒱̂_ℓ within 3ε on most states reached by π̂ from core pairs [PoR Lem. H.2].
- Lem. H.3: regression error of V̂_h against V_h^π on core-set next states, plus a term from the total-variation distance between π̂ and π [PoR Lem. H.3].
- Lem. H.4: the confidence sets contain V^π̃ for a benchmark π̃ ∈ Π_{4ε} that follows the estimated backups wherever they are 4ε-accurate [PoR Lem. H.4].
- Thm H.1 combines these with Lem. K.1 (test passes imply π̂ close to π̃ in TV) [PoR Thm H.1, Lem. K.1].
- Setup I: with V⋆-realizability and gap Δ, Π_{ε'} = {π⋆} for ε' ≤ Δ/2, so Assumption H.1 holds [PoR Lem. I.1]; then P^π̂[π̂_h(x_h) ≠ π⋆_h(x_h)] ≤ ε/(4C_push H³) for ε < Δ/8 [PoR Lem. I.2]. With ε_RVFS = ε/(48H), ε_RVFS < Δ/8 matches the condition ε ≤ 6HΔ in Thm 4.1 [inference from the arithmetic].
- Episode count: M³H²(H+A)N_test N_sim + M²H²N_reg A N_est(M) N_sim, then substitute [PoR App. H.6].

### 5.8 Informal version (Algorithm 2) versus full version

Alg. 2 [PoR §4.2] is the same loop with exact backups P instead of P̂, a single β instead of β(t), no buffers or counters, test and recursion merged as "for (x_{h−1},a_{h−1}) ∈ C_h and ℓ = H,…,h+1", and line 15 regressing on exact returns E^{π̂_{h:H}}[Σ r | x_h]. Its recursive call passes 𝒱̂_{h+1:H} where Alg. 3 passes 𝒱̂_{τ+1:H} [PoR Alg. 2, l. 11; Alg. 3, l. 17]. Implement Alg. 3.

### 5.9 Oracle efficiency

The expensive steps are the regression (Alg. 3, l. 26) and the sup over the confidence set in the test (l. 14). The test reduces to argmax_{V∈𝒱} {± Σ_{i=1}^n V(x̃^(i)) : Σ_{i=1}^n (V(x^(i)) − y^(i))² ≤ β²}, a convex program in function space, provably efficient when V is linearly parameterized. The authors expect a reduction to square-loss regression is possible but do not do it [PoR §4.2].

## 6. RVFS^exo for Exogenous Block MDPs

### 6.1 ExBMDP model [PoR §3.3]

Latent state z_h = (s_h, ξ_h). Endogenous s_h ∈ S (finite, S := |S|) evolves as s_{h+1} ~ T_h^endo(·|s_h, a_h); exogenous ξ_h ∈ Ξ (possibly huge) evolves as ξ_{h+1} ~ T_h^exo(·|ξ_h), unaffected by actions. Observation x_h = g_h^obs(s_h, ξ_h) (equivalently x_h ~ q_h(s_h, ξ_h) with emission randomness folded into ξ) [PoR §3.3, footnote 2]. Decodability: some ϕ⋆ : X → S has ϕ⋆(x_h) = s_h a.s. Rewards depend on x only through ϕ⋆(x). Learner gets a decoder class Φ ∋ ϕ⋆ (Assumption 3.3). C_cov ≤ SA even with time-correlated ξ [PoR §3.3].

Function classes from Φ: Q_h = {x ↦ g(ϕ(x), a) : g ∈ [0,H]^{SA}, ϕ ∈ Φ} realizes Q⋆; Π_h = {x ↦ π(ϕ(x)) : π ∈ A^S, ϕ ∈ Φ} contains π⋆; log|Q_h| = log|Π_h| = Õ(S + log|Φ|) [PoR Lem. 3.1]. V_h = {x ↦ f(ϕ(x)) : f ∈ [0,H]^S, ϕ ∈ Φ} contains V^π for every endogenous π (π(x) depends only on ϕ⋆(x)) [PoR Lem. 4.1, footnote 9].

Why plain RVFS needs help: V^π-realizability (Setup II) fails in ExBMDPs because policies that react to ξ have values that are not functions of ϕ⋆(x) [PoR §4.4].

### 6.2 Randomized rounding

RVFS^exo draws ζ_1,…,ζ_H i.i.d. uniform on [0, 1/2] once and acts with π̂_h(x) ∈ argmax_a ⌈P̂_h[V̂_{h+1}](x,a)/ε + ζ_h⌉, i.e., backups quantized into bins of width ε with random offset [PoR §4.4, Eqs. 5 to 6]. With probability ≥ 1 − 24SAHε over ζ, the induced benchmark policy "snaps" onto an endogenous policy π̄ for every choice of V̂ simultaneously [PoR Lem. J.2], and π̄ ∈ Π_{2ε} [PoR Lem. J.1]. This mimics the gap: rounded values are separated by Θ(ε²) [PoR App. J.1.1]. It relies on S being finite; the authors do not know whether it works for large latent spaces [PoR §4.4].

### 6.3 Algorithm 6, RVFS^exo_h [PoR App. E.2, Alg. 6]

Identical to Algorithm 3 except the lines below (transcribed from the page image).

```
 1: parameters: value function class 𝒱, suboptimality ε ∈ (0,1), seeds ζ_{1:H} ∈ (0,1), confidence δ ∈ (0,1)
 2: input: level h ∈ [0..H], V̂_{h+1:H}, 𝒱̂_{h+1:H}, C_{h:H}, B_{h:H}, t_{h:H}
 3: M ← ⌈8 ε⁻² C_exo S A H⌉
 4: N_test ← 2⁸ M² H ε⁻² log(8 M⁶ H⁸ ε⁻² δ⁻¹),   N_reg ← 2⁸ M² ε⁻² log(8 |𝒱| H M² δ⁻¹)
 5: N_est(k) ← 2 N_reg² log(8 A N_reg H k³ / δ)   and   δ' ← δ / (4 M⁷ N_test² H⁸ |𝒱|)
 6: ε_reg² ← 9 M H² log(8 M² H |𝒱| / δ) / N_reg  +  34 M H³ log(8 M⁶ N_test² H⁸ / δ) / N_test
 7: β(t) ← sqrt( log_{1/δ'}( 8 M A |𝒱| t² / δ ) )
11: Draw x_h ~ T_{h−1}(·|x_{h−1}, a_{h−1}), then draw x_{ℓ−1} by rolling out with π̂_{h+1:H}, where
        ∀τ ∈ [H]:  π̂_τ(·) ∈ argmax_{a∈A} ⌈ P̂_{τ,ε²,δ'}[V̂_{τ+1}](·, a) · ε⁻¹ + ζ_τ ⌉,   with V̂_{H+1} ≡ 0
14: if sup_{f ∈ 𝒱̂_ℓ} | ( P̂_{ℓ−1,ε²,δ'}[V̂_ℓ] − P̂_{ℓ−1,ε²,δ'}[f_ℓ] )(x_{ℓ−1}, a_{ℓ−1}) | > ε² + ε²·β(t_ℓ) then
17:   (…) ← RVFS^exo_τ(V̂_{τ+1:H}, 𝒱̂_{τ+1:H}, C_{τ:H}, B_{τ:H}, t_{τ:H}; 𝒱, ε, ζ_{1:H}, δ)
24: For each a ∈ A, let V̂_h(x_h) be a Monte Carlo estimate for E^{π̂_{h:H}}[Σ_{ℓ=h}^H r_ℓ | x_h] computed by
      collecting N_est(|C_h|) trajectories starting from x_h and rolling out with π̂_{h:H}
25: Update D(x_{h−1}, a_{h−1}) ← D(x_{h−1}, a_{h−1}) ∪ {(x_h, V̂_h(x_h))}
26: V̂_h := argmin_{f ∈ 𝒱_h} Σ_{C_h} Σ_{D_h} (f(x_h) − v_h)²
27: 𝒱̂_h := { f ∈ 𝒱_h : Σ_{C_h} (1/N_reg) Σ_{D_h} (V̂_h(x_h) − f(x_h))² ≤ ε_reg² }   (17)
```

All other lines (8 to 10, 12, 13, 15, 16, 18 to 23, 28) match Algorithm 3. Key changes: ε becomes ε² in P̂ and in the test threshold, and the policy uses rounded backups.

### 6.4 Algorithm 7, RVFS^exo.bc [PoR App. E.2, Alg. 7]

```
 1: input: decoder class Φ, suboptimality ε ∈ (0,1), confidence δ ∈ (0,1)
 2: ε_RVFS ← ε H⁻¹ / 48
 3: 𝒱 = 𝒱_{1:H}, 𝒱_h = {x ↦ f(ϕ(x)) : f ∈ [0,H]^S, ϕ ∈ Φ}
 4: Π = Π_{1:H},  Π_h = {π(·) ∈ argmax_a f(ϕ(·), a) : f ∈ [0,H]^{S×A}, ϕ ∈ Φ}
 5: N_bc ← 8 H² log(4H|Π|/δ)/ε,  N_boost ← log(1/δ)/log(24 S A H ε),  N_eval ← 16² ε⁻² log(2 N_boost / δ)
 6: M ← ⌈8 ε_RVFS⁻¹ S A C_cov H⌉,  N_test ← 2⁸ M² H ε_RVFS⁻¹ log(80 M⁶ H⁸ N_boost ε_RVFS⁻² δ⁻¹),
    δ' = δ / (40 M⁷ N² H⁸ |𝒱| N_boost)
 7: N_reg ← 2⁸ M² ε_RVFS⁻¹ log(80 |Φ|² H M² N_boost δ⁻¹)
 8: V̂_{1:H} ← arbitrary, 𝒱̂_{1:H} ← 𝒱, C_{0:H} ← ∅, B_{0:H} ← ∅, i_opt = 1, J_max = 0
 9: for i = 1, …, N_boost do
10:   (V̂^(i)_{1:H}, ·,·,·,·) ← RVFS^exo_0(V̂_{1:H}, 𝒱̂_{1:H}, C_{0:H}, B_{0:H}; 𝒱, N_reg, N_test, ε_RVFS, δ/(10 N_boost))
11:   π̂_h^RVFS(·) ∈ argmax_a P̂_{h, ε_RVFS, δ'}[V̂^(i)_{h+1}](·, a)
12:   π̂^(i)_{1:H} ← BehaviorCloning(Π, ε, π̂^RVFS_{1:H}, δ/(2 N_boost))
13:   v = 0
14:   for = 1, …, N_eval do
15:     sample trajectory (x_1, a_1, r_1, …, x_H, a_H, r_H) by executing π̂^(i)_{1:H}
16:     v ← v + Σ_{h=1}^H r_h
17:   Ĵ(π̂^(i)_{1:H}) ← v / N_eval
18:   if Ĵ(π̂^(i)_{1:H}) > J_max then
19:     i_opt = i
20:     J_max = Ĵ(π̂^(i)_{1:H})
21: return π̂_{1:H} = π̂^(i_opt)_{1:H}
```

Printed oddities (kept verbatim above): line 10 does not pass fresh seeds ζ^(i) though the proof samples i.i.d. ζ^(1),…,ζ^(n) per run [PoR App. J.1.7]; line 11 defines π̂^RVFS without the rounding of Eq. (6); line 6 uses C_cov where Alg. 6 uses C_exo; see §9.

## 7. Implementing RVFS

### 7.1 What the environment must provide [inference unless tagged]

- `snapshot()` / `restore(snap)` for the full simulator state, plus the layer index h, since the MDP is layer-indexed and resets are to (h, x_h) [PoR §2.1].
- After `restore`, transitions must be resampled with fresh randomness. If the snapshot also freezes the RNG, every "independent" draw in P̂ or line 23 returns the same next state, so P̂ becomes a one-sample estimate in a stochastic environment.
- A finite action set: Eq. (15) and the test loop over a ∈ A enumerate actions [PoR Alg. 3, l. 12]. Continuous actions need discretization.
- Rewards in [0,1] and values in [0,H] are assumed in the concentration bounds [PoR §2, Lem. B.4]; rescale rewards.

### 7.2 Data structures [inference, mapped to paper objects]

Per layer h ∈ {1,…,H}:
- `core[h]`: list of (snapshot at layer h−1, action) pairs, i.e., C_h. C_0 / C_1 also need a representation of the initial distribution (see §9).
- `data[h][pair]`: list of (observation x_h, Monte Carlo return) for each core pair, i.e., D_h. Rebuilt from scratch on each refit (Alg. 3, l. 21).
- `V_hat[h]`: parameters of the fitted function V̂_h.
- `conf[h]`: whatever represents 𝒱̂_h implicitly. For linear V: the fit θ̂_h, the design matrix Σ_h = Σ_{pairs}(1/N_reg)Σ_{x∈D} φ(x)φ(x)ᵀ, and ε_reg.
- `t[h]`: test call counter, only needed if you use β(t).
- No need to store B_h [PoR App. E].
- A cache for P̂ results keyed by (h, snapshot id, action, version of V̂_{h+1}) saves many repeated rollouts, since the same states are queried repeatedly. Invalidate when V̂_{h+1} changes.

Control flow: a recursive function `rvfs(h)` that mutates shared per-layer state. Recursion depth ≤ H. Line 18's "go to line 8" is a restart of the outer test loop; write it as `while True: ... failed = test_all(); if not failed: break`.

### 7.3 Function class and the two oracles

1. Regression oracle (Alg. 3, l. 26): least squares over V on {(x_h, v̂)}. Any regressor works in practice; the theory assumes exact ERM over a finite class [PoR Lem. B.5].
2. Confidence-set test (l. 14): sup over f with small empirical squared distance to V̂_h on core data of |P̂[V̂_ℓ] − P̂[f_ℓ]| at the new pair.

Linear V, f(x) = θᵀφ(x) [inference, derived from Eq. 16 and the reduction in PoR §4.2]: if the same N_sim next states x'_1..x'_N are used for V̂ and for f, rewards cancel and the test statistic is |(θ̂ − θ)ᵀ u| with u = (1/N)Σ_i φ(x'_i). The confidence set (16) is the ellipsoid (θ − θ̂)ᵀ Σ_ℓ (θ − θ̂) ≤ ε_reg² (with Σ_ℓ as in §7.2, θ restricted to the realizable class). So

  sup = ε_reg · sqrt(uᵀ Σ_ℓ⁻¹ u)   (use Σ_ℓ + λI if singular).

The test fails when ε_reg·‖u‖_{Σ_ℓ⁻¹} > ε(1 + β): an elliptical-uncertainty check like the ones in linear core-set methods. This is exact for the unconstrained linear class and is the version to implement first.

Neural V [inference]: no closed form. Options, in order of fidelity: (a) solve the constrained program approximately with a Lagrangian, max_f ±Σ_i f(x'_i) − λ·(squared distance to V̂ on core data), warm-started from V̂; (b) use a bootstrap or randomized-prior ensemble fit on the core data and take max disagreement of backups as the sup; (c) linear head on frozen features plus the closed form. Option (b) changes the semantics from a version-space sup to a heuristic.

3. Policy class Π for RVFS.bc: the theory minimizes 0-1 disagreement [PoR Alg. 8, l. 7]; in practice train a classifier with cross-entropy on (x_h, π̂_h^RVFS(x_h)) [inference].

### 7.4 What you must choose in practice [inference]

The theoretical constants (§7.5) are unusable, so treat these as hyperparameters:
- ε (test threshold scale) and β: start with threshold = c·(estimated noise of P̂) with c ≈ 2 to 3, or fix β = 1 and sweep ε.
- N_sim: 1 for deterministic dynamics; 8 to 64 for stochastic. This dominates cost because it multiplies every step of every rollout by A·N_sim.
- N_test: rollouts per core pair per layer, e.g. 5 to 50.
- N_reg: next-state samples per core pair, e.g. 10 to 50; N_est: Monte Carlo rollouts per target, e.g. 5 to 20 (the paper's N_est is quadratic in N_reg).
- ε_reg: confidence radius; with linear V it scales the elliptical bonus directly.
- Horizon H: costs grow as a high power of H; keep H ≤ 10 to 20 for first experiments.
- Function class: tabular or one-hot features (sanity check), then random/known features with a linear head, then a small MLP.
- Optional cap on |C_h| (the theory cap is M) to bound runtime, logging when it binds.

### 7.5 Costs that blow up

- Episode count bound: M³H²(H+A)N_test N_sim + M²H²N_reg A N_est(M) N_sim with M = ⌈8C_push H/ε⌉ [PoR App. H.6]. The headline rate is Õ(C_push⁸ H²³ A ε⁻¹³) [PoR Thm 4.1].
- Example [inference, arithmetic]: ε = 0.1, H = 5, C_push = 2 gives M = 800 and N_test ≈ 2⁸·800²·5/0.1·log(·) ≈ 8×10⁹·log(·). Not runnable; hand-pick constants.
- Each step of π̂ costs A·N_sim simulator calls; a rollout of length H costs about H·A·N_sim. Monte Carlo targets nest this: N_reg × N_est rollouts per core pair [PoR App. H.6].
- Every test failure triggers recursive refits of layers ℓ down to h+1 and a full restart of the test loop at the calling level [PoR Alg. 3, l. 16 to 18]. Early in training failures are frequent, so expect many restarts.
- RVFS^exo multiplies everything by N_boost runs and uses ε² in place of ε [PoR Alg. 6, Alg. 7].

### 7.6 Toy environments for sanity checks

The paper has no experiments and suggests none [PoR §5]. Suggestions [inference]:
1. Tabular layered MDP ("combination lock"): H layers, 2 to 3 latent states per layer, A = 2 to 4, one good action per state, reward only at the end, stochastic transitions with small slip probability. V = tabular (one-hot linear). Known V⋆, Q⋆ and gap Δ; C_push ≤ number of states per layer. Check: |C_h| stays small, J(π̂) ≈ J(π⋆), test failures go to zero.
2. Rich-observation Block MDP version: observation = random rotation of [one-hot(s), Gaussian noise dims]. Decodable, so it is a Block MDP (C_push ≤ |S| [PoR §4.1]). Use linear V on the observation, or V(x) = f(ϕ(x)) with a learned decoder.
3. ExBMDP version: make the noise dims a discrete Markov chain (exogenous state ξ) with controllable stickiness, so C_exo is finite and tunable; i.i.d. noise across time gives C_exo = 1.
4. Non-decodable version: observation = s-embedding + Gaussian noise with overlapping supports; sweep σ (see §8).
Baselines on these: exact value iteration (ground truth), SimGolf is not practical, DQN without resets, and GowU from the UGTS paper (with resets, no value function).

### 7.7 Pitfalls checklist [inference]

- Using `argmax_a Q̂(x,a)` instead of the backup policy argmax_a P̂[V̂_{h+1}](x,a) departs from what the analysis needs [PoR §4.2].
- Forgetting V̂_{H+1} ≡ 0 [PoR Eq. 15].
- Reusing stale P̂ caches after V̂ changes.
- Refitting V̂_h on Monte Carlo targets generated by a policy that uses an untested V̂_{h+1}.
- Restoring RNG with the snapshot (see 7.1).
- Comparing V̂ and f on different next-state samples in the test, which adds noise the threshold must absorb.

## 8. Noisy observations

Setup for noisy observations: the learner sees o instead of the state, e.g. o = s + η or o ~ q(·|s). Everything in RVFS (V̂(x), core pairs, resets, P̂) is written in terms of one object x that is both the reset key and the function input [PoR §2, Alg. 3]. With noisy observations those split into two objects: the simulator checkpoint (latent s, used for resets) and the observation o (used as function input). Which results survive depends on whether s is recoverable from o.

### 8.1 What the paper already covers: decodable rich observations

- Block MDP (o emitted from a finite latent s, s recoverable from o): C_push ≤ |S|, so Thm 4.1 applies when its realizability conditions hold [PoR §4.1].
- ExBMDP: o = g(s, ξ) with exogenous noise ξ that may be temporally correlated and arbitrarily high-dimensional, but s = ϕ⋆(o) is decodable. SimGolf handles the general case [PoR Cor 3.1]. RVFS needs weak correlation (Assumption 4.7): with a gap it uses Thm 4.1 Setup I; without a gap it uses RVFS^exo [PoR §4.4, Thm 4.2]. Function classes are built on top of a decoder class Φ ∋ ϕ⋆ [PoR Lem. 3.1, Lem. 4.1].
- So "noise" that lives in extra dimensions or distractor content, while the true latent is still readable from o, is within the paper's theory.

### 8.2 Decodable noise: what to check [inference]

- i.i.d.-in-time distractors: P[ξ_h, ξ_{h+1}] factorizes, so C_exo = 1 and C_push ≤ SA via Lem. J.3.
- Temporally correlated continuous distractors, e.g. Gaussian AR(1) with correlation ρ ≠ 0: the density ratio p(ξ, ξ')/(p(ξ)p(ξ')) is unbounded in the tails (with unit variances and ρ > 0, at ξ = ξ' = x it equals (1−ρ²)^{−1/2} exp(ρx²/(1+ρ)); use ξ' = −ξ for ρ < 0), so Assumption 4.7 fails with C_exo = ∞. RVFS's guarantee no longer applies; SimGolf's (Cor 3.1) still does since C_cov ≤ SA does not need weak correlation [PoR §3.3]. A discrete or bounded-ratio exogenous chain keeps C_exo finite.
- Realizability then requires V of the form f(ϕ(o)) with ϕ⋆ in your decoder class; a generic MLP on o is the practical stand-in, with no guarantee.

### 8.3 Non-decodable noise, o = s + η with overlapping supports [inference throughout]

Not a Block MDP or ExBMDP; it is a POMDP. Item by item:
- Markov property over o: T(o' | o, a) depends on the posterior over s given o (and on history), so the "MDP over observations" assumed in §2 does not exist. The paper's setting has no observation model outside ExBMDPs.
- Realizability: V⋆ and π⋆ are functions of s. A class over o can at best represent E[V⋆(s) | o] (or history-dependent beliefs), so Assumptions 4.2/4.3/4.5/4.6 fail. Δ-gap in o-space also fails near decision boundaries where o is ambiguous.
- Pushforward coverability can survive: if s is finite and o' ~ q(·|s'), then T(o'|·) = Σ_{s'} P(s'|·) q(o'|s') ≤ |S|·μ(o') with μ = (1/|S|)Σ_{s'} q(·|s'), so C_push ≤ |S|. The structural condition is not what breaks; realizability and Markovness are.
- Resets: in a simulator you restore the latent checkpoint, not o. Then P̂[f](checkpoint, a) = E[r + f(o') | s, a] is still well defined, and the greedy backup policy π̂ effectively acts on the latent state through the simulator. The non-executable policy is fine; the executable one is not.
- Behavior cloning: distilling a latent-state expert into an o-based policy has irreducible disagreement ε_mis ≥ the Bayes error of predicting π̂(s) from o. By Prop. L.1 the loss is ≤ 4H·ε_mis + ε/2 [PoR Prop. L.1], so performance degrades linearly in H times the per-step confusion rate. History-based policies (frame stacks, recurrent nets) lower ε_mis.
- Regression: V̂_h(o) trained on core-set pushforwards estimates E[return | o] under the mixture of core-pair next-state distributions, which changes as C_h grows, so the target itself moves. Lem. B.5 already has a bounded-bias term (Σ|b_i|) that could model approximate realizability; extending Lem. H.3 with a misspecification term is a natural thesis step.
- Small bounded noise: if η is smaller than half the separation between distinct latent states, rounding recovers s, the model is a Block MDP again, and §8.1 applies. Gaussian noise breaks this with per-step error probability P(|η| > d/2) (d = separation), giving a clean knob between "decodable" and "not decodable".

### 8.4 Experimental knobs for a noisy-observation study [inference]

- Noise type: (a) distractor dims i.i.d., (b) distractor dims Markov with tunable C_exo, (c) additive noise on the state with bounded support below the separation, (d) Gaussian additive noise sweeping σ.
- Reset key: latent checkpoint (realistic simulator) versus "observation-keyed" resets (resample latent from the posterior given o), the latter being closer to having only the observation.
- Policy input: o_t only versus a short history.
- Metrics: J(π̂) of the non-executable backup policy and of the BC policy separately; |C_h| per layer; number of test failures; total simulator calls; regression error against true V^π on held-out latent states.
- Prediction to test: the non-executable policy should degrade slowly in σ (it sees latent next states through resets), while the BC policy degrades roughly with H·ε_mis(σ).

## 9. Ambiguities and apparent typos in the PDF

1. Base call C_0 [PoR App. E, Alg. 4, l. 3]: printed as C_{0:H} = {∅} (main text, App. E) and C_{0:H} ← ∅ (Alg. 4). If C_0 is the empty set, RVFS_0's test loop (line 8) never runs and RVFS returns immediately. The intended reading is almost certainly C_0 = {∅} with ∅ a dummy pair whose "next state" is x_1 ~ T_0(·|∅) [inference from the T_0 convention in §2]. The indexing at h = 0 (x_{−1}, T_{−1}) is not spelled out.
2. Line 26 of Alg. 3 minimizes over "𝒱̂" (hat) while Eq. (16) and Alg. 6 l. 26 use the full class 𝒱 / 𝒱_h. Read as argmin over 𝒱_h [inference].
3. Footnotes 5 (Alg. 2) and 11 (Alg. 3) are referenced next to Eq. (4)/(15) but their text does not appear in the PDF.
4. Thm 4.1 states J(π⋆) − J(π̂) ≤ 2ε; the restated Thm H.2 (same algorithm, same ε) states ≤ ε.
5. Alg. 3 l. 11 rolls out with π̂_{h:H}; Alg. 6 l. 11 says π̂_{h+1:H}. Since x_h is already drawn, π̂_h is the first policy applied, so π̂_{h:H} looks right [inference].
6. Alg. 6 l. 24 says "For each a ∈ A, let V̂_h(x_h) be a Monte Carlo estimate…" but nothing in the line depends on a.
7. Alg. 7: N_boost = log(1/δ)/log(24SAHε) is negative when 24SAHε < 1 (the regime where Lem. J.2 is useful); presumably log(1/δ)/log(1/(24SAHε)), matching the proof's (24SAHε)^{N_boost} ≤ δ/4 [PoR App. J.1.7]. Line 6 uses C_cov where Alg. 6 uses C_exo. Line 10 does not show fresh ζ per run. Line 11 drops the rounding.
8. Alg. 4 l. 5 and Alg. 7 l. 6 write δ' with "N²" where Alg. 3 has N_test². Alg. 4's comment "Set parameters for BehaviorCloning" sits above RVFS parameters. Alg. 4 l. 6 passes (N_reg, N_test, ε_RVFS, δ/10) to RVFS_0, a different signature than Alg. 3's (𝒱, ε, δ).
9. Counters start at t = 1 in the App. E base call but t_i ← 0 in Alg. 4 l. 3.
10. Alg. 8 sets N_bc = 16H² log(|Π|/δ)/ε, the proof uses 16H² log(2|Π|/δ)/ε, and Alg. 7 sets N_bc = 8H² log(4H|Π|/δ)/ε without passing it [PoR App. L, Alg. 7].
11. SimGolf: confidence-set sum printed over "h ≤ t" (read i ≤ t); return over π^(1..T) with T undefined (read N) [PoR Alg. 1].
12. §4.2 text says P̂ uses "K rewards and next-state transitions"; Alg. 5 calls the count N_sim [PoR §4.2, Alg. 5].
13. ExBMDP resets: the paper treats observations x as the states one resets to, and Lem. J.3's proof computes T(x'|x,a) by conditioning only on x_{h−1} = x (marginalizing ξ_{h−1}). It does not say what a reset to x restores when ξ is not recoverable from x [PoR App. J.4]. In a real simulator a reset restores (s, ξ) [inference].
