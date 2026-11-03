# Model-based RL

Sources: Notes 4, AJKS ch. 2. Model-based means estimating the dynamics P (and possibly r) and then planning in the estimate; model-free methods find π or Q⋆ without ever estimating P. Neither dominates.

## Generative model protocol (Def. 1)

An oracle that, given any (s, a), returns r(s, a) and a sample s' ~ P(·|s,a), without having to reach s. Strongest access short of knowing the MDP. Compare: online (only trajectories from s_0 ~ μ) < local simulator (reset to visited states, Power of Resets) < generative (any state).

## Tabular: naive model estimation (Alg. 1)

Query each (s, a) N times; P̂(s'|s,a) = (count of s')/N. Solve the estimated MDP with VI/QVI/PI to get π̂ = π⋆(P̂).

PAC framing (Prop. 2): "probably approximately correct" means P(V^{π̂} ≤ V⋆ − ε) ≤ δ once the sample count exceeds f(|S|, |A|, γ, 1/ε, 1/δ), the sample complexity. Upper bounds come from an algorithm; lower bounds must hold for every algorithm. Sample complexity is about data, not computation.

Results (from AJKS, as stated in Notes 4 §3.1):
- Thm 3: with SAN ≥ γ S²A log(cSA/δ) / ((1−γ)⁴ ε²) (c a universal constant) total samples, with probability 1−δ: max_{s,a} ‖P̂(·|s,a) − P(·|s,a)‖_1 ≤ (1−γ)² ε; every policy's Q is estimated within ε; π̂ = π⋆(P̂) has ‖Q^{π̂} − Q⋆‖ ≤ 2ε. This needs N ≈ S per pair, because it controls all policies at once (too strong).
- Sharper: about SA log(SA/δ) / ((1−γ)³ ε²) samples suffice for the plug-in policy and this is minimax optimal (variance-based analysis). N = o(S) per pair is enough because a concentrated P(·|s,a) is learned from few samples, and a near-uniform one only matters when actions are nearly equally bad anyway.
- Uniform sampling suffices here (no bandit-style adaptivity), probably because the worst-case instances have tiny gaps.

A useful experiment measures three errors for each dataset size (k samples per (s, a) pair), averaged over seeds:
- E^P = max_{s,a,s'} |P̂ − P| (model error, sup over entries, not the L1 of Thm 3),
- E⋆ = max_s (V⋆ − V^{π̂}) (true suboptimality, evaluated under real P),
- E^V = max_s (V^{π̂}_{P̂} − V^{π̂}) (how much the model flatters its own policy).
[inference] Expect all three to shrink as k grows, and E^V to tend positive at small k because the planner exploits model errors; it is a max of signed differences, so it is not guaranteed positive. Use the generator's seed argument for reproducibility.

## Continuous spaces

- LQR with noise (§4.1): query isotropic Gaussian s and a, fit [A B] by least squares (`scripts/lqr.py estimate_AB`). Noise-free LQR needs only n + m basis queries.
- General continuous MDPs: provable methods exist but get intricate fast and can need intractable subroutines. In robotics practice: fit a neural net or GP model and plan with heuristics. Under a generative model, assume roughly uniform model error over S.

## Model-predictive control (§5.1)

At each step: observe s_t; solve trajectory optimization over K steps in the model; execute only the first action; repeat. Replanning from the real next state absorbs model error. Deterministic model: continuous optimization (if differentiable) or tree search (discrete A).

## Sampling-based MPC (§5.2): stochastic models

Template: keep a distribution D over action sequences; sample sequences and model rollouts; score returns; shift D toward high-return samples.

| Method | D and update |
|---|---|
| Predictive sampling (Alg. 2) | Gaussian of fixed σ around the current plan; keep the single best sample |
| Cross-entropy method (Fig. 1, original authors' listing) | parametric f(·; v); keep the top ρ fraction ("elites") and refit v by maximum likelihood on them |
| MPPI (Fig. 2, Model-Predictive Path Integral) | perturb a nominal control sequence; average the perturbations weighted by exp(−S̃(τ_k)/λ) where S̃ is trajectory cost and λ a temperature; execute u_0, shift the sequence, append u_init |

CEM refits on the top ⌈ρN⌉ elites and MPPI weights the whole batch, so both use more than the single best sample and are less noisy than predictive sampling [inference]; λ (MPPI) and the elite fraction ρ (CEM) trade off greediness against averaging.

## Exploration without a generative model (§6)

Need to reach states to learn about them. Two templates, episodic, finite horizon:
- Optimism (UCB-style): keep a confidence set 𝒫_i of MDPs consistent with the data; jointly pick (P̂_i, π_i) maximizing E V^π_{P̂}(s_1) over the set; run π_i one episode; intersect the set with the new data.
- Thompson sampling: keep a posterior over MDPs; sample P̂_i, solve it, run the optimal policy one episode, Bayes-update.
TS's per-episode problem is much easier (solve one MDP, not optimize over a set), but representing the set or posterior is hard for neural-network model classes (no conjugate priors). That difficulty is one reason practical robot RL leans on model-free methods, with learned models used for planning (MPC) or imagined rollouts (Dreamer, TD-MPC2, SuperMimic).
