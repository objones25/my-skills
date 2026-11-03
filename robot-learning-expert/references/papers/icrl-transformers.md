# Transformers Provably Implement In-Context Reinforcement Learning with Policy Improvement

Haodong Liang, Lifeng Lai (UC Davis). Preprint, May 2026 (no venue stated). arXiv:2605.05755 (v1).

## The idea

In-context RL (ICRL) means a frozen model reads trajectory data in its prompt and outputs an improved policy, with no weight updates. The paper writes down explicit weights for a single linear self-attention layer whose output is exactly one batch step of semi-gradient SARSA (Thm 3.1) or actor-critic (Thm 3.3), and shows that training the layer to imitate those teacher updates converges locally and exponentially to the set of such weights (Thm 3.2) [ICRL §1]. Earlier theory (Wang et al., ICLR 2025) only covered policy evaluation (TD) and only showed the solution set is invariant under training.

## Notation

[ICRL §2, §3.1]

- Discounted MDP (S, A, P, R, γ); return G_t = Σ_k γ^k r_{t+1+k}; q_π(s, a), v_π(s) as in Sutton and Barto.
- Linear self-attention block with parameters θ = (P, V), P, V ∈ R^{D×D}. P = Q^T K merges query and key matrices. On input H ∈ R^{D×n}: H_out = H + (1/n) V H (H^T P H). TF_θ(H) = f_read(H_out), a fixed readout. No softmax, one head, one layer.
- SARSA features φ : S × A → R^d, critic q_w(s, a) = w^T φ(s, a), step size α.
- Trajectory window τ = (s_0, a_0, r_1, ..., s_n, a_n). Shorthands: φ_i = φ(s_i, a_i), φ_i^+ = φ(s_{i+1}, a_{i+1}), TD error δ_i = r_{i+1} + γ w^T φ_i^+ − w^T φ_i.
- Token x_i = [φ_i; γ φ_i^+; r_{i+1}] ∈ R^{2d+1}; w̃ = [1; w] ∈ R^{d+1}.
- Prompt H(z) = [[x_0 ... x_{n−1}, 0], [0 ... 0, w̃]] ∈ R^{D×(n+1)}, D = 3d + 2 (Eq. 4). The last column carries the current parameter.
- Statistics: Σ̂(z) = (1/n) Σ x_i x_i^T, R(z) = (1/n) Σ φ_i x_i^T, b(z) = (1/n) Σ x_i δ_i.
- z = (M, φ, w, τ): a training sample (task MDP, features, behavior parameter, trajectory).
- Actor-critic: policy features φ_π ∈ R^m, softmax policy π_λ(a | s) ∝ exp(λ^T φ_π(s, a)), value features φ_V ∈ R^d, critic v_w(s) = w^T φ_V(s), score g_λ(s, a) = φ_π(s, a) − Σ_b φ_π(s, b) π_λ(b | s), step sizes α (actor), β (critic).

## Core definitions

Batch semi-gradient SARSA update (Eq. 2): w_SARSA = w + (α/n) Σ_{i=0}^{n−1} δ_i φ_i. Plain meaning: one averaged TD step on the whole window; "semi-gradient" because the target γ w^T φ_i^+ is not differentiated.

Batch actor-critic update (Eq. 3): w_AC = w + (β/n) Σ δ^V_i φ_V(s_i), λ_AC = λ + (α/n) Σ γ^i δ^V_i g_λ(s_i, a_i), with δ^V_i = r_{i+1} + γ w^T φ_V(s_{i+1}) − w^T φ_V(s_i). Plain meaning: TD(0) critic step plus a policy-gradient actor step that uses the TD error as the advantage estimate.

Optimal parameter manifold (Eq. 9). Only two blocks matter in training (Prop 3.1): θ_eff = (P_12, V̄_21). The target values are P*_12 = [[0, −I_d], [0, I_d], [1, 0]] and V̄*_21 = [α I_d, 0, 0], and any rescaling (c P*_12, c^{−1} V̄*_21) works. M_I = {(c P*_12, c^{−1} V̄*_21) : c ∈ I} for a compact interval I ⊂ (0, ∞). Plain meaning: a one-dimensional curve of equivalent "correct" weights, which is why ordinary strong convexity fails and a local Polyak-Łojasiewicz (PL) inequality is used instead.

Why the construction works [ICRL Thm 3.1, App E; the arithmetic below is a check, not quoted]. P*_12 w̃ = [−w; w; 1], so for each token x_i^T P*_12 w̃ = −w^T φ_i + γ w^T φ_i^+ + r_{i+1} = δ_i. Attention then sums x_i δ_i, giving b(z), and the bottom rows of V* pick the φ-part and multiply by α: output w + α (1/n) Σ φ_i δ_i. One attention layer computes all TD errors and the batch update at once.

Assumption 1, richness of training MDPs [ICRL §3.2]. For every behavior policy used: (1) bounded features, rewards, and w̃; (2) parameter excitation E[w̃ w̃^T] ⪰ κ_w̃ I; (3) Bellman-regressor excitation E[R(z)^T R(z) | w] ⪰ κ_R I for every w; (4) TD-target excitation E[b(z) b(z)^T] ⪰ κ_q I; (5) no destructive cancellation, a cross-correlation bound with ρ < 1 on the normal space of the manifold. Plain meaning: the training tasks, features, and trajectories must vary in every direction, so the loss pins down the weights. (1) and (2) can be enforced by normalization and random w; (3) to (5) are claimed to hold generically but are not verified for any MDP family.

## Main results

| Result | Statement | Assumptions |
|---|---|---|
| Thm 3.1 | For any α > 0, θ* = (c P*, c^{−1} V*), c ≠ 0, with the block structure in Eq. 5, makes TF_θ*(H(z)) equal w_SARSA exactly; f_read takes the last d entries of the last column | Prompt format Eq. 4 |
| Prop 3.1 | Under gradient flow, blocks P_11, P_21, V_11, V_12 and first rows of V_21, V_22 do not affect output and never move; if P_22 = 0 and V̄_22 = 0 at init they stay 0 | Gradient flow on L |
| Thm 3.2 | With loss L(θ) = E_z[(1/2)||TF_θ(H(z)) − w_SARSA(z)||²] and gradient flow dθ/dt = −∇L, if θ_eff(0) is within an explicit distance of M_{I_0} (Eq. 12), the flow stays in a tube T_r(I) around the manifold, L(θ_eff(t)) ≤ e^{−2 μ_r t} L(θ_eff(0)), and θ_eff(t) converges to a point of M_I at rate e^{−μ_r t} | Assumption 1; initialization P_22 = 0, V̄_22 = 0; local only |
| Thm 3.3 | Analogous exact construction for batch actor-critic, prompt Eq. 13 with D = 3d + 2m + 2, readout of the last d + m entries | Prompt format Eq. 13 |

Constants in Thm 3.2: B_Σ = 2B_φ² + B_r², C_Q = (1/2) B_Σ B_w̃, m_0 = (1 − ρ) min{c_+^{−2} α² κ_R κ_w̃, c_−² κ_q}, M_0 = B_Σ² B_w̃² (c_−^{−2} α² + 2 c_+²), tube radius r < sqrt(m_0)/(3 C_Q), μ_r = (m_0 − 3 C_Q sqrt(m_0) r)² / (sqrt(M_0) + C_Q r)². The convergence analysis for actor-critic is stated to be analogous and omitted [ICRL §3.3].

## Algorithm: IC-SARSA transformer training [ICRL Alg 1]

1. For each of K training MDPs: sample M^(k) and a feature map φ^(k); random initial w^(0) and s_0.
2. For t = 0, ..., T − 1:
   a. Act ε-greedy w.r.t. Q = w^(t)T φ for n steps to get a window.
   b. Build H^(t) from the window and w^(t); transformer output w^(t+1) = TF_θ(H^(t)).
   c. Teacher target w_SARSA^(t+1) from Eq. 2.
   d. Loss (1/2)||w^(t+1) − w_SARSA^(t+1)||²; gradient step on θ.
   e. Continue from s_n; set w^(t+1) to the teacher's w_SARSA (teacher forcing).
3. Carry θ to the next MDP. Alg 2 (App B) does the same for actor-critic.
Deployment: freeze θ and loop: prompt with the latest window and parameters, get new parameters, act with them, collect the next window [ICRL §4.2].

Experiments: random tabular MDPs with 9 states, 4 actions, γ = 0.5, Dirichlet transitions, rewards Unif(−1, 1), windows n = 20, 10,000 training MDPs, Adam. Learned P and V match the predicted block structure up to scale; on 100 held-out MDPs the transformer tracks the analytical teacher and approaches the value-iteration oracle; the remaining gap matches the teacher's own linear-approximation gap [ICRL §4].

## Relations to the other papers

- No exploration guarantee. Exploration is ε-greedy (SARSA) or softmax with ε = 0.1 mixing (actor-critic); the paper proves the transformer imitates the update rule, not that the rule explores well. Against Coverage, BE, and Resets, which are about sample-efficient exploration, this paper sits at the representation and optimization layer: it says which update a network can compute in context, while those papers say which update rule finds a good policy with few samples [inference].
- Excitation as a coverage condition. Assumption 1(3) and 1(4) are lower bounds on second-moment matrices of the trajectory data, which is the same flavor as the linear feature-coverage condition E_{d^π}[φφ^T] ⪯ C E_μ[φφ^T] that the Coverage paper proposes as an open direction [Coverage §6.2]. Here the condition is over the training-task distribution and is used for optimization, not for value estimation; the resemblance is structural only [inference].
- Concurrency and seeds. The deployment loop is single-agent. Running many in-context agents that share one prompt would give identical updates; the seed sampling paper's diversity argument suggests injecting per-agent randomness (for example a per-agent prior sample in the prompt) [inference, not studied in either paper].
- Resets and planning. No resets or simulator access are used; data is on-policy windows from a single stream. A natural project question is whether a transformer could implement a GOLF or SimGolf style update in context; this paper only covers single-step linear updates, and GOLF's optimistic argmax over a confidence set is a much harder computation [inference].

## Connections

Directly about Q-learning-family updates (SARSA with linear function approximation) and actor-critic (policy gradient), viewed as functions a transformer layer can compute; also an example of imitating a teacher algorithm, related to the learning-from-demonstrations unit.
