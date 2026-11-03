# SAPG: Split and Aggregate Policy Gradients

Jayesh Singla, Ananye Agarwal, Deepak Pathak (Carnegie Mellon). ICML 2024 (oral). arXiv:2407.20230 (v1 used here; the arXiv PDF itself does not print the venue).

## The idea

With GPU simulators running tens of thousands of environments, PPO stops improving as the batch grows, because one Gaussian policy sends most environments near the same mean action and the extra data is redundant [SAPG §1, §4, Fig. 2]. SAPG splits the environments into M blocks, each run by a different policy (one leader, M−1 followers that share a backbone but have their own latent vector and optionally their own entropy bonus), and updates the leader with its own on-policy data plus importance-weighted, PPO-clipped off-policy data from the followers [§4.1-4.6, Algorithm 1]. On the AllegroKuka manipulation tasks, where PPO barely makes progress at 24,576 environments, SAPG gets the best scores in the paper [Fig. 5 caption, Table 1].

## Notation

| Symbol | Meaning |
|---|---|
| N | Total number of parallel environments (24,576 in the experiments) [§5] |
| M | Number of policies (6 in the experiments); each runs a block of N/M environments [§5, Fig. 3] |
| π₁ | Leader policy; π₂, ..., π_M are followers [§4.3] |
| π_{i,old} | Policy i at data-collection time (the behavior policy for its own block) |
| D_j | Rollout buffer collected by policy j on its block of environments [Algorithm 1] |
| X ⊆ {1, ..., M} | Set of policies whose data is used off-policy when updating policy i. Leader: X = {2, ..., M}. Followers: X = ∅ [§4.3] |
| B_θ, C_ψ | Shared actor backbone (params θ) and shared critic backbone (params ψ) [§4.4] |
| φ_j | Learned per-policy latent vector fed to both backbones; φ_j ∈ ℝ³² for AllegroKuka tasks, ℝ¹⁶ for ShadowHand and AllegroHand [§4.4, §5] |
| r_t(π_θ) | PPO probability ratio π_θ(aₜ∣sₜ)/π_old(aₜ∣sₜ) [Eq. 2] |
| ε | PPO clip parameter |
| r_{π_i}(s,a) | Off-policy ratio π_i(s,a)/π_j(s,a) for a sample collected by π_j [Eq. 3] |
| μ | Off-policy correction π_{i,old}(s,a)/π_j(s,a) [Eq. 3] |
| A^{π_{i,old}}(s,a) | Advantage estimate for policy i (hyperparameter tables list GAE-style τ = 0.95) |
| λ | Weight on the off-policy terms in actor and critic losses; λ = 1 [§4.2-4.3] |
| λ_ent (also written σ) | Entropy coefficient for followers, written λ_ent(i−1) for policy i; the leader gets none [§4.5, §5] |
| V_{π_i}(s) | Critic for policy i (shared C_ψ conditioned on φ_i) |

## The method

### 1. Split the environments into blocks, one policy per block

Instead of one PPO policy over all N environments, run M policies π₁, ..., π_M, each on N/M environments [§4, Algorithm 1]. All policies share the actor backbone B_θ and critic backbone C_ψ and differ only through their latent vector φ_j (and, optionally, their entropy bonus and their learnable std vector) [§4.4, App. B note]. Fully separate networks would leave each follower ignorant of the others and prone to bad local optima [§4.4].

### 2. Off-policy PPO update for the leader (importance weighted)

To update π_i with data from π_j, j ∈ X, SAPG uses an off-policy PPO loss (from Meng et al. 2023, their ref. [18]) [§4.1, Eq. 3]:

L_off(π_i; X) = (1/∣X∣) Σ_{j∈X} E_{(s,a)~π_j}[ min( r_{π_i}(s,a), clip(r_{π_i}(s,a), μ(1−ε), μ(1+ε)) ) · A^{π_{i,old}}(s,a) ]

with r_{π_i} = π_i/π_j and μ = π_{i,old}/π_j. When j = i this reduces to ordinary PPO. One way to read the clip interval (derived here, not stated in the paper): r_{π_i}/μ = π_i/π_{i,old}, so the clip still limits how far π_i moves from its own old policy, while the factor π_i/π_j corrects for the data having come from π_j.

The total actor loss is L(π_i) = L_on(π_i) + λ·L_off(π_i; X) [Eq. 4], with λ = 1. Because off-policy gradients are noisier, the off-policy data is subsampled so each minibatch has equal amounts of on-policy and off-policy data [§4.2-4.3].

### 3. Critic targets

On-policy data uses a 3-step return target V^target_on(sₜ) = Σ_{k=t}^{t+2} γ^{k−t} r_k + γ³ V_old(s_{t+3}) [Eq. 5]. Off-policy transitions are treated as 1-step returns, V^target_off(s′ₜ) = rₜ + γ V_old(s′_{t+1}) [Eq. 6]. The critic loss is the squared error to each target, combined as L^critic = L^critic_on + λ·L^critic_off [Eq. 7-9]. The paper's subscripts are inconsistent here (Eq. 6 indexes the bootstrap value by π_j, Eq. 8 by π_i), so check the official code for which φ the bootstrap uses.

### 4. Leader-follower aggregation (asymmetric on purpose)

If every policy trains on everyone's data (symmetric aggregation, §4.2), they converge to the same behavior, data diversity disappears, and the method reduces to vanilla PPO [§4.3, §6.3]. SAPG breaks the symmetry: only the leader (i = 1) receives off-policy data from all followers; each follower trains only on its own data with standard PPO [§4.3, Fig. 3-4].

### 5. Extra diversity from entropy (optional)

Policy i (the (i−1)th follower) adds an entropy term with coefficient written λ_ent(i−1); the leader has none. The notation can be read as a separate coefficient per follower or as λ_ent times (i−1); the text only says followers get different scales, which give them different explore/exploit trade-offs [§4.5]. For these runs each block has its own learnable std vector [App. B note].

### Per-iteration loop [Algorithm 1]

1. For each j = 1..M, collect D_j on environments jN/M to (j+1)N/M with the shared θ, ψ and latent φ_j.
2. Sample ∣D₁∣ transitions from the followers' data ∪_{j=2..M} D_j to form D′₁.
3. L = OffPolicyLoss(D′₁) + OnPolicyLoss(D₁) + Σ_{j=2..M} OnPolicyLoss(D_j).
4. Gradient step on θ and ψ (φ_j only receive gradients from policy j's terms).

### Why it helps at very large environment counts

PPO was designed around roughly 100 parallel environments; above about 10k environments many of them sample nearly identical on-policy data, and PPO's asymptotic performance plateaus as batch size grows [§3, §4, Fig. 2]. Splitting into differently conditioned policies spreads the data, and importance sampling lets the leader learn from all of it, unlike population-based training (DexPBT), where data from worse policies is thrown away [§4]. Diversity is measured two ways: PCA reconstruction error of visited states falls more slowly with more components for SAPG than PPO, and small autoencoder MLPs have higher reconstruction error on SAPG's states, both indicating more diverse state coverage [§6.4, Figs. 7-8].

## Key results

Setup: IsaacGym, N = 24,576 environments, M = 6 policies for SAPG and DexPBT, about 2×10¹⁰ transitions, 48-60 hours on a single GPU per run, 5 seeds, mean ± standard error. PPO baseline scales its batch with the environment count (128 to 24,576 envs). PQL is a parallel DDPG variant with mixed exploration noise [§5, §5.2]. Wall-clock is not compared because runs used different machines; curves are plotted against samples [§5.2].

Performance after 2×10¹⁰ samples [Table 1] (successes per episode for AllegroKuka tasks, episode reward for the two hand tasks):

| Task | PPO | DexPBT | PQL | SAPG (λ_ent = 0) | SAPG (λ_ent = 0.005) |
|---|---|---|---|---|---|
| Allegro Hand | 1.01e4 ± 6.31e2 | 7.28e3 ± 1.24e3 | 1.01e4 ± 5.28e2 | 1.23e4 ± 3.29e2 | 9.14e3 ± 8.38e2 |
| Shadow Hand | 1.07e4 ± 4.90e2 | 1.01e4 ± 1.80e2 | 1.28e4 ± 1.25e2 | 1.17e4 ± 2.64e2 | 1.28e4 ± 2.80e2 |
| Regrasping | 1.25 ± 1.15 | 31.9 ± 2.26 | 2.73 ± 0.02 | 35.7 ± 1.46 | 33.4 ± 2.25 |
| Throw | 16.8 ± 0.48 | 19.2 ± 1.07 | 2.62 ± 0.08 | 23.7 ± 0.74 | 18.7 ± 0.43 |
| Reorientation | 2.85 ± 0.05 | 23.2 ± 4.86 | 1.66 ± 0.11 | 33.2 ± 4.20 | 38.6 ± 0.63 |
| Two Arms Reorientation | 1.73 ± 0.51 | 14.46 ± 2.91 | not reported | not reported | 28.58 ± 1.55 |

- SAPG is 12-66% better than DexPBT on Regrasping, Throw and Reorientation (66% on Reorientation) and more than twice DexPBT on Two Arms Reorientation, even though DexPBT mutates reward scales to optimize success directly [§6.1].
- The text says PPO and PQL learn no useful behavior on the AllegroKuka tasks [§6.1]; Table 1 bears this out except for PPO on Throw (16.8 vs SAPG's 23.7).
- On the easier hand tasks PQL is more sample-efficient early, but SAPG beats it by 21% on Allegro Hand and matches it on Shadow Hand [§6.2].
- Ablations [§6.3, Fig. 6]: symmetric aggregation is much worse everywhere; removing the off-policy term is much worse; using all follower data without subsampling ("high off-policy ratio") is worse on Allegro Hand and Shadow Hand, slightly worse on Regrasping and Throw, and more sample-efficient but lower asymptotically on Reorientation; entropy coefficient 5e-3 is up to 16.5% better on Reorientation.

## Hyperparameters [App. B, Tables 2-4]

| | AllegroKuka tasks | Shadow Hand | Allegro Hand |
|---|---|---|---|
| Policy network | MLP 768-512-256 (ELU) into a 1-layer LSTM with 768 units | MLP 512-512-256-128 (ELU) | MLP 512-256-128 (ELU) |
| Discount γ | 0.99 | 0.99 | 0.99 |
| τ (GAE λ) | 0.95 | 0.95 | 0.95 |
| Learning rate | 1e-4 | 5e-4 | 5e-4 |
| KL threshold for LR update | 0.016 | 0.016 | 0.016 |
| Grad norm clip | 1.0 | 1.0 | 1.0 |
| Entropy coefficient (base) | 0 | 0 | 0 |
| Clip ε | 0.1 | 0.1 | 0.2 |
| Mini-batch size | num envs · 4 | num envs · 4 | num envs · 4 |
| Critic coefficient λ′ | 4.0 | 4.0 | 4.0 |
| Horizon length | 16 | 8 | 8 |
| LSTM sequence length | 16 | n/a | n/a |
| Bounds loss coefficient | 0.0001 | 0.0001 | 0.0001 |
| Mini epochs | 2 | 5 | 5 |

The Gaussian std is a learnable vector independent of the observation [App. B.1]. SAPG-specific: M = 6, λ = 1, λ_ent tuned per task from {0, 0.003, 0.005}: 0 was best for Allegro Hand, Regrasping and Throw; 0.005 for Shadow Hand and Reorientation [§5.2] (Fig. 5's caption also lists Two Arms Reorientation under 0.005).

## Gotchas when reproducing

- Subsample follower data to match the leader's on-policy data size; feeding all of it hurts asymptotic performance [§4.3, §6.3].
- Keep the leader-follower asymmetry; symmetric sharing collapses diversity [§6.3].
- Shared backbones plus per-policy latents φ (32-dim for hard tasks, 16-dim for easy) are part of the method, not a detail [§4.4].
- λ_ent matters per task: 0.005 helps Reorientation and Shadow Hand but drops Allegro Hand from 1.23e4 to 9.14e3 [Table 1].
- Equations 2 and 3 are printed as min(r, clip(r, ...))·A, with the advantage outside the min. Standard PPO is min(r·A, clip(r, ...)·A); these differ when A < 0. Follow the standard form or the official code.
- Rollout length is inconsistent in the paper: §5.2 says 16 steps per environment before each update, while the Shadow Hand and Allegro Hand tables say horizon 8 (AllegroKuka says 16).
- Other inconsistencies: §5 opens with "5 manipulation tasks (3 hard and 2 easy)" but evaluates 6 (4 hard, 2 easy); Table 4's caption says Shadow Hand but sits under the Allegro Hand section B.3; Algorithm 1 passes ψ_j where φ_j is meant.
- Hard tasks use curricula on success tolerance (Regrasping: 7.5 cm down to 1 cm, reduced 10% whenever average successes per episode cross 3) [App. A].
- Runs need about 2×10¹⁰ samples and 48-60 GPU-hours; results are per sample, not wall-clock [§5.2].

## Connections

PPO and its variants: this is PPO plus importance sampling, so it is a direct example of why on-policy data is required (Eq. 1) and how the ratio and clip let you use slightly off-policy data. It also ties to the actor-critic unit through its n-step and 1-step critic targets.
