# Canonical deep RL algorithm papers

Key facts from 23 papers and 2 blog posts, each checked against the paper itself. Every fact ends with a location tag. Section numbers follow the version named in the citation line. Unnumbered sections are named in the tag, for example [DDQN §"Double DQN"].

How these were checked (2026-11-02): for each arXiv paper, the abstract page (arxiv.org/abs/ID) gave the title, authors, comments and journal-ref, and the PDF (arxiv.org/pdf/ID) was read for the facts. Tables that exist only as images (DQN Extended Data Table 1) were read from a rendered page image. The blogs were read from their own pages. A venue is listed only when it was printed on a page that was opened: the PDF header or footer, the arXiv comments field, or a proceedings or publisher index page. "(unverified)" marks anything that was not checked this way.

Symbols used in many sections: s state, a action, r reward, s′ next state, γ discount factor, θ network parameters, θ⁻ or θ′ or θ̄ target-network parameters, π policy, Q action-value function, V state-value function, E expectation, Ê empirical average over a batch.

---

## DQN (2015)

Human-level control through deep reinforcement learning. Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, Joel Veness, et al. (19 authors). Nature 518(7540):529-533, February 2015, doi:10.1038/nature14236. No arXiv version of the Nature paper (the 2013 workshop paper is arXiv 1312.5602, "Playing Atari with Deep Reinforcement Learning", NIPS Deep Learning Workshop 2013). Opened: https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf (DeepMind-hosted PDF of the Nature letter), https://ideas.repec.org/a/nat/nature/v518y2015i7540d10.1038_nature14236.html (volume and pages).

- Loss: L_i(θ_i) = E_{(s,a,r,s′)~U(D)}[(r + γ max_{a′} Q(s′,a′; θ_i⁻) − Q(s,a; θ_i))²]. D is the replay memory, U(D) uniform sampling from it, θ_i the online Q-network parameters at iteration i, θ_i⁻ the target-network parameters, copied from θ_i every C steps and held fixed in between. [DQN main text, unnumbered loss; Methods; Algorithm 1]
- Terminal handling: target y_j = r_j if the episode ends at step j+1, else r_j + γ max_{a′} Q̂(φ_{j+1}, a′; θ⁻), where φ is the preprocessed frame stack. [DQN Algorithm 1]
- The TD error r + γ max Q̂ − Q is clipped to [−1, 1], which the paper says is the same as an absolute-value loss outside (−1, 1) (a Huber-style loss). Rewards are clipped to [−1, 1] (positive to 1, negative to −1, zero unchanged) during training only. [DQN Methods "Training details", "Training algorithm"]
- Extended Data Table 1: minibatch 32; replay memory 1,000,000 frames; agent history length 4; target network update frequency C = 10,000 parameter updates; γ = 0.99; action repeat 4; update frequency 4 (actions between SGD updates); RMSProp with learning rate 0.00025, gradient momentum 0.95, squared gradient momentum 0.95, min squared gradient 0.01; ε-greedy ε from 1 to 0.1 over the first 1,000,000 frames; replay start size 50,000 (random policy first); no-op max 30. Values came from an informal search on 5 games, no grid search. [DQN Ext. Data Tab. 1]
- Result: one architecture and one hyperparameter set across 49 Atari games; more than 75% of the human tester's score on 29 games. Extended Data Table 3 ablates replay and the target network (each helps). [DQN main text; Ext. Data Tab. 3]

## Double DQN (2016)

Deep Reinforcement Learning with Double Q-learning. Hado van Hasselt, Arthur Guez, David Silver. AAAI 2016 (arXiv comments field: "AAAI 2016"; the PDF carries the AAAI copyright line). arXiv 1509.06461 (v3). Opened: https://arxiv.org/abs/1509.06461, https://arxiv.org/pdf/1509.06461.

- Target: Y_t^DoubleDQN = R_{t+1} + γ Q(S_{t+1}, argmax_a Q(S_{t+1}, a; θ_t); θ_t⁻). The online network θ_t picks the action, the target network θ_t⁻ scores it. Everything else is unchanged from DQN. Compare DQN's Y_t^DQN = R_{t+1} + γ max_a Q(S_{t+1}, a; θ_t⁻), where the same network both picks and scores. [DDQN §"Double DQN", unnumbered eq. after Eq. 4]
- Theorem 1: if all true action values at s equal V*(s) and the estimates Q_t are unbiased on average but have mean squared error C > 0 over m ≥ 2 actions, then max_a Q_t(s,a) ≥ V*(s) + sqrt(C/(m−1)). The bound is tight, and the matching lower bound for Double Q-learning's absolute error is zero. So estimation error of any source biases max-based targets upward. [DDQN §"Overoptimism due to estimation errors", Thm. 1]
- Same DQN hyperparameters, only the target changed: median normalized score over 49 games (no-op starts) rises from 93.5% to 114.7%, mean from 241.1% to 330.3%. [DDQN Tab. 1]
- "Tuned" Double DQN: target copy period 10,000 → 30,000 frames, training ε 0.1 → 0.01, evaluation ε 0.001, one shared bias across action values. Human-start median 47.5% (DQN) → 88.4% (Double DQN) → 116.7% (tuned). [DDQN §"Robustness to Human starts", Tab. 2]

## Prioritized Experience Replay (2016)

Prioritized Experience Replay. Tom Schaul, John Quan, Ioannis Antonoglou, David Silver. ICLR 2016 (PDF header "Published as a conference paper at ICLR 2016"). arXiv 1511.05952 (v4). Opened: https://arxiv.org/abs/1511.05952, https://arxiv.org/pdf/1511.05952.

- Sampling probability: P(i) = p_i^α / Σ_k p_k^α, with p_i > 0 the priority of transition i and α the amount of prioritization (α = 0 is uniform). Proportional variant: p_i = |δ_i| + ε, where δ_i is the TD error and ε a small constant. Rank-based variant: p_i = 1/rank(i), rank by |δ_i|. [PER §3.3 Eq. 1]
- Importance-sampling correction: w_i = (1/N · 1/P(i))^β, N the replay size, always divided by max_i w_i so weights only scale updates down. Use w_i δ_i in place of δ_i. β is annealed linearly from β₀ to 1 by the end of training. New transitions enter with the current maximum priority. [PER §3.4; Algorithm 1 lines 6, 10]
- Chosen values: rank-based α = 0.7, β₀ = 0.5; proportional α = 0.6, β₀ = 0.4; step size η reduced 4× from the DQN baseline (0.00025/4) because sampled TD errors are larger. [PER §4; App. B.2.2 Tab. 3]
- Proportional sampling uses a sum-tree (each node holds the sum of its children). Rank-based uses k equal-probability segments, one sample per segment. [PER §3.3; App. B.2.1]
- With Double DQN as the base, median normalized score over 57 games goes from 111% to 128% (proportional), and learning is about twice as fast. [PER §4 Tab. 1, Fig. 4]

## DDPG (2016)

Continuous control with deep reinforcement learning. Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, Daan Wierstra. ICLR 2016 (PDF header "Published as a conference paper at ICLR 2016"). arXiv 1509.02971 (v6). Opened: https://arxiv.org/abs/1509.02971, https://arxiv.org/pdf/1509.02971.

- Critic target y_i = r_i + γ Q′(s_{i+1}, μ′(s_{i+1}|θ^{μ′}) | θ^{Q′}); critic loss L = (1/N) Σ_i (y_i − Q(s_i,a_i|θ^Q))². μ(s|θ^μ) is the deterministic actor, Q′ and μ′ are target copies. Actor gradient ∇_{θ^μ}J ≈ (1/N) Σ_i ∇_a Q(s,a|θ^Q)|_{s=s_i, a=μ(s_i)} ∇_{θ^μ} μ(s|θ^μ)|_{s_i}. [DDPG §3 Eq. 6; Algorithm 1]
- Soft target updates θ′ ← τθ + (1−τ)θ′ with τ ≪ 1, for both actor and critic. Both target networks were needed to train the critic without divergence. [DDPG §3]
- Exploration: μ′(s_t) = μ(s_t|θ^μ_t) + N, with N an Ornstein-Uhlenbeck process (θ = 0.15, σ = 0.2) for temporally correlated noise. [DDPG §3 Eq. 7; §7]
- Hyperparameters: Adam, actor lr 10⁻⁴, critic lr 10⁻³; critic L2 weight decay 10⁻²; γ = 0.99; τ = 0.001; 400 and 300 ReLU hidden units, tanh actor output; actions enter Q at the second hidden layer; final-layer init uniform [−3×10⁻³, 3×10⁻³] (low-dim); minibatch 64 (low-dim), 16 (pixels); replay 10⁶. [DDPG §7 "Experiment details"]
- Batch normalization on the state input and on all actor layers and the critic layers before the action input, so one setting works across tasks with different observation units. [DDPG §3]

## TD3 (2018)

Addressing Function Approximation Error in Actor-Critic Methods. Scott Fujimoto, Herke van Hoof, David Meger. ICML 2018 (PDF footer "Proceedings of the 35th International Conference on Machine Learning"; arXiv comments "Accepted at ICML 2018"). arXiv 1802.09477 (v3). Opened: https://arxiv.org/abs/1802.09477, https://arxiv.org/pdf/1802.09477.

- Full critic target: y = r + γ min_{i=1,2} Q_{θ′_i}(s′, π_{φ′}(s′) + ε), with ε ~ clip(N(0, σ̃), −c, c). Q_{θ1}, Q_{θ2} are two critics, θ′_i their targets, π_{φ′} the target actor. Both critics regress to the same y. [TD3 §6 Eq. 15; Algorithm 1]
  - Clipped double-Q (the min) limits overestimation. [TD3 §4.2 Eq. 10]
  - Target policy smoothing (the clipped noise ε) fits the value of a small region around the target action. [TD3 §5.3 Eq. 14]
  - Delayed policy updates: the actor and all target networks update once every d critic updates. The actor follows ∇_φ J = N⁻¹ Σ ∇_a Q_{θ1}(s,a)|_{a=π_φ(s)} ∇_φ π_φ(s), using critic 1 only. [TD3 §5.2; Algorithm 1]
- Defaults: σ̃ = 0.2, c = 0.5, d = 2, τ = 0.005; Adam lr 10⁻³ for actor and critic; batch 100 sampled uniformly from the full history; 400-300 ReLU networks, tanh actor output; the critic takes (s, a) at the first layer; exploration noise N(0, 0.1), uncorrelated (OU gave no benefit); purely random actions for the first 10,000 steps (HalfCheetah, Ant) or 1,000 steps (others); 1M steps, 10 seeds. [TD3 §6.1]
- Table 3 ("Ours" column, their re-tuned DDPG, same settings as TD3): γ = 0.99, no observation normalization, no gradient clipping, reward scale 1.0, 1 iteration per time step. [TD3 App. C Tab. 3]
- Time limits: the target uses y = r only for real terminal states, not for episodes cut off at the max horizon. Smoothed target actions are also clipped to the valid action range. [TD3 App. D]

## SAC (2018)

Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor. Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, Sergey Levine. ICML 2018 (arXiv comments "ICML 2018"). arXiv 1801.01290 (v2). Opened: https://arxiv.org/abs/1801.01290, https://arxiv.org/pdf/1801.01290.

- Objective: J(π) = Σ_t E_{(s_t,a_t)~ρ_π}[r(s_t,a_t) + α H(π(·|s_t))]. ρ_π is the state-action marginal under π, H is entropy, α is the temperature (α → 0 gives standard RL). This version folds α into the reward scale. [SAC §3.2 Eq. 1]
- Soft Bellman backup: T^π Q(s_t,a_t) = r + γ E_{s_{t+1}}[V(s_{t+1})], with the soft value V(s_t) = E_{a_t~π}[Q(s_t,a_t) − log π(a_t|s_t)]. [SAC §4.1 Eq. 2-3]
- Losses in this version: a separate value net V_ψ with J_V(ψ) = E[½(V_ψ(s_t) − E_{a_t~π_φ}[Q_θ(s_t,a_t) − log π_φ(a_t|s_t)])²] [Eq. 5]; Q target Q̂(s_t,a_t) = r + γ E[V_ψ̄(s_{t+1})], where ψ̄ is an EMA of ψ [Eq. 7-8]; reparameterized policy loss J_π(φ) = E_{s_t~D, ε_t~N}[log π_φ(f_φ(ε_t;s_t)|s_t) − Q_θ(s_t, f_φ(ε_t;s_t))], with a_t = f_φ(ε_t; s_t) the noise-to-action map [Eq. 11-12]. Two Q-functions; the min is used in the value and policy gradients. [SAC §4.2; Algorithm 1]
- Squashing: a = tanh(u) with u Gaussian, and log π(a|s) = log μ(u|s) − Σ_i log(1 − tanh²(u_i)). [SAC App. C Eq. 21]
- Table 1: Adam, lr 3·10⁻⁴, γ = 0.99, replay 10⁶, 2 hidden layers of 256 (all networks), batch 256, ReLU, τ = 0.005, 1 gradient step per env step. Reward scale is tuned per task (Table 2: 5 for Hopper, Walker, HalfCheetah, Ant; 20 for Humanoid-v1; 10 for rllab Humanoid). [SAC App. D Tab. 1-2]

## SAC Algorithms and Applications (2018)

Soft Actor-Critic Algorithms and Applications. Tuomas Haarnoja, Aurick Zhou, Kristian Hartikainen, George Tucker, Sehoon Ha, Jie Tan, Vikash Kumar, Henry Zhu, Abhishek Gupta, Pieter Abbeel, Sergey Levine. Venue unverified (arXiv preprint; no venue printed on the abstract page or PDF). arXiv 1812.05905 (v2). Opened: https://arxiv.org/abs/1812.05905, https://arxiv.org/pdf/1812.05905.

- Drops the separate value network (footnote: "later found it to be unnecessary"). Soft value V(s_t) = E_{a_t~π}[Q(s_t,a_t) − α log π(a_t|s_t)]. Q loss J_Q(θ) = E[½(Q_θ(s_t,a_t) − (r + γ E_{s_{t+1}}[V_θ̄(s_{t+1})]))²], θ̄ an EMA of θ, estimated with one sample a_{t+1} ~ π. Two soft Q-functions, use the min. [SAC-Apps §3.2 Eq. 3; §4.2 Eq. 5-6 and footnote 1; §6]
- Policy loss: J_π(φ) = E_{s_t~D, ε_t~N}[α log π_φ(f_φ(ε_t;s_t)|s_t) − Q_θ(s_t, f_φ(ε_t;s_t))]. [SAC-Apps §4.2 Eq. 9]
- Learned temperature. Constrained problem: maximize expected return subject to E[−log π_t(a_t|s_t)] ≥ H̄ for all t, where H̄ is the minimum expected entropy [Eq. 11]. Temperature loss, one gradient step per update: J(α) = E_{a_t~π_t}[−α log π_t(a_t|s_t) − α H̄]. [SAC-Apps §5 Eq. 11, 17; §6 Eq. 18; Algorithm 1]
- Table 1: entropy target H̄ = −dim(A) (e.g. −6 for HalfCheetah); otherwise the SAC defaults (Adam 3·10⁻⁴, γ 0.99, replay 10⁶, 2×256 ReLU, batch 256, τ 0.005, 1 gradient step). [SAC-Apps App. D Tab. 1]
- Learned α matches or beats fixed α across the benchmarks and "effectively eliminates the need for tuning the temperature". Real Minitaur quadruped learned to walk from 160k steps (about 400 episodes, about 2 hours), with state = last 5 observations and actions to handle latency. [SAC-Apps §7.1; §7.2]

## PPO (2017)

Proximal Policy Optimization Algorithms. John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov. Venue unverified (arXiv preprint; the abstract page has no comments or journal-ref field). arXiv 1707.06347 (v2). Opened: https://arxiv.org/abs/1707.06347, https://arxiv.org/pdf/1707.06347.

- Ratio r_t(θ) = π_θ(a_t|s_t) / π_θold(a_t|s_t), so r_t(θ_old) = 1. Clipped surrogate: L^CLIP(θ) = Ê_t[min(r_t(θ)Â_t, clip(r_t(θ), 1−ε, 1+ε)Â_t)], where Â_t is the advantage estimate and ε the clip parameter ("say, ε = 0.2"). The min makes it a pessimistic bound: ratio changes are ignored only when they would improve the objective. [PPO §3 Eq. 6-7, Fig. 1]
- Combined loss for shared networks: L^{CLIP+VF+S}_t(θ) = Ê_t[L^CLIP_t(θ) − c₁ L^VF_t(θ) + c₂ S[π_θ](s_t)], with L^VF_t = (V_θ(s_t) − V_t^targ)², S an entropy bonus, c₁ and c₂ coefficients. [PPO §5 Eq. 9]
- Advantages: truncated GAE over a length-T segment, Â_t = δ_t + (γλ)δ_{t+1} + ... + (γλ)^{T−t−1} δ_{T−1}, with δ_t = r_t + γV(s_{t+1}) − V(s_t). The PDF prints the last exponent as T−t+1 (see Disagreements). Algorithm 1: N parallel actors each collect T steps, then K epochs of minibatch SGD (minibatch M ≤ NT) on the NT samples. [PPO §5 Eq. 11-12; Algorithm 1]
- Adaptive KL alternative: after each update compute d = Ê_t[KL[π_θold(·|s_t), π_θ(·|s_t)]]; if d < d_targ/1.5 then β ← β/2; if d > 1.5·d_targ then β ← 2β. It scored worse than clipping. [PPO §4 Eq. 8; Tab. 1]
- MuJoCo (1M steps) hyperparameters: horizon T = 2048, Adam step size 3×10⁻⁴, 10 epochs, minibatch 64, γ = 0.99, λ = 0.95. Clip ε is not in the table (it was swept). Policy: MLP with 2×64 tanh layers outputting a Gaussian mean, "variable standard deviations"; no parameter sharing, no entropy bonus. [PPO Tab. 3; §6.1]
- Clip sweep (normalized score, 7 tasks × 3 seeds): no clipping −0.39; ε = 0.1: 0.76; ε = 0.2: 0.82 (best); ε = 0.3: 0.70; best adaptive KL 0.74; best fixed KL 0.72. [PPO §6.1 Tab. 1]
- Atari: T = 128, step size 2.5×10⁻⁴·α, 3 epochs, minibatch 32×8, 8 actors, ε = 0.1·α, c₁ = 1, c₂ = 0.01, with α annealed linearly from 1 to 0. Roboschool: 32 or 128 actors, minibatch 4096, log std annealed from −0.7 to −1.6. [PPO Tab. 4-5]

## GAE (2016)

High-Dimensional Continuous Control Using Generalized Advantage Estimation. John Schulman, Philipp Moritz, Sergey Levine, Michael I. Jordan, Pieter Abbeel. ICLR 2016 (PDF header "Published as a conference paper at ICLR 2016"). arXiv 1506.02438 (v6). Opened: https://arxiv.org/abs/1506.02438, https://arxiv.org/pdf/1506.02438.

- TD residual δ_t^V = r_t + γV(s_{t+1}) − V(s_t). Estimator: Â_t^{GAE(γ,λ)} = Σ_{l=0}^∞ (γλ)^l δ_{t+l}^V, the exponentially weighted (weight λ) average of k-step advantage estimates. [GAE §3 Eq. 16]
- λ = 0: Â_t = δ_t (low variance, biased unless V is exact). λ = 1: Â_t = Σ_l γ^l r_{t+l} − V(s_t) (unbiased for the γ-discounted problem whatever V is, high variance). λ in between trades the two. γ is treated as a variance-reduction parameter for an undiscounted problem. [GAE §3 Eq. 17-18; §2]
- The best λ is much lower than the best γ, because λ adds far less bias than γ when V is accurate. [GAE §3]
- Value fitting with a trust region: minimize Σ_n ‖V_φ(s_n) − V̂_n‖² subject to (1/N) Σ_n ‖V_φ(s_n) − V_φold(s_n)‖²/(2σ²) ≤ ε, with σ² the mean squared error of the old fit. Solved approximately with conjugate gradient. [GAE §5 Eq. 29-30]
- Found ranges: cart-pole γ ∈ [0.96, 0.99], λ ∈ [0.92, 0.99]; 3D biped γ ∈ [0.99, 0.995], λ ∈ [0.96, 0.99]; quadruped best with a value function and λ = 0.96 (γ fixed at 0.995). Conclusion: λ in [0.9, 0.99] usually works best. [GAE §6.3.1, §6.3.2, §6.3.3, §7]

## What Matters in On-Policy RL (2021)

What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study. Marcin Andrychowicz, Anton Raichuk, Piotr Stańczyk, Manu Orsini, Sertan Girgin, Raphael Marinier, Léonard Hussenot, Matthieu Geist, Olivier Pietquin, Marcin Michalski, Sylvain Gelly, Olivier Bachem. arXiv 2006.05990 (v1, June 2020, the only arXiv version; no venue on the abstract page). Published at ICLR 2021 as "What Matters for On-Policy Deep Actor-Critic Methods? A Large-Scale Study" (seen on https://iclr.cc/virtual/2021/poster/2527). The facts below come from arXiv v1. The ICLR camera-ready was not opened (unverified whether its numbers differ). Opened: https://arxiv.org/abs/2006.05990, https://arxiv.org/pdf/2006.05990, https://iclr.cc/virtual/2021/poster/2527.

- Setup: more than 250,000 agents on 5 MuJoCo tasks with a PPO-like base configuration at 256 parallel envs. Choices are labeled C1, C2, ... (described in App. B). Reported scores are the 95th percentile over random configurations, a proxy for "what you get after about 20 random-search trials". [WM §2; §5]
- Policy init: initialize the last policy layer with 100× smaller weights, so the initial action distribution is centered at 0 and independent of the observation (this alone gives +66% on Humanoid). Initial action std of about 0.5 was best on all tasks except Hopper. Use softplus for the std with a negative offset, and tune the offset. State-dependent vs global std made no clear difference. tanh squashing beat clipping slightly. [WM §3.2, Fig. 2; recommendation in §3.2]
- Normalization: always normalize observations (needed for good results on all tasks but Hopper). Value-target normalization matters a lot but can help or hurt (it hurt Walker2d), so test it. Per-minibatch advantage normalization "seems not to affect the performance too much". Gradient clipping helps a little. [WM §3.3]
- Advantages and value loss: GAE λ = 0.9. PPO-style value loss clipping hurts at every threshold. Huber value loss is worse than MSE. [WM §3.4]
- Training setup and other choices: multiple passes over data; shuffle transitions and recompute advantages once per pass; more envs and bigger batches cut wall-clock time but can hurt sample efficiency. γ is among the most important hyperparameters (start at 0.99). Adam with β₁ = 0.9, lr 3×10⁻⁴. Clip ε: start at 0.25. Separate policy and value networks; wide value MLP; tanh activations. No regularizer (entropy, KL) helped except on HalfCheetah. [WM §3.1, §3.5-3.8]

## Implementation Matters (2020)

Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO. Logan Engstrom, Andrew Ilyas, Shibani Santurkar, Dimitris Tsipras, Firdaus Janoos, Larry Rudolph, Aleksander Madry. ICLR 2020 (PDF header "Published as a conference paper at ICLR 2020"; arXiv comments "ICLR 2020 version"). arXiv 2005.12729 (v1). Opened: https://arxiv.org/abs/2005.12729, https://arxiv.org/pdf/2005.12729.

- The nine "code-level optimizations" in OpenAI baselines PPO2 that are not in the PPO paper [Impl §3]:
  1. Value function clipping: L^V = max[(V_θt − V_targ)², (clip(V_θt, V_θt−1 − ε, V_θt−1 + ε) − V_targ)²], with ε the policy clip value.
  2. Reward scaling: divide rewards by the std of a rolling discounted sum of rewards (no mean subtraction) [App. A.2 Algorithm 1].
  3. Orthogonal initialization with per-layer scaling.
  4. Adam learning-rate annealing.
  5. Reward clipping (usually [−5, 5] or [−10, 10]).
  6. Observation normalization to zero mean, unit variance.
  7. Observation clipping (usually [−10, 10]).
  8. tanh activations.
  9. Global gradient-norm clipping at 0.5.
- Ablation of the first four (Humanoid, Walker2d): reward normalization, Adam annealing and network init each significantly change the reward landscape and were needed for the best PPO reward. [Impl Fig. 1 caption]
- Code-level optimizations matter more than PPO vs TRPO. Hopper-v2: PPO 2513, PPO-M (no optimizations) 2142, TRPO 2043, TRPO+ (TRPO with the optimizations) 2466. Average code-level improvement (ACLI) exceeds average algorithmic improvement (AAI) on all three tasks. [Impl §5 Tab. 2]
- PPO-NoClip (optimizations on, no ratio clipping) is uniformly better than PPO-M and often matches PPO. Ratio clipping does not keep the max ratio inside [1−ε, 1+ε]. PPO's KL trust region comes largely from the optimizations and the optimizer. [Impl §4 Fig. 2; §5 Tab. 3]

## The 37 Implementation Details of PPO (2022)

The 37 Implementation Details of Proximal Policy Optimization. Shengyi Huang, Rousslan Fernand Julien Dossa, Antonin Raffin, Anssi Kanervisto, Weixun Wang. ICLR Blog Track, posted 25 Mar 2022 (page header "The ICLR Blog Track"). No arXiv id. Opened: https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/.

- The reference is openai/baselines ppo2 at commit ea25b9e. The 37 details are 13 core, 9 Atari, 9 continuous-action, 5 LSTM and 1 MultiDiscrete. [37D §Background, §Reproducing]
- Core details, as implemented: a vectorized env with fixed-length segments (N envs × M steps) that bootstraps across segment ends; orthogonal init (hidden scale √2, policy output 0.01, value output 1, biases 0); Adam ε = 1e-5; learning rate annealed linearly to 0 (3e-4 for MuJoCo); GAE with returns = advantages + values; shuffled minibatches; advantage normalization per minibatch; clipped surrogate; value loss clipping; loss = policy_loss − c₂·entropy + c₁·value_loss with one optimizer; global grad-norm clip 0.5. [37D §13 core details]
- The 9 continuous-action details: Gaussian actions; state-independent log std initialized to 0; independent action dimensions (log-probs summed); separate 2×64 tanh policy and value MLPs; sample unbounded, clip to bounds for the env, store the unclipped action; observation normalization (running mean and std) and observation clipping to [−10, 10]; reward scaling by the std of a rolling discounted return; reward clipping to [−10, 10]. MuJoCo defaults: nsteps 2048, nminibatches 32, λ 0.95, γ 0.99, 10 epochs, ent_coef 0, lr 3e-4 annealed, clip 0.2, num_envs 1. The blog does not rank these 9 by importance. It cross-references Andrychowicz and Engstrom for each one. [37D §9 details for continuous action domains]
- Truncation: ppo2 does not bootstrap the value at time-limit truncations, although "we (intuitively) should". The blog kept this for fidelity. [37D §13 core details, GAE item]
- Debug checks: the ratio must be exactly 1 on the first minibatch of the first epoch; approx_kl usually stays below 0.02 (higher usually means a bug); approxkl in ppo2 is the k1 estimator, and the blog points to Schulman's k3 as unbiased with lower variance. [37D §Recommendations; §13 core details, debug variables]
- On num_envs: the blog disputes judging large N by sample efficiency (Andrychowicz) and argues for wall-clock time, citing Brax PPO with N = 2048, M = 20, and Isaac Gym at N = 4096. [37D §13 core details, vectorized architecture; §Is asynchronous PPO better?]

## Deep RL that Matters (2018)

Deep Reinforcement Learning that Matters. Peter Henderson, Riashat Islam, Philip Bachman, Joelle Pineau, Doina Precup, David Meger. AAAI 2018 (arXiv comments "Accepted to the Thirthy-Second AAAI Conference On Artificial Intelligence (AAAI), 2018"). arXiv 1709.06560 (v3). Opened: https://arxiv.org/abs/1709.06560, https://arxiv.org/pdf/1709.06560.

- Seed finding: 10 TRPO runs on HalfCheetah-v1 with one hyperparameter setting, split into two groups of 5 seeds. The two group means give statistically different learning curves (2-sample t-test t = −9.0916, p = 0.0016). The paper gives no specific number of seeds to use and suggests power analysis to choose one. [Henderson §"Random Seeds and Trials", Fig. 5]
- Codebases: different public TRPO and DDPG implementations with matched hyperparameters give very different results. [Henderson §"Codebases", Tab. 1-2, Fig. 6]
- Reward scale has a large but inconsistent effect across environments, and layer normalization changes how it acts. [Henderson §"Reward Scale", Fig. 3]
- Network architecture and activation choices change TRPO, PPO and DDPG results significantly. Defaults in this study: (64, 64, tanh) for TRPO and PPO policies, (64, 64, ReLU) for DDPG. [Henderson §"Experimental Analysis", §"Network Architecture"]
- Recommendations: run many seeds; do proper significance testing (bootstrap confidence intervals, t-test, KS test, power analysis); match baseline implementations to the original reported results; report all hyperparameters and implementation details. [Henderson §"Reporting Evaluation Metrics", §"Discussion and Conclusion"]

## Statistical Precipice (2021)

Deep Reinforcement Learning at the Edge of the Statistical Precipice. Rishabh Agarwal, Max Schwarzer, Pablo Samuel Castro, Aaron Courville, Marc G. Bellemare. NeurIPS 2021 (PDF footer "35th Conference on Neural Information Processing Systems (NeurIPS 2021)"; arXiv comments "Outstanding Paper Award at NeurIPS 2021"). arXiv 2108.13264 (v4). Opened: https://arxiv.org/abs/2108.13264, https://arxiv.org/pdf/2108.13264.

- Report interval estimates, not point estimates. Stratified bootstrap: resample runs with replacement independently within each task (N runs × M tasks), recompute the aggregate statistic, and repeat many times. Percentile CIs had good coverage with as few as N = 10 runs. [Agarwal §4.1; Tab. 1]
- IQM (interquartile mean, a 25% trimmed mean): drop the bottom and top 25% of all N·M runs pooled across tasks and average the middle 50%. It is robust to outliers and more statistically efficient than the median. Median CIs stay wide even with 50 runs. [Agarwal §4.3; Tab. 1; Fig. 2 caption]
- Performance profiles: plot the fraction of runs scoring above τ for each normalized score τ, which shows the score distribution across tasks and runs. Also report probability of improvement and optimality gap. [Agarwal §4.2; Tab. 1]
- The tools are designed for 3-10 runs per task. Library: rliable (github.com/google-research/rliable). [Agarwal §1; Tab. 1]

## DAgger (2011)

A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning. Stéphane Ross, Geoffrey J. Gordon, J. Andrew Bagnell. AISTATS 2011 (PDF: "Appearing in Proceedings of the 14th International Conference on Artificial Intelligence and Statistics (AISTATS) 2011"). arXiv 1011.0686 (v3). Opened: https://arxiv.org/abs/1011.0686, https://arxiv.org/pdf/1011.0686.

- Notation: T is the horizon; C(s,a) ∈ [0,1] the task cost; J(π) = Σ_{t=1}^T E_{s~d_π^t}[C_π(s)] the total T-step cost; d_π the average state distribution of π over T steps; ℓ(s,π) the surrogate loss (e.g. 0-1 loss against the expert π*). [DAgger §2]
- Behavior cloning bound: if E_{s~d_π*}[ℓ(s,π)] = ε (error measured on the expert's states), then J(π) ≤ J(π*) + T²ε, and this is tight (there are problems where the cost grows as T²ε). [DAgger §2.1 Thm. 2.1]
- Algorithm: at iteration i, roll out the mixture π_i = β_i π* + (1−β_i) π̂_i, label every visited state with the expert's action, aggregate all data D ← D ∪ D_i, and train π̂_{i+1} on all of D. Typically β₁ = 1. The parameter-free choice β_i = I(i = 1) (expert only in the first iteration) "often performs best in practice". [DAgger §3 Algorithm 3.1]
- Guarantee: with ε_N the best-in-hindsight loss on the learner's own state distributions and u a bound on how much one non-expert action raises the expert's cost-to-go, Theorem 3.2 gives J(π̂) ≤ J(π*) + uTε_N + O(1) for some π̂ in the sequence, when N is Õ(uT). With u = O(1) this is linear in T, against behavior cloning's T². [DAgger §3 Thm. 3.1-3.2; Thm. 2.2 for u]
- The finite-sample version needs N = O(u²T² log(1/δ)) iterations with m = O(1) trajectories each, and holds with probability 1−δ. [DAgger §3 Thm. 3.4]

## GAIL (2016)

Generative Adversarial Imitation Learning. Jonathan Ho, Stefano Ermon. NIPS 2016, "Advances in Neural Information Processing Systems 29 (NIPS 2016)" (proceedings page). arXiv 1606.03476 (v1). Opened: https://arxiv.org/abs/1606.03476, https://arxiv.org/pdf/1606.03476, https://papers.nips.cc/paper/2016/hash/cc7e2b878868cbae992d1fb743995d8f-Abstract.html.

- Occupancy measure ρ_π(s,a) = π(a|s) Σ_t γ^t P(s_t = s | π). IRL followed by RL equals matching occupancy measures under a regularizer ψ (Prop. 3.2: RL∘IRL_ψ(π_E) = argmin_π −H(π) + ψ*(ρ_π − ρ_πE)), where π_E is the expert, H causal entropy and ψ* the convex conjugate of ψ. [GAIL §3 Prop. 3.2]
- With the GAIL regularizer, the problem becomes min_π D_JS(ρ_π, ρ_πE) − λH(π), solved at the saddle point of E_π[log D(s,a)] + E_πE[log(1 − D(s,a))] − λH(π). D: S×A → (0,1) is a discriminator and D_JS the Jensen-Shannon divergence. [GAIL §5 Eq. 14-16]
- Algorithm: an Adam step on the discriminator weights w to increase Eq. 16, then a TRPO step on θ with cost log D_w(s,a). Convention: D → 1 on policy samples, so the policy lowers log D. [GAIL §5 Algorithm 1 Eq. 17-18]
- Setup: policy, value and discriminator are 2×100 tanh MLPs; GAE with γ = 0.995, λ = 0.97; entropy λ = 0 on all tasks except Reacher. BC baseline: Adam, minibatch 128, 70/30 split, stop when validation error stops falling. [GAIL §6; App. B]
- Result: BC does well with lots of data or on simple tasks (it beat GAIL on Reacher). GAIL reaches at least 70% of expert performance on the MuJoCo tasks at every tested dataset size, and beats BC when data is scarce. [GAIL §6, Fig. 1]

## DreamerV3 (2023 arXiv, 2025 Nature)

Mastering Diverse Domains through World Models (arXiv title). Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, Timothy Lillicrap. arXiv 2301.04104 (v2, 17 Apr 2024). Published as "Mastering diverse control tasks through world models", Nature 640(8059):647-653, April 2025, doi:10.1038/s41586-025-08744-2 (seen on the RePEc index and the project page; the Nature article itself was not opened, so whether its equations or hyperparameters differ from arXiv v2 is unverified). Facts are from arXiv v2. Opened: https://arxiv.org/abs/2301.04104, https://arxiv.org/pdf/2301.04104, https://danijar.com/project/dreamerv3/, https://ideas.repec.org/a/nat/nature/v640y2025i8059d10.1038_s41586-025-08744-2.html.

- World model loss: L(φ) = E[Σ_t (β_pred L_pred + β_dyn L_dyn + β_rep L_rep)] with β_pred = 1, β_dyn = 1, β_rep = 0.1. L_dyn = max(1, KL[sg(q_φ(z_t|h_t,x_t)) ‖ p_φ(z_t|h_t)]) and L_rep = max(1, KL[q_φ(z_t|h_t,x_t) ‖ sg(p_φ(z_t|h_t))]). Here sg is stop-gradient, q the encoder posterior, p the dynamics prior, h_t the recurrent state, z_t the stochastic latent, and the max(1, ·) is "free bits" (1 nat). Latent categoricals mix 1% uniform. [DV3 §"World model learning" Eq. 2-3]
- Actor loss (REINFORCE for discrete and continuous actions): L(θ) = −Σ_t sg((R_t^λ − v_ψ(s_t)) / max(1, S)) log π_θ(a_t|s_t) + η H[π_θ(a_t|s_t)], η = 3×10⁻⁴. S = EMA(Per(R^λ, 95) − Per(R^λ, 5), 0.99) is the 5th-to-95th percentile return range. Returns are only scaled down, never up (the max with 1). [DV3 §"Actor learning" Eq. 6-7]
- Critic: λ-return R_t^λ = r_t + γc_t((1−λ)v_t + λR_{t+1}^λ), γ = 0.997, λ = 0.95, with c_t the continue flag. The critic outputs a categorical over exponentially spaced bins trained on twohot targets. Extras: replay loss scale 0.3, regularized toward an EMA of its own weights, zero-initialized output weights. [DV3 §"Critic learning" Eq. 5; Tab. 4]
- symlog(x) = sign(x) ln(|x| + 1), inverse symexp(x) = sign(x)(exp(|x|) − 1). It is applied to vector observations (encoder input and decoder target), and reward and value use symexp twohot bins B = symexp([−20, ..., +20]). [DV3 §"Robust predictions" Eq. 8-11]
- One fixed hyperparameter set for every domain (Table 4: replay 5×10⁶, batch 16 × length 64, lr 4×10⁻⁵, LaProp, AGC 0.3, imagination horizon 15). Ablations: the KL objective (free bits plus balance) matters most, then return normalization, then symexp twohot. First to collect diamonds in Minecraft from scratch. [DV3 Tab. 4; §"Ablations" Fig. 6; §Minecraft]

## TD-MPC2 (2024)

TD-MPC2: Scalable, Robust World Models for Continuous Control. Nicklas Hansen, Hao Su, Xiaolong Wang. ICLR 2024 (PDF header "Published as a conference paper at ICLR 2024"). arXiv 2310.16828 (v2). Opened: https://arxiv.org/abs/2310.16828, https://arxiv.org/pdf/2310.16828.

- No decoder. Components: encoder z = h(s,e), latent dynamics z′ = d(z,a,e), reward R(z,a,e), terminal value Q(z,a,e), policy prior p(z,e). e is a learned task embedding. [TD-MPC2 §3.1 Eq. 2]
- Model loss over a horizon H: L(θ) = E Σ_{t=0}^H λ^t (‖z′_t − sg(h(s′_t))‖²₂ + CE(r̂_t, r_t) + CE(q̂_t, q_t)), with TD target q_t = r_t + γ Q̄(z′_t, p(z′_t)). Q̄ is an EMA of Q; CE is cross-entropy over discrete bins in a log-transformed space; λ ∈ (0,1] down-weights later steps. Coefficients 20 / 0.1 / 0.1, λ = 0.5. Ensemble of 5 Q-functions; targets use the min of 2 randomly chosen. [TD-MPC2 §3.1 Eq. 3; Tab. 8]
- SimNorm: split z into L groups of size V and softmax each group with temperature τ (V = 8, τ = 1). Essential for stability. [TD-MPC2 §3.1 Eq. 5; App. H]
- Planning: MPPI over action sequences of length H = 3, maximizing E[γ^H Q(z_{t+H}, a_{t+H}) + Σ_{h=t}^{H−1} γ^h R(z_h, a_h)]. 6 iterations, 512 samples, 24 from the policy prior, 64 elites. [TD-MPC2 §3.2 Eq. 6; Tab. 8]
- One hyperparameter set for 104 tasks (DMControl, Meta-World, ManiSkill2, MyoSuite). Discount heuristic γ = clip((T/5 − 1)/(T/5), [0.95, 0.995]), with T the episode length after action repeat (gives 0.99 for T = 500). Adam lr 3×10⁻⁴, batch 256, UTD 1, buffer 10⁶. A single 317M-parameter agent was trained on 80 tasks. [TD-MPC2 Tab. 8; App. Eq. 7; Fig. 1]

## RLPD (2023)

Efficient Online Reinforcement Learning with Offline Data. Philip J. Ball, Laura Smith, Ilya Kostrikov, Sergey Levine. ICML 2023 (PDF footer "Proceedings of the 40th International Conference on Machine Learning"; arXiv comments "Short Presentation at ICML 2023"). arXiv 2302.02948 (v4). Opened: https://arxiv.org/abs/2302.02948, https://arxiv.org/pdf/2302.02948.

- Symmetric sampling: every batch is 50% online replay and 50% offline data (128 + 128). No pretraining and no offline-RL constraint; it is SAC-style online RL. [RLPD §4.1; App. B.2 Tab. 1]
- LayerNorm in the critic bounds Q-values by the norm of the last-layer weights (‖Q‖ ≤ ‖w‖), which limits extrapolation to unseen actions. Without it, critics diverged. [RLPD §4.2 Fig. 2-3, Fig. 7]
- Sample-efficient updates: ensemble of E = 10 critics, target = min over a random subset of Z ∈ {1, 2}, UTD (gradient steps per env step) G = 20. Clipped double-Q and entropy backups are switched per domain (Table 2). [RLPD §4.3-4.4; Algorithm 1; Tab. 2]
- Table 1: γ 0.99, Adam lr 3×10⁻⁴, critic EMA ρ = 0.005, width 256, initial α 1.0, target entropy −dim(A)/2. [RLPD App. B.2 Tab. 1]

## PQL (2023)

Parallel Q-Learning: Scaling Off-policy Reinforcement Learning under Massively Parallel Simulation. Zechu Li, Tao Chen, Zhang-Wei Hong, Anurag Ajay, Pulkit Agrawal. ICML 2023 (PDF footer "Proceedings of the 40th International Conference on Machine Learning"; arXiv comments "Accepted by ICML 2023"). arXiv 2307.12983 (v1). Opened: https://arxiv.org/abs/2307.12983, https://arxiv.org/pdf/2307.12983.

- Three concurrent GPU processes: Actor (Isaac Gym rollouts), V-learner (critic updates), P-learner (actor updates). It builds on DDPG with double Q and n-step returns. [PQL §3.1 Fig. 1]
- Speed ratios: β_{a:v} = f_a/f_v and β_{p:v} = f_p/f_v. f_a is env steps per unit time, f_v critic updates per unit time, f_p policy updates per unit time. Suggested start: β_{p:v} = 1:2, β_{a:v} = 1:8. With more envs, the critic needs more updates per env step (Shadow Hand: 1:4 best at N ≤ 4096, 1:12 at N ≥ 8192). Syncing the policy acts as a hard target-policy update, so there is no separate target actor. [PQL §3.2; §4.4 Fig. 6-7]
- Mixed exploration: env i of N uses Gaussian noise std σ_i = σ_min + (i−1)/(N−1)·(σ_max − σ_min), σ_min = 0.05, σ_max = 0.8. It matches or beats every fixed σ tested, which removes the need to tune σ. [PQL §3.3; §4.3 Fig. 4]
- Table B.1: 4,096 envs; actor and critic lr 5×10⁻⁴; τ = 5×10⁻²; batch 8,192; γ 0.99; observation normalization on; grad clip 0.5; 3-step targets; replay 5×10⁶. PPO baseline uses the Isaac Gym defaults. [PQL App. B.1 Tab. B.1]
- Findings: PQL (and PQL-D, with distributional critic) learned fastest in wall-clock time on 5 of 6 Isaac Gym tasks against PPO, DDPG(n) and SAC(n). DDPG(n) beat SAC(n) on every task. Large batches speed learning up to a GPU-bound limit. [PQL §4.2 Fig. 3; §4.4.3 Fig. 10]

## PQN (2025)

Simplifying Deep Temporal Difference Learning. Matteo Gallici, Mattie Fellows, Benjamin Ellis, Bartomeu Pou, Ivan Masmitja, Jakob Nicolaus Foerster, Mario Martin. ICLR 2025 (PDF header "Published as a conference paper at ICLR 2025"; the abstract page has no venue). arXiv 2407.04811 (v6). Opened: https://arxiv.org/abs/2407.04811, https://arxiv.org/pdf/2407.04811.

- Theory: LayerNorm plus ℓ2 regularization on the final layer makes TD stable without target networks or replay, even off-policy, for wide enough networks. Only the final-layer weights affect stability. Use ℓ2 sparingly, because it pulls the fixed point toward 0. LayerNorm alone stabilized every environment except Baird's counterexample. [PQN §3.2 Eq. 9, Thm. 2]
- Target: Q(λ) returns from short on-policy rollouts of length T across I parallel envs. R_{t+T}^λ = max_{a′} Q_φ(s_{t+T}, a′), then recursively R_t^λ = r_t + γ[λR_{t+1}^λ + (1−λ) max_{a′} Q_φ(s_{t+1}, a′)], or r_t if s_t is terminal. Loss Σ (R_t^λ − Q_φ(x_t))² over minibatches and epochs, like PPO. No replay buffer beyond the I·T rollout, no target network. [PQN §4; Algorithm 1]
- Atari hyperparameters: 128 envs, 32 steps, ε 1.0 → 0.001 over 10% of training, 2 epochs, 32 minibatches, LayerNorm, lr 2.5×10⁻⁴, max grad norm 10, γ 0.99, λ 0.65. [PQN App. Tab. 5]
- λ = 0.65 beat λ = 0 (one-step TD) by a wide margin. Adding a 1M replay buffer reached the same score about 6× slower. Rainbow's Atari median with about 50× less training time. [PQN §5.5 Fig. 6; §5.2]

## CrossQ (2024)

CrossQ: Batch Normalization in Deep Reinforcement Learning for Greater Sample Efficiency and Simplicity. Aditya Bhatt, Daniel Palenicek, Boris Belousov, Max Argus, Artemij Amiranashvili, Thomas Brox, Jan Peters. ICLR 2024 (PDF header "Published as a conference paper at ICLR 2024"; arXiv comments "Published at ICLR 2024"). arXiv 1902.05605 (v4; v1 dates from 2019). Opened: https://arxiv.org/abs/1902.05605, https://arxiv.org/pdf/1902.05605.

- Change to SAC: remove target networks, add BatchNorm (Batch Renormalization, BRN) to the critic and actor, and evaluate Q(s,a) and Q(s′,a′) in one joint forward pass with a′ ~ π(s′). So BatchNorm statistics come from a 50/50 mix of both batches. Critic loss: mean of (Q(s,a) − (r + γ sg(min_i Q_i(s′,a′) − α log π(a′|s′))))². [CrossQ §3.1-3.2; Fig. 2 code]
- Why joint: with a target network, the target's BatchNorm statistics never see (s′, π(s′)) samples, which are out of distribution. That is why naively adding BRN to SAC with target networks hurts. [CrossQ §3.2; §4.4]
- Table 1: UTD 1 (against 20 for REDQ and DroQ); critic width 2048; Adam β₁ = 0.5; policy delay 3; 2 critics; BRN momentum 0.99, warm-up 10⁵ steps; lr 10⁻³; batch 256; γ 0.99. [CrossQ App. A.3 Tab. 1]
- Ablations: removing BRN is worst; swapping BatchNorm for LayerNorm drops roughly to SAC level; critic width 256 is worse but still well above SAC; SAC gains nothing from width 2048 alone. Wall-clock time is up to 4× faster than REDQ and DroQ at matching sample efficiency. [CrossQ §4.4 Fig. 9; §1]

## Approximating KL Divergence (2020)

Approximating KL Divergence. John Schulman, blog post dated 2020/03/07. No venue, no arXiv id. Opened: http://joschu.net/blog/kl-approx.html.

- Goal: estimate KL[q,p] = E_{x~q}[log(q(x)/p(x))] from samples x ~ q, when you can evaluate p(x) and q(x) but not the sum. Ratio r = p(x)/q(x). [KL blog]
- k1 = −log r: unbiased, high variance (it is negative for about half the samples, while KL ≥ 0). [KL blog]
- k2 = (log r)²/2: biased, lower variance, always ≥ 0. Its bias is small when p ≈ q, because any f-divergence with f″(1) = 1 matches KL to second order. [KL blog]
- k3 = (r − 1) − log r: unbiased (since E_q[r − 1] = 0, it is k1 plus a zero-mean control variate) and always ≥ 0 (log x ≤ x − 1). The blog calls it apparently "strictly better" than k2 at larger KL. For the reverse direction KL[p,q], use r log r − (r − 1). [KL blog]
- Blog numbers (bias/true, stdev/true). True KL 0.005: k1 0 and 20; k2 0.002 and 1.42; k3 0 and 1.42. True KL 0.5: k1 0 and 2; k2 0.25 and 1.73; k3 0 and 1.7. [KL blog tables]
- PPO use: with samples from π_old, q = π_old and p = π_θ, so r is PPO's ratio r_t(θ). ppo2's approxkl = mean(−log r) is k1, and the 37-details blog recommends mean((r − 1) − log r), which is k3. [KL blog; 37D §13 core details, debug variables]

---

## Disagreements and errors found

- PPO Eq. 10-11 print the last exponent as γ^{T−t+1} and (γλ)^{T−t+1}. Counting terms from t to T−1 gives T−t−1. [PPO §5]
- Value loss clipping: in the ppo2 code (Engstrom, 37D) but not in the PPO paper. Andrychowicz finds it hurts at every threshold [WM §3.4]. Engstrom's ablation does not list it among the optimizations that mattered [Impl Fig. 1].
- Clip ε: the PPO paper's best was 0.2 [PPO Tab. 1]; Andrychowicz recommends starting at 0.25 [WM §3.1].
- GAE λ: PPO uses 0.95 [PPO Tab. 3]; Andrychowicz recommends 0.9 [WM §3.4]; GAE found 0.92-0.99 [GAE §6.3].
- Initial action std: ppo2 initializes log std to 0, so std 1 [37D]; Andrychowicz found an initial std of about 0.5 best [WM §3.2]; PPO's Roboschool runs annealed log std from −0.7 to −1.6 [PPO Tab. 4].
- Advantage normalization: ppo2 does it per minibatch [37D]; Andrychowicz finds little effect [WM §3.3].
- Reward clipping range: Engstrom says "usually [−5, 5] or [−10, 10]" [Impl §3]; 37D says [−10, 10] [37D].
- Number of parallel envs: Andrychowicz reports sample efficiency falls as num_envs grows [WM §3.5]; 37D argues wall-clock time is the right measure [37D].
- LayerNorm vs BatchNorm in critics: CrossQ finds LayerNorm in place of BatchNorm falls to about SAC level [CrossQ §4.4]; PQN prefers LayerNorm and uses BatchNorm only for input normalization [PQN §3.3, §5.5]; RLPD relies on LayerNorm [RLPD §4.2]. The settings differ (CrossQ removes target networks and keeps replay; PQN removes both; RLPD keeps target networks).
- Target entropy: SAC-Apps uses −dim(A) [SAC-Apps Tab. 1]; RLPD uses −dim(A)/2 [RLPD Tab. 1].
- Learning rate for SAC: the SAC paper uses 3·10⁻⁴ [SAC Tab. 1]; CrossQ's Table 1 lists 10⁻³ for all algorithms including its SAC baseline [CrossQ Tab. 1].
- Soft target rate: DDPG τ = 0.001 [DDPG §7]; TD3 and SAC τ = 0.005 [TD3 §6.1, SAC Tab. 1]; PQL τ = 0.05 [PQL Tab. B.1].
- Time-limit truncation: TD3 bootstraps through time-outs [TD3 App. D]; ppo2 does not [37D]; Andrychowicz found it made no difference with 1000-step limits [WM §3.6].
- DreamerV3: the text gives prediction horizon T = 16 [DV3 §"Critic learning"], while Table 4 lists imagination horizon H = 15.
- Typos in the PDFs: SAC-Apps Eq. 7 writes Q_θ(s_t, s_t) for Q_θ(s_t, a_t); RLPD Algorithm 1 line 17 writes y = y + γα log π, where an entropy bonus needs −γα log π; DDPG §7 prints the pixel init range as [3×10⁻⁴, 3×10⁻⁴] (missing a minus sign).
- KL blog: the text sets q = N(0,1), p = N(0.1,1), but the code defines p = Normal(0,1), q = Normal(0.1,1). For equal-variance Gaussians the two KL directions are equal, so the table is unaffected.
- 37D says PPO uses a "Gaussian distribution with full covariance ... each dimension performed independently". The code and the "independent action components" detail describe a diagonal Gaussian.

---

## Which paper for which question

| Question | Paper |
|---|---|
| What exactly is the PPO clipped loss, and what are the paper's MuJoCo defaults? | PPO §3 Eq. 7, Tab. 3 |
| How do I compute advantages, and what γ and λ should I use? | GAE Eq. 16-18, §6.3; PPO Eq. 11; What Matters §3.4 |
| Which PPO tricks matter for continuous control (obs norm, init, std, value clipping)? | What Matters §3.2-3.4; Implementation Matters §3; 37 Details §continuous |
| My PPO does not match the reference: what is in the ppo2 code but not in the paper? | 37 Details; Implementation Matters §3 |
| Is my PPO update too large? Which KL estimate should I log? | KL blog (k3); 37 Details §Recommendations (approx_kl < 0.02); PPO §4 |
| Does more envs or bigger batches help PPO? | What Matters §3.5; 37 Details (wall-clock argument) |
| How many seeds, and how do I report results honestly? | Henderson §Random Seeds; Agarwal §4 (IQM, stratified bootstrap) |
| What are the DQN target and Atari hyperparameters? | DQN Algorithm 1, Ext. Data Tab. 1 |
| Why do Q-values overestimate, and how do I fix it? | Double DQN Thm. 1; TD3 §4.2 |
| Should I use prioritized replay, and with what α and β? | PER §3.3-3.4, §4 |
| What are DDPG's or TD3's update rules and defaults? | DDPG §3, §7; TD3 Eq. 15, §6.1 |
| What are SAC's losses and how is the temperature learned? | SAC Eq. 1-12; SAC-Apps Eq. 5-9, 18, Tab. 1 |
| How do I handle time-limit truncation? | TD3 App. D; 37 Details (GAE item); What Matters §3.6 |
| Can off-policy RL beat PPO in Isaac Gym wall-clock time? | PQL §3-4 |
| Can I drop target networks and replay buffers? | PQN §3-4; CrossQ §3 |
| Can I get REDQ-level sample efficiency at UTD 1? | CrossQ §3, Tab. 1 |
| How do I fine-tune online with demonstrations or offline data? | RLPD §4 |
| Why does behavior cloning drift, and how does DAgger fix it? | DAgger Thm. 2.1, Algorithm 3.1, Thm. 3.2 |
| How do I imitate without expert action labels at every state? | GAIL §5 Algorithm 1 |
| Which world model works across domains with fixed hyperparameters? | DreamerV3 Eq. 2-11, Tab. 4 |
| What is model-predictive control with a learned latent model and value? | TD-MPC2 §3, Tab. 8 |
| How should I normalize returns or rewards of unknown scale? | DreamerV3 Eq. 6-9 (percentile return scale, symlog); Implementation Matters §3 (reward scaling) |
