# SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model

Qiayuan Liao, Yiyang Shao, Junfeng Long, Kevin Zakka, Roberto Horowitz, Pieter Abbeel, Koushil Sreenath (UC Berkeley). CoRL 2026 (10th Conference on Robot Learning, Austin TX; printed on the arXiv PDF). arXiv:2610.28480 (v1, 20 Oct 2026). Project page: https://supermimic.github.io

## The idea

Train a Unitree G1 motion tracker by fitting one deterministic, differentiable world model to logged simulator transitions and then backpropagating the policy's return through 32 imagined steps of that model, with a critic for the return beyond 32 steps [SuperMimic §3, Fig. 2]. The world model, the policy and the critic train together in one Dyna-style loop (Dyna: interleave real data collection, model fitting and policy updates from model rollouts) [SuperMimic §3.1]. In 3 h on one H100 it beats the return of SONIC, a PPO tracker trained for 7 days on 128 L40 GPUs, with 25 failed motions versus 179 on a 129,514-motion benchmark [SuperMimic Abstract, §4.1, Tab. A.8]. A controlled comparison shows simulator PPO barely improving from 4 to 8 GPUs [SuperMimic §4.1, Fig. 4a].

## Notation

| Symbol | Meaning |
|---|---|
| s_t, ŝ_t | Simulator state and world-model predicted state (a hat marks the model's version of a sim quantity) [SuperMimic §3.1] |
| o_t = ō_t + η_t | Proprioceptive observation = noise-free part ō_t + sensor noise η_t ~ ρ_η (noise sampled outside the model) [§3.1, Tab. A.1] |
| g_t | Goal; for tracking, K = 10 future reference frames [§3.1, App. A] |
| a_t | Action from a deterministic policy a_t = π_θ(o_t, g_t); maps to joint setpoint q_sp = α a_t + q⁰ [§3.1, App. A] |
| q⁰, α | Default joint pose; action scale (α's value is not given) [App. A] |
| ε_t ~ ρ_ε | Action noise added only during sim data collection, N(0, 0.1²) per dimension [§3.1, Tab. A.5] |
| d_t, d̂_t | Sim termination flag (1 if terminal); model's termination probability in [0, 1] [§3.1] |
| c_t | Soft survival weight: c₀ = 1, c_{t+1} = c_t(1 − d_{t+1}) [§3.1] |
| ξ | Per-episode sim context = domain-randomization parameters (friction, joint calibration offsets, torso center-of-mass offset) [§3.2, Tab. A.3] |
| f_ω, r_ω, g_ω, d_ω | World-model dynamics, reward, observation and termination heads with weights ω; each head may be analytic instead of learned [§3.2] |
| k | Length of the autoregressive replay windows used to fit the model, k = 8 [§3.2] |
| H | Imagination horizon for policy updates, H = 32 [§3.3] |
| V_ψ | Critic with weights ψ, input (ŝ_t, g_t, ξ) [§3.3] |
| γ, λ | Discount 0.99; GAE (generalized advantage estimation) parameter 0.95 [Tab. A.5] |
| R̂_t | GAE return target computed backward over the imagined rollout [Eq. 3] |
| BPTT | Backpropagation through time: gradient of the summed return through every model step |
| enc_{θ,i}, dec_θ, z | Per-goal-interface encoder, shared decoder, 32-dim tanh-bounded latent [App. B, Tab. A.4] |
| p(I) | Distribution over goal interfaces (motion tracking, velocity command, ...) [App. B] |

## The method

### 1. Collect data in the simulator
The current policy runs in MuJoCo Warp (GPU MuJoCo) with action noise; transitions (s_t, a_t + ε_t, ξ) go to a fixed-size FIFO replay buffer [SuperMimic §3.1, Fig. 2, §4]. 4096 sim environments per GPU, buffer 1000 steps per environment [Tab. A.5]. Old transitions stay valid for model fitting because dynamics do not depend on the policy that collected them [§3.1]. Resets use return-based sampling (step 5).

### 2. Fit the world model
The model is deterministic: it conditions on ξ and predicts the noise-free observation, since the sim's randomness has known sources [SuperMimic §3.2]. Fit on k = 8 step windows, seeded at a logged state and rolled forward autoregressively on logged actions [Eq. 1]:

L_WM(ω) = E_B Σ_{j=1..k} [ ‖ŝ_j − s_j‖² + (r̂_j − r_j)² + ‖ō̂_j − ō_j‖² + BCE(d̂_j, d_j) ]

Analytic heads drop their term. The paper says this multi-step fit trains the model to correct its own errors, so it stays stable over rollouts longer than k [§3.3].

### 3. Improve the policy by BPTT through the frozen model
Start from a replay state s₀ with its ξ, re-add sensor noise at each step, act, and maximize [SuperMimic Eq. 2]:

Ĵ(θ) = E[ Σ_{t=0..H−1} γ^t ĉ_t r̂_{t+1} + γ^H ĉ_H V_ψ(ŝ_H, g_H, ξ) ]

The gradient flows through the model's Jacobians with respect to ŝ_t and a_t (a pathwise gradient), versus PPO's likelihood-ratio (score function) estimate; the paper argues lower variance over short horizons [§3.3]. The critic is fit to GAE targets over imagined steps only, R̂_t = r̂_{t+1} + γ(1 − d̂_{t+1})[(1 − λ)V_{t+1} + λR̂_{t+1}], with R̂_H = V_H, targets frozen while fitting [Eq. 3]. This is a short-horizon actor-critic like SHAC, with the soft survival weight in place of hard termination and no target network [§3.3].

### 4. Loop
Each iteration: collect, one world-model step, one policy step, critic fit; ω is frozen only inside the policy's backward pass [SuperMimic Alg. A.1, Tab. A.5].

### 5. Return-based reset sampling [SuperMimic App. D.1]
The motion timeline is cut into 1 s bins; each bin b keeps a running estimate G_b of discounted return from it. Reset start points: 50% uniform over bins, 40% from softmax_b(−25 G_b / G_max), 10% in proportion to each bin's past softmax weight (so learned hard motions keep some practice). G_max is the largest attainable discounted return.

### 6. Multiple goal interfaces [SuperMimic App. B]
a = dec_θ(enc_{θ,i}(g), o). All interfaces share f_ω. A downstream interface trains either by distillation (motion-tracking reward, command read from the reference) or by RL on its own task reward; its gradient passes through the decoder to its encoder with a stop-gradient on decoder weights.

### Hyperparameters [SuperMimic Tab. A.5]

| Item | Value |
|---|---|
| Model window k / batch | 8 / 10 batches of 2048 windows per update |
| Learning rate (model, policy, critic) | 1 GPU: 2e-4; 2 GPUs: √2 × 2e-4; Adam (policy β = (0.9, 0.95)) |
| Grad clipping | global norm 1.0 (policy: then element-wise 0.1) |
| H / policy batch | 32 / 4096 imagined rollouts per update |
| γ / GAE λ | 0.99 / 0.95 |
| Critic | 5 epochs, 4 shuffled minibatches each |
| Sim envs per GPU / action noise / buffer | 4096 / N(0, 0.1²) / 1000 steps per env |
| Update-to-data ratio | 0.4 (not defined further) |
| LR schedule / length | cosine to 0.03× over 250,000 iterations, then constant / 320,000 iterations |
| Multi-GPU | data parallel, gradients averaged |

Networks [Tab. A.4]: dynamics MLP (2048, 2048, 2048); motion encoder MLP (2048, 1024, 512, 256); shared decoder MLP (2048, 1024, 512, 256); velocity encoder MLP (256, 256); critic MLP (256, 128), one per interface.

### G1 setup [SuperMimic App. A]
Observation (policy gets the last 10): joint positions q − q⁰ (29, noise U[−0.01, 0.01] rad), joint velocities (29, U[−0.5, 0.5] rad/s), IMU angular velocity (3, U[−0.2, 0.2] rad/s), projected gravity (3, U[−0.05, 0.05]), previous action (29) [Tab. A.1]. Low PD gains make the setpoint a torque proxy. Reward: BeyondMimic tracking reward without the self-collision penalty, joint-limit weight −10 → −2 [Tab. A.2]. Termination: root height error > 0.25 m or tilt error > 0.8 (about 78°); no end-effector termination. DR: friction U[0.3, 1.2], joint offsets U[−0.01, 0.01] rad, torso CoM offset x U[−0.025, 0.025] m, y and z U[−0.05, 0.05] m; root-velocity pushes removed because the deterministic model cannot capture the jumps [Tab. A.3]. Control 50 Hz = four 5 ms MuJoCo Warp substeps [Fig. 5 caption]; every reward term is multiplied by Δt = 0.02 s [Tab. A.6 caption].

Velocity-tracking reward for the RL velocity encoder [Tab. A.6]: linear velocity exp(−‖v_xy − v_cmd‖²/0.5²) weight 2.0; yaw rate exp(−(ω_z − ω_cmd)²/0.5) weight 2.0; action smoothness ‖a_t − a_{t−1}‖² −0.3; joint limit −2; torso sideways lean sin²φ −4; torso roll and pitch rate −0.05. Fall termination: root height < 0.35 m or 1 + u_z > 0.7 (about 73°). Commands: v_x ∈ [−1.5, 2] m/s, v_y ∈ [−1.32, 1.32] m/s, ω_z ∈ [−3.35, 3.35] rad/s, zero with probability 0.1, resampled every 250 steps (5 s) [App. G.1].

## Results that matter

- vs SONIC [SuperMimic §4.1, Fig. 3a]: SONIC checkpoint normalized return 0.90 (dashed line label). SuperMimic passes it within 3 h on one H100. Ratios about 56× wall-clock and about 7200× GPU-hours (21,000 vs 3). Caveats the authors state: 264.56 h of BONES-SEED data vs SONIC's 611 h (2.3×), and an H100 has 0.74× the fp32 and 5.5× the tensor-core peak of an L40.
- Failed motions on 129,514 training motions [Tab. A.8]: SONIC 179 (all 44 flips, all 22 side rolls), SuperMimic 25 after 3 h on 1 GPU, 0 after 9.8 h on 2 GPUs. SONIC v1.1 fails 239 [App. F.2].
- Controlled PPO baseline on LAFAN1 (40 clips, 2.45 h) [§4.1, Fig. 4a]: PPO-Sim shares observations, reward, DR and network sizes, uses BeyondMimic's PPO settings and 8192 envs per GPU. SuperMimic reaches 0.82 in 1.2 h on 1 GPU; PPO-Sim never does in 18 h: final 0.60 (1 GPU), 0.70 (4), 0.72 (8).
- Dataset sweep 1, 7, 40, 264 h [App. F.3]: 8-step prediction error stays 0.007-0.011 [Fig. 3c]; GPU-hours to target rise from about 0.5 to about 2 (read from Fig. 3b, approximate).
- Rollout speed [Fig. 5a]: world model is more than 10× faster than MuJoCo Warp over 2k-16k envs and 62× at 64k, where MuJoCo Warp has flattened (caption); the figure's annotations show 11× near 2k and 26× near 16k (positions read from Fig. 5a, approximate).
- Pathwise vs PPO on the same model [Fig. 5b]: SuperMimic reaches PPO-WM's best training return with about 64× fewer imagined steps. PPO-WM batch sizes tried: 4k×32, 8k×24, 8k×32, 16k×32 (envs × steps); 16k×32 was best.
- Critic [§4.3, Fig. 4b-c]: without it training fails at H ≤ 16; with it H = 16 matches critic-free H = 32; H = 32 is fastest to 0.82; H 64 → 128 nearly triples time. Under 0.5-4 m/s pushes in a T-pose, the critic policy settles in 0.5-2.4 s; the critic-free one never settles.
- Real G1 [§4.4, Fig. 6]: one tracker, no per-motion tuning: skipping jump, side roll, 90° flip, in-place 360° flip.
- Velocity encoder [§4.5, App. G.2]: RL-trained beats distillation on 650 of 672 step commands (distillation better on 15); median relative errors 0.16 vs 0.49; distillation falls on 7 commands, all with turn rate ≥ 2.8 rad/s.

## Use it for

- G1 PPO num_envs sweep: Fig. 4a is a published data point that simulator PPO on the same robot saturates as compute grows (0.70 → 0.72 from 4 to 8 GPUs); the 64..16384 sweep on one A100 (`../training/on-policy-at-scale.md`) probes the same saturation inside one GPU [inference]. PPO-WM's best batch was the largest tried (16k × 32 = 524,288 samples per update) [Fig. 5b], though that is PPO on a learned model, not the sim [inference].
- Same robot, same simulator family: mjlab also runs on MuJoCo Warp (outside knowledge, not stated in this paper), so Tab. A.1 noise, Tab. A.3 DR and Tab. A.6 velocity reward are reasonable starting values for a G1 walking task [inference].
- "How to use the extra data": SuperMimic's answer is to stop discarding it. A replay buffer of 4096 × 1000 transitions feeds model fitting, which on-policy PPO cannot reuse [§3.1, Tab. A.5; contrast is inference].
- Model-based RL: Dyna loop, multi-step model loss, model-error compounding handled by short H plus a critic. Policy gradient foundations: pathwise vs score-function gradient [§3.3]. Actor-critic (Nov 17): GAE critic targets, SHAC.
- Reset-based exploration: return-based reset sampling is a practical "reset to states the policy is bad at" rule, a heuristic cousin of resetting to previously seen states in local-simulator RL [App. D.1; link is inference].

## Gotchas / ambiguities

- Deterministic policy: the self-collision penalty was dropped because only a stochastic policy can exploit it [App. A]. A PPO pipeline (stochastic Gaussian policy) should keep it [inference].
- Pushes were removed for the world model's sake [App. A]; PPO sim-to-real recipes usually keep them (see `legged-gym.md`).
- Fig. 5a shades 2k-16k envs as "used in practice", but Tab. A.5 lists 4096 sim envs per GPU and 4096 imagined rollouts per update.
- Alg. A.1 samples one interface per update; App. G.1 says each imagined batch mixes both interfaces (the paper flags this).
- Tab. A.2 caption points to §3.3 for the joint-limit change; the change is stated in App. A.
- 650 + 15 = 665 of 672 commands; the other 7 are presumably the 7 distillation-fall commands, not stated outright [App. G.2].
- Symbol clashes: g_t is the goal and g_ω the observation head; α is the action scale and α_ω, α_θ learning rates.
- Fig. 4b-c uses training return under return-based resets, which differs from checkpoint scores in Fig. 4a (caption says so).
- No seed variation is plotted ("training very stable"); each velocity encoder has one seed [§4, App. G.2].
- Benchmark motions are training motions, so failure counts measure coverage, not generalization [App. F.2]. SONIC was scored with SuperMimic's termination rule, not SONIC's own criterion [App. F.1].
- Not given: PD gains, action scale α, the meaning of update-to-data ratio 0.4. Flat ground only [§6].
