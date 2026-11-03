# Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning (legged_gym)

Nikita Rudin, David Hoeller, Philipp Reist, Marco Hutter (ETH Zurich and NVIDIA). Conference on Robot Learning (CoRL) 2021, PMLR vol. 164, pp. 91-100 (published 2022; venue taken from FastTD3's reference list, the arXiv PDF does not print it). arXiv:2109.11978 (v3 used here). Code: https://leggedrobotics.github.io/legged_gym/

## The idea

Run thousands of robots in one Isaac Gym simulation with the simulation, observation and reward computation, and PPO all on the GPU, so no data crosses PCIe [legged-gym §2]. Massive parallelism forces changes to PPO: very few steps per robot per update (small n_steps), very large minibatches, and correct bootstrapping on time-out resets [§2.2]. With a game-inspired terrain curriculum, ANYmal C learns perceptive locomotion (proprioception plus a terrain height map) over rough terrain, slopes, stairs and obstacles in under 20 minutes (flat terrain in under 4 minutes) on one workstation GPU and transfers to the real robot [Abstract, §4.2-4.3].

## Notation

| Symbol | Meaning |
|---|---|
| n_robots | Number of robots (environments) simulated in parallel |
| n_steps | Steps each robot takes per policy update (the rollout length) |
| B | Batch size per policy update, B = n_robots × n_steps [§2.2.1] |
| GAE | Generalized Advantage Estimation; λ ("GAE discount factor") = 0.95 [Table 3] |
| kl* | Target KL divergence between new and old policy for the adaptive learning rate (0.01) [Alg. 1] |
| α | PPO learning rate (adapted from KL) |
| q_j, q̇_j, q̈_j | Joint positions, velocities, accelerations [Table 1] |
| q*_j | Target joint positions (the policy's actions) |
| τ_j | Joint torques |
| v_b, ω_b | Base linear and angular velocity (3-vectors); subscripts xy and z pick components, z aligned with gravity |
| v*_b, ω*_b | Commanded base linear and angular velocity |
| n_c | Number of collisions (knee, shank, or foot-on-vertical-surface contacts) |
| t_air | Per-foot air time (4-vector) |
| dt | Environment (policy) time step; policy runs at 50 Hz, so dt = 0.02 s [App. A.1.1] |
| φ(x) | Tracking kernel exp(−‖x‖²/0.25) [Table 2] |

## The method

### 1. End-to-end GPU pipeline

Policy inference, simulation (PhysX in Isaac Gym), reward and observation computation, storage and PPO updates all live on the GPU; CPU simulation would require PCIe transfers that can be up to 50× slower than GPU compute [§2]. Thousands of robots share one simulation, so all terrain types and levels are tiled into a single mesh, and a robot's difficulty is changed by physically moving it to a different tile [§2.1]. Throughput tricks [App. A.1]: simulation step 0.005 s, i.e. 4 sim steps per 50 Hz policy step (smaller steps made the actuator network unstable); keep only necessary collision bodies (feet, shanks, knees, base); convert a low-resolution height field to a triangle mesh with corrected vertical faces; spread robots apart, since PhysX still detects (but ignores) robot-robot contacts. Simulation time per step halves between the first minutes of training and the end as robots disperse and stop crashing [App. A.1.2].

### 2. PPO adapted to the massively parallel regime

Custom PPO with every operation and all data on the GPU [§2.2].

- Small n_steps. With n_robots in the thousands, n_steps must be small to keep B reasonable, but not arbitrarily small: GAE needs rewards from several consecutive steps, and the algorithm struggled below 25 consecutive steps (0.5 s of sim time) [§2.2.1]. Episodes (20 s time-out) are not reset at each iteration, so one episode spans many policy updates.
- Large minibatches. Minibatches of tens of thousands of samples stabilized learning without increasing training time [§2.2.1].
- Time-out bootstrapping. Resets from falls or goals are predictable by the critic; resets from the time limit are not (episode time is not observed). On a time-out, the reward is augmented with the critic's own estimate of the discounted future return, i.e. the value target bootstraps instead of treating the time-out as terminal. The standard Gym interface does not separate the two cases and Stable-Baselines ignored it, so they modified the interface to flag time-outs [§2.2.2]. This lowers critic loss and raises total reward by about 10-20% on flat and rough terrain [App. A.2, Fig. 10].
- Adaptive learning rate from KL [Alg. 1, following Heess et al.]: compute kl = KL(π_new, π_old); if kl > 2·kl*, α ← max(1e-5, α/1.5); else if kl < 0.5·kl*, α ← min(1e-2, 1.5·α).

PPO hyperparameters of the tested policy [App. A.4, Table 3]:

| Hyperparameter | Value |
|---|---|
| Batch size | 98,304 (4096 robots × 24 steps) |
| Minibatch size | 24,576 (4096 × 6), i.e. 4 minibatches |
| Epochs | 5 |
| Clip range | 0.2 |
| Entropy coefficient | 0.01 |
| Discount γ | 0.99 |
| GAE λ | 0.95 |
| Desired KL kl* | 0.01 |
| Learning rate | Adaptive (Alg. 1) |
| Policy updates | 1500 [§3.1, §4.2] |

The paper does not give the policy or critic network architecture, the initial learning rate, PD gains, or action scaling; those are in the open-source code.

### 3. Task, observations, actions, rewards [§3, §3.2, App. A.3]

Task: follow randomized base-heading and linear-velocity commands (fixed for an episode) across rough, sloped, stair, and obstacle terrain. Observations: base linear and angular velocity, projected gravity vector, joint positions and velocities, previous actions, and 108 terrain height samples on a grid around the base (each is terrain-to-base height). Actions: target joint positions, tracked by a PD controller (for ANYmal, a learned actuator network, see step 5). No gait-specific reward or action terms. A single policy and reward set are used for all terrains. Base contact counts as a crash and resets the robot.

Reward = weighted sum of nine terms; weights are multiplied by dt [Table 2]:

| Term | Definition | Weight |
|---|---|---|
| Linear velocity tracking | φ(v*_{b,xy} − v_{b,xy}) | 1·dt |
| Angular velocity tracking | φ(ω*_{b,z} − ω_{b,z}) | 0.5·dt |
| Linear velocity penalty | −v²_{b,z} | 4·dt |
| Angular velocity penalty | −‖ω_{b,xy}‖² | 0.05·dt |
| Joint motion | −‖q̈_j‖² − ‖q̇_j‖² | 0.001·dt |
| Joint torques | −‖τ_j‖² | 0.00002·dt |
| Action rate | −‖q̇*_j‖² | 0.25·dt |
| Collisions | −n_collision | 0.001·dt |
| Feet air time | Σ_f (t_air,f − 0.5) | 2·dt |

φ(x) = exp(−‖x‖²/0.25). The feet air time term encourages longer steps, chosen for visual quality [§3.2]. The table prints the sum as f = 0 to 4, which would be five feet for a quadruped; check the code.

### 4. Game-inspired curriculum [§3.1]

Five procedurally generated terrain types (flat, sloped, randomly rough, discrete obstacles, stairs), each an 8 m square tile; slopes and stairs are arranged as pyramids so they can be crossed in any direction. Each robot has a terrain type and a difficulty level. Stairs and obstacles go from 5 cm to 20 cm step height; slopes from 0 to 25 degrees. Rules:

- Walk past the border of your tile, and your level goes up at the next reset.
- At the end of an episode, if you moved less than half the distance your commanded velocity asked for, your level goes down.
- Robots that solve the top level are sent back to a random level, for diversity and to avoid forgetting.

No tuning is needed, the distribution of robots over levels is itself the estimate of policy performance (no generator network or particle filter), and it costs almost nothing to run in parallel. With 4000 robots, after 500 updates robots handle slopes and stair descent but not yet stair climbing or obstacles; after 1000 updates they reach the top level on all types; training runs 1500 updates [§3.1, Fig. 3].

### 5. Sim-to-real additions [§3.3, §4.3, App. A.5]

- Friction per robot sampled uniformly in [0.5, 1.25].
- Random pushes every 10 s that change base velocity by up to ±1 m/s in x and y.
- Uniform observation noise, scaled from real robot data [Table 4]: joint positions ±0.01 rad, joint velocities ±1.5 rad/s, base linear velocity ±0.01 m/s (as printed), base angular velocity ±0.2 rad/s, projected gravity ±0.05 (printed with units rad/s²), commands 0, terrain heights ±0.1 m.
- Actuator network: ANYmal's series elastic actuators are modeled by an LSTM that maps current measurements to torque, a simplification of Hwangbo et al.'s feed-forward network over a history of measurements. Giving the policy memory of actuator history did not improve final performance.
- On hardware: policy fixed, no action filtering or constraint checks, heights queried from a Lidar-built elevation map. Map errors hurt robustness mainly at high speed, so max linear velocity command was cut to 0.6 m/s on the robot.

## Key results

Wall clock (tested policy: 4096 robots, B = 98,304, 1500 updates): under 20 minutes on an i9-11900k CPU and NVIDIA RTX A6000 GPU; flat terrain in under 4 minutes [Abstract, §4.2, footnote 2]. Training times cited for prior quadruped work: 12 h (blind, Lee et al. 2020), 82 h and 88 h (learning plus optimal control), 120 h (Miki et al.) [§1].

Parallelism study [§4.1, Fig. 4], curriculum removed (and the hardest stair/obstacle heights reduced) so total reward is comparable:

- Upper-bound baseline: 20,000 robots × 50 steps = 1M samples per update gave the best policy but long training.
- Batch sizes 49,152, 98,304, 196,608 were tested with 128 to 16,384 robots (16,384 is near the maximum for rough terrain on one workstation GPU), 5 runs each, reward after 1500 updates.
- Too many robots (so too few steps each) makes performance drop sharply; larger batches shift this drop to more robots. Too few robots gives a slow decline, attributed to highly correlated consecutive samples.
- Training time scales nearly linearly with robots up to about 4000, then gains slow.
- Best trade-off for this task: 2048 to 4096 robots with B ≈ 100k or 200k.

Simulation tests of the deployed policy [§4.2, Fig. 5] (forward command 0.75 m/s, side command in [−0.1, 0.1] m/s): nearly 100% success on stairs up to 0.2 m (the hardest trained level, near the robot's kinematic limit); success on random obstacles falls steadily with height (neighboring obstacles can differ by twice the nominal height); past 25 degrees robots cannot climb slopes but slide down with moderate success.

Generality [§4.2, Fig. 6]: same rewards and hyperparameters retrain for ANYmal C with an arm (about 20% more weight) and ANYmal B. Unitree A1 needs the actuator model removed, lower PD gains and torque penalties, and new default joint angles. Cassie (biped) needs one extra reward for standing on a single foot. Gaits always converge to trotting, often with artifacts (dragging leg, odd base height) until reward weights are tuned [§4.2].

Compute [App. A.1, Figs. 8-9]: simulation is the largest cost per step, then observation/reward computation; policy and actuator-net inference are near constant. VRAM for 4096 robots on rough terrain: 9 GB with rendering, 6 GB without (flat: 7 GB and 5 GB).

## Gotchas when reproducing

- Bootstrap the value target on time-outs but not on falls. With short rollouts, nearly every batch contains time-outs, so ignoring this is not a rare-edge-case bug [§2.2.2, App. A.2].
- n_steps has a floor (about 25 steps, 0.5 s, for this task) because of GAE; the tested config actually uses 24 steps (B = 4096 × 24), so treat 24-25 as the floor rather than a hard rule [§2.2.1, Table 3].
- Do not scale robots past the point where per-robot horizon gets too short; pick n_robots and B jointly (Fig. 4c) [§4.1].
- Minibatches are huge (24,576), only 4 per epoch, 5 epochs [Table 3].
- The adaptive KL-based learning rate (kl* = 0.01, factor 1.5, bounds 1e-5 to 1e-2) is part of the recipe [Alg. 1].
- Reward weights are scaled by dt; gaits have artifacts until weights are tuned [Table 2, §4.2].
- The actuator network (ANYmal only) is what makes sim-to-real work for series elastic actuators; drop it and retune PD gains for other robots [§3.3, §4.2].
- Policy frequency 50 Hz with 4 sim substeps of 0.005 s; going lower destabilized the actuator net [App. A.1.1].
- Episode length 20 s, pushes every 10 s, friction [0.5, 1.25] [§2.2.1, §3.3].

## Connections

PPO in practice: a concrete, fully specified PPO + GAE setup (clip 0.2, γ 0.99, λ 0.95, adaptive LR) with the engineering that makes on-policy RL work at thousands of environments, plus reward shaping, curricula and domain randomization for sim-to-real. Its time-out bootstrapping point is the actor-critic unit's value-target definition in practice.
