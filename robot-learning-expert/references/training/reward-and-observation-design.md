# Rewards, observations, actions and terminations for robot tasks

Most "RL doesn't work" robot problems are MDP design problems: the reward pays for the wrong thing, the policy cannot observe what it needs, or a termination teaches it the wrong lesson. The worked example throughout is mjlab 1.6.0's shipped G1 velocity task (`mjlab/tasks/velocity/velocity_env_cfg.py`, `config/g1/env_cfgs.py`, `asset_zoo/robots/unitree_g1/g1_constants.py`, read from the installed package). Paper sources: `papers/legged-gym.md` (Table 2 rewards), `papers/canon-robotics.md`.

## The G1 velocity task as a template

Timing: physics dt 0.005 s, decimation 4, so the policy runs at 50 Hz; episodes last 20 s.

Actions: joint position offsets. q_target = q_default + scale·a per joint, scale = 0.25 · effort_limit / stiffness (`G1_ACTION_SCALE`). A PD controller with stiffness k_p = armature·ω² and damping k_d = 2·ζ·armature·ω, ω = 2π·10 Hz, ζ = 2.0, turns targets into torques. Choosing gains from the reflected inertia and a target natural frequency gives every joint the same closed-loop bandwidth.

Command: (v_x, v_y, ω_z) resampled every 3 to 8 s; 10% of envs get a zero (standing) command, 30% use heading control.

Observations, actor (with uniform noise during training): base linear velocity (±0.5), base angular velocity (±0.2), projected gravity (±0.05), joint positions relative to default (±0.01), joint velocities (±1.5), last action, command. Critic (no noise): the same plus privileged terms (foot height, foot air time, foot contact, contact forces). That split is asymmetric actor-critic (`concepts/actor-critic.md`).

Rewards (weight): track linear velocity exp(−(‖v_cmd,xy − v_xy‖² + v_z²)/σ²) with σ² = 0.25, body frame (2.0); track yaw rate, σ² = 0.5 (2.0); upright (1.0); posture with per-joint tolerances for standing, walking and running (1.0); body angular velocity (−0.05, G1 override); angular momentum (−0.02); joint limits (−1.0); action rate L2 (−0.1); foot clearance (−2.0); swing height (−0.25); foot slip (−0.1); soft landing (−1e−5); self collision (−1.0, G1); air time (0.0 for G1).

Terminations: time-out at 20 s (flagged time_out=True, so it is bootstrapped); fell over when tilt exceeds 70°.

Randomization: foot friction 0.3 to 1.2 (shared across foot geoms), encoder bias ±0.015 rad, torso center-of-mass offset (±2.5 cm x/y, ±3 cm z), random pushes every 1 to 3 s.

## Reward design rules `[practice]`

- Shape for the task with a bounded kernel: exp(−error²/σ²) in (0, 1] keeps tracking terms on one scale and gives gradient near the target; σ sets how close counts as "good". Both legged_gym and mjlab use it.
- Keep alive bonuses small or absent and make falling end the episode. A large positive per-step reward plus termination teaches "survive", which is exactly the standing-still optimum seen in small-N G1 runs (`on-policy-at-scale.md`).
- Penalties regularize the gait for hardware: action rate (smoothness), joint limits, torques or energy, foot slip, self collision. Start them small; heavy penalties early stop the robot from moving at all. Curricula on penalty weights are common.
- Log every reward term separately (`Episode_Reward/*`) and check which one is moving. A term that dominates the sum is the one the policy optimizes.
- Reward scale interacts with the value function: Q magnitudes scale like r_max/(1−γ); observation and reward normalization keep the critic's targets well conditioned.

## Observation design rules `[practice]`

- Give the actor only what the robot will have (IMU, joint encoders, last action, command); give the critic anything (asymmetric actor-critic).
- Include the last action: the policy cannot otherwise know where its PD targets currently are.
- Partial observability (latency, unobserved friction or payload) is handled by history (stacked observations or an RNN) or by a learned latent (RMA, teacher-student); see `sim-to-real.md`. Much RL theory assumes full state, justified by external state estimation or finite history stacking (Notes 0).
- Noise and bias in training observations should match the real sensors' magnitudes, not be "a bit of noise".

## Terminations and truncation

| Episode end | Meaning | Bootstrap? |
|---|---|---|
| Time limit | the world did not end, we stopped watching | yes: target uses γV(s_{t+1}) |
| Fall, unsafe state | the MDP ends (absorbing, zero future reward) | no |
| Out of bounds of a finite terrain | practical cut, not a failure | yes (mjlab flags it time_out=True) |

Getting this wrong biases values toward "the world ends at step H". How much it matters depends on episode length T relative to the effective horizon 1/(1−γ): with γ = 0.99 and 1000-step episodes the bias is about 10% averaged over an episode's states and concentrated in the last few hundred steps (`scripts/rl_math.py` truncation_bias), and Andrychowicz et al. saw no difference on MuJoCo with 1000-step limits (`papers/canon-algorithms.md`). It matters much more when episodes are short relative to 1/(1−γ), or when rollout segment ends are mistaken for episode ends. Gymnasium returns `terminated` and `truncated` separately for this reason; rsl_rl receives `time_outs` in extras. `scripts/rl_math.py` shows the bias numerically.
