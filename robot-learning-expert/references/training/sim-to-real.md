# Sim-to-real for learned robot controllers

A policy trained in simulation fails on hardware when the real robot sits outside the distribution of worlds it trained on. The working answer since 2017: train on many randomized worlds (so reality looks like one more sample), give the policy enough memory or history to infer which world it is in, and use privileged information only where the deployed robot will not need it (the critic, or a teacher that is distilled away). Paper facts with locations: `papers/canon-robotics.md` (verified; its "Recipe" section lists which paper supports each component). Simulator and library facts: `simulators-and-tooling.md`.

## Terms

| Term | Meaning |
|---|---|
| reality gap | mismatch between simulated and real dynamics, sensing and actuation |
| domain randomization (DR) | sample simulator parameters (friction, mass, gains, latency, visuals) per episode or per env so the policy must work across them |
| dynamics randomization | DR restricted to physical parameters (Peng et al. 2018) |
| ADR | automatic DR: widen each parameter's range while the policy succeeds at its boundary (OpenAI 2019) |
| privileged information | simulator state the robot cannot measure (true velocity, contact forces, friction, terrain) |
| asymmetric actor-critic | critic sees privileged information, actor sees only deployable observations (Pinto et al. 2018) |
| teacher-student | train a teacher with privileged inputs by RL, then train a student on onboard sensors to imitate it (Lee et al. 2020; RMA regresses the teacher's latent from history) |
| actuator network | a learned model of the real motor's torque response, used inside the simulator (Hwangbo et al. 2019) |
| sim-to-sim | test the policy in a second simulator before hardware (Radosavovic et al. 2024; ASAP) |

## The pipeline, and where the shipped mjlab G1 task stands

Components from `papers/canon-robotics.md` "Recipe"; the right column was read from mjlab 1.6.0 source (`velocity_env_cfg.py`, `config/g1/env_cfgs.py`, `g1_constants.py`), details in `reward-and-observation-design.md`.

| Component | Literature | mjlab G1 velocity task (shipped) |
|---|---|---|
| GPU-parallel PPO, thousands of envs | Rudin 2021, Radosavovic 2024 | yes: rsl_rl PPO |
| PD joint-position targets, not torques | Hwangbo 2019, RMA, Radosavovic | yes: offsets from default pose, scale 0.25·effort/stiffness; gains from armature at 10 Hz, damping ratio 2 |
| Policy 30 to 200 Hz, PD faster | Hwangbo 200 Hz, RMA 100 Hz, Lee and Miki 50 Hz | 50 Hz policy (physics dt 0.005 s × decimation 4); the position actuators apply PD every 200 Hz physics step |
| Proprioception plus short history | RMA 50 steps, Lee up to 2 s, Radosavovic 16 steps | current frame only, plus last action (no history) |
| Dynamics randomization (mass, COM, friction, gains or motor strength, latency) | Peng 2018 Tab. I, OpenAI 2018 Tab. 1, RMA Tab. I | foot friction 0.3 to 1.2, torso COM offset, encoder bias ±0.015 rad; no mass, gain, motor-strength or latency randomization |
| Observation noise and delays | Peng 2018 (no obs noise: real success 0.25 vs 0.89), OpenAI 2018 | uniform noise on every actor term; no delay |
| Actuator modeling | Hwangbo (actuator net), ASAP (delta actions on G1 ankle) | armature and gains from motor specs; parallel-linkage joints approximated as 2× (`g1_constants.py` comment) |
| Pushes and external forces | Miki 2022, OpenAI | velocity pushes every 1 to 3 s |
| Asymmetric critic | Pinto 2018 | yes: critic adds foot height, air time, contacts, contact forces, without noise |
| Teacher-student distillation | Lee 2020, Miki 2022, RMA | no |
| Memory to infer dynamics | Peng 2018 (LSTM 0.89 vs feedforward 0.67) | no (MLP actor) |
| Curriculum | Lee, OpenAI 2019 ADR, Rudin 2021 | command-velocity curriculum (step-count based) |
| Bootstrapped time-outs | Rudin 2021 §2.2.2 | yes |
| Sim-to-sim check | Radosavovic, ASAP | not in the task; replay in CPU MuJoCo or another simulator is the cheap version [inference] |

What that means for a real G1 deployment [inference from the table]: the shipped task is a training demo, not a deployment recipe. For velocity locomotion the usual additions, in order of how often papers show them mattering (locomotion-specific: for humanoid motion tracking BeyondMimic found that adding history made tracking worse and that randomizing delay "makes learning more difficult", `motion-tracking.md`): action latency and observation delay, PD gain or motor-strength randomization, link mass randomization, an observation history or recurrent actor, then sim-to-sim validation. Each addition is a new experiment arm; the G1 case-study sweep deliberately changed none of these.

## Rules of thumb `[practice]`

- Randomize what you cannot measure, measure what you can: calibrate first (motor constants, masses from CAD), randomize around the calibrated values (ADR starts from calibrated values, OpenAI 2019 §5.1).
- Too-wide randomization produces conservative, slow policies; ranges should cover the plausible real values, not everything. Two ways to set them: DORAEMON widens the training distribution as far as a success constraint allows (success target α, KL trust region): on one real manipulation task (SAC, history-conditioned policy, 17 randomized parameters, 30 real rollouts) fixed wide DR gave 0.02% sim and 0% real success versus 66.57% sim and 60% real, with No-DR at 15.06% and 13.33% [2311.01885 Table 1 lines 595-601]. DrEureka's RAPP trains once without DR, sweeps each parameter alone, keeps the interval where the policy still succeeds, and randomizes inside it; using the full sweep range ("Prompt DR") over-exerted forces and hurt real performance [2406.01967 lines 190-203, 437-439; small real-robot n].
- ADR (OpenAI 2019) can stall or underperform uniform randomization when the difficulty landscape is rugged or has unfeasible subspaces: in TeachMyAgent ADR was significantly worse than Random (p < 0.05) on the rugged-difficulty challenge [2103.09815 lines 419-423; Box2D walkers, a re-implementation, not OpenAI's robot ADR]. OpenAI's own ADR hyperparameters: boundary sampling probability 0.5, step size 0.02, increase threshold 20 and decrease threshold 10 successes, performance queue 240, max φ 4.0 [1910.07113 lines 2610-2617]. Their curriculum finding: the larger the fixed randomization entropy, the longer training from scratch took, and it may become infeasible, while ADR starts from zero range and progressed faster [1910.07113 lines 1231-1236].
- Match observation noise to the real sensor's measured noise and bias; noise-free training observations were the single biggest failure in Peng et al.'s ablation.
- Too-heavy regularization rewards yield a policy that stands still (Hwangbo et al. note this; the G1 sweep's small-N runs show the same local optimum).
- Evaluate in sim under the training randomization and outside it before touching hardware; report both (`evaluation.md`).

## Teacher-student: realizability and what to distill with

- Realizability caveat: DAgger-style distillation has an irreducible floor when the student cannot infer the teacher's privileged variable from its observation history (state aliasing makes labels conflict, and repeated aggregation can raise the error). Fixes proposed: a history or recurrent student when the quantity is inferable, querying the teacher only at critical states (CritiQ), or RL for the student with resets to teacher states [2505.09546 lines 134-146; tiny search tasks, 12 real episodes, from the findings report for the evidence scale (unverified)]. Related theory: `papers/power-of-resets.md`.
- Specialist-to-generalist (EAGLE, 2602.02960): keep the PPO loss and add distillation terms, L = L_PPO + α·L_a + β·L_e with α = 0.02 and β = 1 [lines 275-281, 311]; 4096 envs with robot types sampled across them [line 311 area]. Its comparison with Kickstarting is confounded (loss form vs latent loss; no seeds reported) (from the findings report, unverified).
- Hybrid DAgger + PPO for motion skills (lambda_D decay to a floor of 0.1, pre-train the critic) is from the findings report and not re-verified (unverified).

## Motion tracking and contact-rich manipulation

- Humanoid motion-tracking DR, delay and history findings: `motion-tracking.md`.
- Contact-rich assembly (FORGE, 2408.04587): the paper attributes the main real gains to force observations plus an excessive-force penalty; dynamics randomization did not significantly change nominal success, and mattered for robustness to controller gains [lines 306-308]. Other pieces (force-threshold conditioning, success-prediction termination, AutoMate 86.5% and DextrAH-G 87% zero-shot) come from the findings report and were not re-read (unverified). Precise FORGE Table I values were not re-confirmed.
- Manipulation DR ranges worth copying as starting points (DemoBot actuator model, POISE ranges: findings report, unverified): gains and motor strength about x U(0.9, 1.1), friction x U(0.5, 2), delay up to 150 ms. They fill the gain, motor-strength and latency gaps in the shipped G1 task row above.

## Related

Notes 0 asks whether a dynamics model exists, how accurate it is, and whether it is a simulator or hardware; Notes 7 §3.2 argues that sim-to-real training should exploit the simulator's resets instead of acting as if it were the real world, which links this file to `../concepts/resets-and-exploration.md`, `papers/power-of-resets.md` and SuperMimic's return-based reset sampling (`papers/supermimic.md`).
