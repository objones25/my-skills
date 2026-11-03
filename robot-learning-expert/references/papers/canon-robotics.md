# Canon: sim-to-real and robot learning papers

Eighteen papers that the rest of the field cites when it talks about sim-to-real transfer, legged locomotion, imitation and robot foundation models. Built 2026-11-02 for a G1 humanoid walking project (mjlab 1.6.0, rsl-rl-lib 5.4.2, PPO on one A100).

How this was checked: titles, author lists, first-version dates and arXiv comments come from the arXiv API (https://export.arxiv.org/api/query?id_list=..., checked 2026-11-02). Facts come from the arXiv PDFs (https://arxiv.org/pdf/<id>, fetched and read 2026-11-02), with locations such as [§IV-C], [Table 1] or [Fig. 4]. Science Robotics papers have no numbered sections, so their locations name the subsection heading. Where a venue is not printed on the arXiv page, the venue line says which other page was opened. [inference] marks a statement that is my synthesis, not something a paper says.

## Terms used below

- Sim-to-real (sim2real): train a controller in a physics simulator, then run it on the physical robot. The "reality gap" is everything the simulator gets wrong.
- Domain randomization (DR): during training, sample simulator settings (textures, masses, friction, delays) from a range each episode, so the real world looks like one more sample. Visual DR randomizes rendering; dynamics randomization randomizes physics.
- PD control: the policy outputs joint position targets q*, and a proportional-derivative controller turns them into torques, τ = Kp (q* − q) + Kd (q̇* − q̇), usually with target velocity q̇* = 0. Kp is stiffness, Kd is damping.
- Policy frequency vs PD frequency: the policy picks a new target every policy step (for example 50 Hz); the PD loop runs faster (for example 1 kHz) between policy steps.
- Proprioception: the robot's internal sensing (joint encoders, IMU). Exteroception: sensing of the outside world (cameras, LiDAR, height maps).
- Privileged information: simulator quantities that the real robot cannot measure (true friction, contact forces, terrain heights).
- Asymmetric actor-critic: the critic (value function, only used in training) gets privileged state; the actor (policy, deployed) gets only real sensor inputs.
- Teacher-student (privileged learning, distillation): first train a teacher policy with privileged inputs using RL, then train a student policy with deployable inputs to copy the teacher.
- DAgger (dataset aggregation): roll out the student, label each visited state with the teacher's action, train on those labels. This trains on the student's own state distribution.
- Curriculum: change task difficulty during training based on progress.
- LSTM / TCN: recurrent network with memory / temporal convolutional network over a fixed window of past inputs. Both let a policy infer hidden dynamics from history.
- Action chunking: the policy predicts the next k actions at once instead of one.
- VLA (vision-language-action model): a pretrained vision-language model fine-tuned to output robot actions.
- PPO, TRPO: on-policy policy-gradient algorithms (see `trpo.md` and `legged-gym.md` in this folder).

---

## 1. Tobin et al. 2017: Domain Randomization (visual DR)

- Title: Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World
- Authors: Josh Tobin, Rachel Fong, Alex Ray, Jonas Schneider, Wojciech Zaremba, Pieter Abbeel (6)
- Venue as shown: arXiv comment says "Submitted to 2017 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2017)". A web search lists IEEE Xplore document 8202133 for it, but that page returned no content, so acceptance at IROS 2017 is not confirmed from an opened page.
- arXiv: 1703.06907 (v1, 20 Mar 2017)
- Read it when: you need the origin of visual DR, or evidence for which randomizations matter.

Facts:
1. Task is object localization from one monocular RGB camera, not control; the detector is accurate to about 1.5 cm on real images and is used to grasp objects in clutter. First claimed transfer of a deep net trained only on simulated RGB (no real-image pre-training) to robot control [Abstract].
2. Randomized per training sample: number and shape of distractor objects; position and texture of all table objects; textures of table, floor, skybox and robot; camera position, orientation and field of view; number of lights; light position, orientation and specular properties; type and amount of image noise [§III-A]. Textures are a random RGB value, a gradient, or a checker pattern; 0 to 10 distractors; camera placed within a 10 × 5 × 10 cm box around a nominal pose [§III-A].
3. Ablation (real images, geometric shapes, error in cm) [Table II]: full method 1.3 / 1.8 / 2.4 (object only / distractors / occlusions); no camera randomization 2.0 / 2.4 / 2.9; no distractors in training 1.5 / 7.2 / 7.4; no noise 1.4 / 1.9 / 2.4. Distractors in training drive robustness to real clutter; noise had negligible effect on accuracy but helped convergence [§IV-C].
4. With few training images, texture variety mattered more than object-position variety [§IV-C, Fig. 5].
5. Deployed on a Fetch robot: picked the target in 38 of 40 trials, including heavy clutter and occlusion [§IV-D].

## 2. Peng et al. 2018: Sim-to-Real Transfer with Dynamics Randomization

- Title: Sim-to-Real Transfer of Robotic Control with Dynamics Randomization
- Authors: Xue Bin Peng, Marcin Andrychowicz, Wojciech Zaremba, Pieter Abbeel (4)
- Venue as shown: arXiv lists DOI 10.1109/ICRA.2018.8460528 (an ICRA 2018 DOI). No other page opened.
- arXiv: 1710.06537 (v3; v1 18 Oct 2017)
- Read it when: you want the parameter list and ranges for dynamics randomization, or evidence that memory (LSTM) helps transfer.

Facts:
1. Task: push a puck to a goal with a 7-DOF Fetch arm, sparse binary reward, trained with Hindsight Experience Replay (HER: relabel failed episodes with goals that were reached) and Recurrent Deterministic Policy Gradient [§IV-A, §IV-E].
2. State is 52-D (arm joint positions and velocities, gripper position, puck pose and velocities). Actions are target joint angles for a position controller, given as offsets from current joint angles (7-D) [§IV-B].
3. 95 randomized parameters, sampled once per episode except action timestep and observation noise, which change every step [§IV-C]. Ranges [Table I]: link mass 0.25 to 4 × default; joint damping 0.2 to 20 × default; puck mass 0.1 to 0.4 kg; puck friction 0.1 to 5; puck damping 0.01 to 0.2 N·s/m; table height 0.73 to 0.77 m; position-controller gains 0.5 to 2 × default; action timestep Δt ~ 0.04 s + Exp(λ) with λ in [125, 1000] s⁻¹ (a model of control latency). Mass, damping, friction and gains are sampled log-uniformly [text beside Table I]. Observation noise is independent Gaussian per state feature [§IV-C].
4. Value function receives the true dynamics parameters μ, the policy does not (an early asymmetric critic) [§IV-E].
5. Real-robot success [Table II]: LSTM 0.89 (28 trials); feedforward with randomization 0.67; feedforward plus history 0.70; feedforward without randomization 0.0. Holding parameters fixed during training hurts real success [Table III]: fixed action timestep 0.29, no observation noise 0.25, fixed link mass 0.64, fixed puck friction 0.48 (vs 0.89 with all). Little calibration was done; sim and real joint trajectories differed significantly [§V, Fig. 5].

## 3. OpenAI 2018: Learning Dexterous In-Hand Manipulation

- Title: Learning Dexterous In-Hand Manipulation
- Authors: OpenAI, Marcin Andrychowicz, Bowen Baker, Maciek Chociej, Rafal Jozefowicz, ... (17 listed)
- Venue as shown: arXiv preprint only; the arXiv comment asks that it be cited as "Learning Dexterous In-Hand Manipulation" by OpenAI et al. A later journal version was not checked.
- arXiv: 1808.00177 (v5; v1 1 Aug 2018)
- Read it when: you need a full list of physics, unmodeled-effect and visual randomizations with ranges, or an asymmetric-critic observation table.

Facts:
1. Shadow Dexterous Hand (24 DoF, 20 tendon pairs) reorients a block. The policy sends relative joint position targets at about 12 Hz to a low-level controller at about 1 kHz [§2.1]. Simulated in MuJoCo, rendered with Unity [§2.2].
2. Physics randomization, sampled per episode [Table 1, §3.2]: object dimensions ×U(0.95, 1.05); object and link masses ×U(0.5, 1.5); surface friction ×U(0.7, 1.3); joint damping ×loguniform(0.3, 3.0); actuator force gains (P term) ×loguniform(0.75, 1.5); joint limits +N(0, 0.15) rad; gravity vector +N(0, 0.4) m/s² per coordinate. Also "unmodeled effects": motor backlash, action delay and action noise, marker tracking dropouts and occlusion, small random forces on the object; and observation noise with a per-episode correlated part plus a per-step part [§3.2].
3. Asymmetric actor-critic [§4.1, Table 2]: policy sees fingertip positions (15-D), object position, relative target orientation; value network additionally sees object orientation, target orientation, all 24 joint angles and velocities, and object velocities. Actions are discretized into 11 bins per joint because discrete actions trained better than a Gaussian [§4.2].
4. Ablation on the real robot, median consecutive successes [Table 4, §6.3]: all randomizations 13; no randomizations 0; no physics randomizations 2; no unmodeled effects 2; no observation noise 8.5. In simulation the same policy reaches a median of 50 [§6.2].
5. Memory matters: LSTM policy median 13 vs feedforward policy 3.5 on the real robot [Table 5, §6.4]. Scale: 384 machines × 16 CPU cores (6,144 cores) plus 8 GPUs, about 2 years of simulated experience per hour; full randomization needs about 100 years of experience (about 50 hours) vs 3 years without randomization [§4.3, §6.3, §6.5].

## 4. OpenAI 2019: Solving Rubik's Cube with a Robot Hand (ADR)

- Title: Solving Rubik's Cube with a Robot Hand
- Authors: OpenAI, Ilge Akkaya, Marcin Andrychowicz, Maciek Chociej, Mateusz Litwin, ... (19 listed)
- Venue as shown: arXiv preprint ("A PREPRINT - OCTOBER 17, 2019" in the page header).
- arXiv: 1910.07113 (v1, 16 Oct 2019)
- Read it when: you want automatic domain randomization (ADR) or evidence that memory plus wide randomization yields in-context adaptation.

Facts:
1. ADR (automatic domain randomization): instead of hand-set ranges, each randomized parameter λᵢ has a uniform range [φᵢᴸ, φᵢᴴ] that starts at a single calibrated value and widens as the policy improves [§5.1-5.2].
2. Algorithm [§5.2]: with probability p_b ("boundary sampling probability"), pick one parameter, pin it to its lower or upper boundary, sample the rest normally, record performance into a buffer for that boundary. When the buffer is full, average it: above a high threshold t_H, widen that boundary; below a low threshold t_L, shrink it. For policies, performance is the number of successes per episode. Otherwise (prob. 1 − p_b) sample normally and generate training data.
3. ADR entropy, H = (1/d) Σ log(φᵢᴴ − φᵢᴸ) in nats per dimension, measures how wide the distribution has grown [§5.2].
4. Real results, fair-scramble goal sequence, N = 10 [Table 6, §8.4.1]: manual DR policy median 2 successes, 0% half / 0% full scramble; ADR (XXL, months of training) median 22, 60% half and 20% full scramble with the Giiker sensor cube; 20% / 0% when face angles came from vision. The solve sequence comes from the Kociemba solver; learning handles only sensing and control [§2.2].
5. Actions: 11 bins per joint over 20 actuated joints, relative joint position changes; rewards for distance decrease, +5 per goal, −20 for a drop [§6.1]. LSTM policies trained under ADR show signs of meta-learning (adapting within an episode to perturbations) [§9].

## 5. Pinto et al. 2018: Asymmetric Actor Critic

- Title: Asymmetric Actor Critic for Image-Based Robot Learning
- Authors: Lerrel Pinto, Marcin Andrychowicz, Peter Welinder, Wojciech Zaremba, Pieter Abbeel (5)
- Venue as shown: Robotics: Science and Systems XIV (RSS 2018), paper p08 (https://roboticsproceedings.org/rss14/p08.html, opened 2026-11-02).
- arXiv: 1710.06542 (v1, 18 Oct 2017)
- Read it when: justifying a critic that sees more than the actor (mjlab's G1 critic does this).

Facts:
1. Actor gets RGB or RGBD images; critic gets the full simulator state (and full goal state). Since the critic is only used in training, deployment needs only images [§I, Fig. 2].
2. Built on DDPG plus HER. Asymmetric versions beat symmetric (images to both) versions by large margins in simulation, most on the hard sparse-reward Fetch Pick task [§V-C, Fig. 5].
3. Visual DR over textures (random RGB, gradient, checker), lighting, camera location and depth [§IV-C]. An auxiliary "bottleneck" loss that predicts full state from the actor's features further speeds training [§V-E].
4. Real Fetch arm, 5 trials per task [Table I]: asymmetric HER 5/5 on Pick, Forward Push and Block Move; asymmetric DDPG, symmetric HER and behavior cloning mostly 0/5; DAgger behavior cloning 3/5 on Pick. Without any randomization 0/5 on all; without viewpoint randomization 0/5, 0/5, 4/5.

## 6. Hwangbo et al. 2019: Agile and dynamic motor skills (actuator network)

- Title: Learning agile and dynamic motor skills for legged robots
- Authors: Jemin Hwangbo, Joonho Lee, Alexey Dosovitskiy, Dario Bellicoso, Vassilios Tsounis, ... (7)
- Venue as shown: arXiv journal_ref "Science Robotics 4.26 (2019): eaau5872", DOI 10.1126/scirobotics.aau5872.
- arXiv: 1901.08652 (v1, 24 Jan 2019)
- Read it when: your simulated actuators do not match the real ones; you want the "learn the actuator from real data" option.

Facts:
1. Actuator network: an MLP (3 hidden layers × 32 units, softsign) maps a history of joint position errors and velocities at t, t − 0.01 s, t − 0.02 s to joint torque. Trained by supervised learning on more than a million samples collected at 400 Hz from the 12 real actuators of ANYmal [Materials and Methods: "Modeling the actuation"]. Average validation torque error 0.740 Nm vs 3.55 Nm for an ideal actuator model (0.966 vs 5.74 Nm on data from trained policies) [same section, Fig. 6].
2. Policy outputs joint position targets; the paper's PD reference uses fixed gains kp = 50 Nm/rad, kd = 0.1 Nm/(rad/s), and target velocity 0. kp is chosen as roughly nominal torque range (±30 Nm) over nominal motion range (±0.6 rad) [Materials and Methods: "Observation and action"]. Position targets train better than torques because the policy starts as a standing controller [same section].
3. Policy evaluated at 200 Hz for locomotion and 100 Hz for fall recovery [Materials and Methods, last subsection]. Observation includes the gravity vector in the IMU frame, base twist, joint states, previous action and joint-state history at t − 0.01 s and t − 0.02 s [Materials and Methods: "Observation and action"].
4. Randomization: 30 ANYmal models with sampled inertia; COM positions +U(−2, 2) cm, link masses +U(−15, 15)%, joint positions +U(−2, 2) cm [Materials and Methods: simulation subsection]. TRPO; a multiplicative curriculum factor ramps cost coefficients from k₀ to 1 to avoid the "stand still" local optimum [Materials and Methods: "Policy training details"].
5. Results: beat ANYmal's previous speed record by 25% (1.5 m/s on hardware); 23 to 36% less torque than prior gaits; fall recovery worked on the first hardware attempt. Training on a single CPU plus GPU desktop, each run under 11 hours. Policies trained with an ideal actuator model or with an analytical model hand-tuned for over a week could not take a single step without falling; the limbs shook violently, likely from unmodeled delays [Results; Movies S4, S5].

## 7. Lee et al. 2020: Quadrupedal locomotion over challenging terrain (teacher-student)

- Title: Learning Quadrupedal Locomotion over Challenging Terrain
- Authors: Joonho Lee, Jemin Hwangbo, Lorenz Wellhausen, Vladlen Koltun, Marco Hutter (5)
- Venue as shown: arXiv journal_ref "Science Robotics 2020 Vol. 5, Issue 47, eabc5986", DOI 10.1126/scirobotics.abc5986.
- arXiv: 2010.11251 (v1, 21 Oct 2020)
- Read it when: designing a privileged teacher and a history-based student, or an adaptive terrain curriculum.

Facts:
1. Teacher input = robot observation oₜ plus privileged xₜ: terrain profile (9 height samples on a 10 cm circle around each foot), foot contact states and forces, friction coefficients, external disturbance forces. An MLP encoder compresses xₜ to a latent l̄ₜ; trained with TRPO [Materials and Methods: "Teacher policy"].
2. Student is a TCN over a history of N proprioceptive observations. Loss = (teacher action − student action)² + (teacher latent − student latent)², trained with DAgger (student rolls out, teacher labels) [Materials and Methods: "Student policy"]. Ablation: a TCN student trained directly with RL, without the teacher, learns much worse [Fig. 5F]. Longer history (TCN-100, 2 s) beats TCN-1 [Fig. 5B-D].
3. Action space (PMTG, policies modulating trajectory generators): 16-D, per-leg frequency offsets plus foot position residuals added to a periodic foot trajectory with base frequency 1.25 Hz; analytic inverse kinematics then joint PD control [Materials and Methods: "Motion synthesis"; action defined under "Teacher policy"]. Policy runs at 50 Hz, the joint PD loop at 400 Hz [Fig. 4C].
4. Adaptive terrain curriculum: a particle filter tracks terrain parameters whose traversability (fraction of steps making forward progress) lies in [0.5, 0.9], so terrains are neither too easy nor too hard. Used in both teacher and student training [Materials and Methods: "Adaptive terrain curriculum"].
5. Proprioception only (no camera or LiDAR); deployed zero-shot on ANYmal-B and ANYmal-C over mud, snow, rubble, vegetation and water [Results]. Listed limitation: the gait pattern is narrower than animal gaits, and the controller does not use perception [Discussion].

## 8. Kumar et al. 2021: RMA, Rapid Motor Adaptation

- Title: RMA: Rapid Motor Adaptation for Legged Robots
- Authors: Ashish Kumar, Zipeng Fu, Deepak Pathak, Jitendra Malik (4)
- Venue as shown: Robotics: Science and Systems XVII (RSS 2021), paper p011 (https://roboticsproceedings.org/rss17/p011.html, opened 2026-11-02); arXiv comment also says "RSS 2021".
- arXiv: 2107.04034 (v1, 8 Jul 2021)
- Read it when: you want online adaptation from state-action history without a phase or trajectory generator.

Facts:
1. Two phases [§III, Fig. 1]. Phase 1: base policy π(xₜ, aₜ₋₁, zₜ) trained with PPO, where zₜ ∈ R⁸ ("extrinsics") = μ(eₜ) encodes the privileged 17-D environment vector eₜ (payload mass and position, 12 motor strengths, friction, local terrain height). Phase 2: an adaptation module φ (1-D CNN) regresses ẑₜ from the last 50 state-action pairs (0.5 s), MSE against zₜ, using on-policy rollouts [§III-B, §IV-A].
2. Deployment: base policy at 100 Hz, adaptation module at 10 Hz, asynchronous [§III-C]. State 30-D (12 joint positions, 12 velocities, torso roll and pitch, 4 binary foot contacts); action = 12 desired joint angles, PD with Kp = 55, Kd = 0.8, target velocity 0 [§IV-A]. No reference gait or trajectory generator.
3. Randomization ranges, training / testing [Table I]: friction 0.05 to 4.5 / 0.04 to 6.0; Kp 50 to 60 / 45 to 65; Kd 0.4 to 0.8 / 0.3 to 0.9; payload 0 to 6 / 0 to 7 kg; COM ±0.15 / ±0.18 cm (as printed); motor strength 0.90 to 1.10 / 0.88 to 1.22; resample probability per step 0.004 / 0.01. Fractal terrain from RaiSim [§IV-A].
4. Real A1 results, 5 trials each: e.g. 15 cm step down 100% for RMA vs 0% without adaptation; uneven foam 80% vs 0% [Fig. 3]. Without adaptation the robot fails to move with payloads above 8 kg [Fig. 3 caption].
5. Printed inconsistency: §IV-A gives policy control frequency 100 Hz and "simulation time step is 0.025 s" in the same paragraph; 0.025 s would be 40 Hz. Treat the sim timestep as unclear.

## 9. Miki et al. 2022: Robust perceptive locomotion

- Title: Learning robust perceptive locomotion for quadrupedal robots in the wild
- Authors: Takahiro Miki, Joonho Lee, Jemin Hwangbo, Lorenz Wellhausen, Vladlen Koltun, Marco Hutter (6)
- Venue as shown: arXiv journal_ref "Science Robotics, 19 Jan 2022, Vol 7, Issue 62", DOI 10.1126/scirobotics.abk2822.
- arXiv: 2201.08117 (v1, 20 Jan 2022)
- Read it when: adding a height map or camera to a teacher-student pipeline and needing robustness to bad perception.

Facts:
1. Teacher (PPO, MLP) sees proprioception, height samples around each foot at five radii, and privileged state: contact states, forces and normals, friction, thigh and shank contacts, external force and torque, swing phase duration [Materials and Methods: "Observation and Action"].
2. Student = recurrent "belief state encoder" with a gate that learns how much to trust exteroception, plus the teacher's MLP head (weights reused to initialize). Loss = behavior cloning (squared distance to teacher action) + reconstruction (decode noiseless height samples and privileged state from the belief). Trained on student rollouts [Materials and Methods: "Student policy training"].
3. Height-sample noise model during student training: lateral shifts of scan points and height perturbations, each at three scopes (per point, per foot, per episode), with three regimes (nominal, large per-foot offsets for pose drift, large per-point noise for missing maps) [Materials and Methods: "Height sample randomization"].
4. Action: per-leg phase difference plus residual joint position targets on a CPG-style nominal stepping trajectory, inverse kinematics, then joint PD [Materials and Methods: "Observation and Action"]. Policy at 50 Hz, elevation map at 20 Hz [Fig. 5]. DR: body and leg masses, initial state, external force and torque, occasional low foot friction [Materials and Methods: "Domain randomization"].
5. Results and a failure: 2.2 km Alpine hike with 120 m elevation gain, summit in 31 min (signage says 35), no falls; reliable steps up to 30.5 cm vs frequent hind-leg failures for the proprioceptive baseline [Results]. Stairs modeled as height-map ramps were exploited by the policy through non-vertical edges and transferred poorly, so stairs were rebuilt from boxes [Materials and Methods: terrain subsection].

## 10. Peng et al. 2018: DeepMimic

- Title: DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills
- Authors: Xue Bin Peng, Pieter Abbeel, Sergey Levine, Michiel van de Panne (4)
- Venue as shown: page header "ACM Trans. Graph. 37, 4, Article 143 (August 2018)"; DOI 10.1145/3197517.3201311. (SIGGRAPH 2018 papers appear in this TOG issue; "SIGGRAPH" is not printed on the arXiv page.)
- arXiv: 1804.02717 (v3; v1 8 Apr 2018)
- Read it when: imitating a reference motion with RL, including humanoid motion tracking tasks.

Facts:
1. Reward = w_I·r_I (imitation) + w_G·r_G (task). Imitation reward weights: pose 0.65, velocity 0.1, end-effector 0.15, center of mass 0.1 [§5].
2. Actions are PD target orientations per joint (axis-angle for spherical joints); policy at 30 Hz; physics at 1.2 kHz in Bullet with stable PD controllers [§4, §9]. A phase variable φ ∈ [0, 1] is part of the state [§4].
3. Reference state initialization (RSI): start episodes at random states sampled from the reference motion. Early termination (ET): end the episode when certain links (torso, head) touch the ground [§6.1, §6.2].
4. Ablation, normalized return [Table 5]: backflip 0.791 with RSI + ET, 0.730 ET only, 0.379 RSI only (no ET); walking is insensitive (0.980 / 0.981 / 0.974). RSI and ET matter most for dynamic skills with flight phases [§10.4].

## 11. Peng et al. 2021: AMP, Adversarial Motion Priors

- Title: AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control
- Authors: Xue Bin Peng, Ze Ma, Pieter Abbeel, Sergey Levine, Angjoo Kanazawa (5)
- Venue as shown: page header "ACM Trans. Graph. 40, 4, Article 1 (August 2021)"; DOI 10.1145/3450626.3459670.
- arXiv: 2104.02180 (v2; v1 5 Apr 2021)
- Read it when: you want natural-looking gaits from a motion dataset without a tracking reward or phase variable.

Facts:
1. Reward r = w_G·r_G (task) + w_S·r_S (style) with w_G = w_S = 0.5 for all tasks [Eq. 4, §8.2, appendix hyperparameter table].
2. Style reward comes from a discriminator D on state transitions (s, s′), trained with a least-squares GAN loss: data → +1, policy → −1; r_S = max(0, 1 − 0.25 (D(s, s′) − 1)²) [§5.2, Eq. 7].
3. Gradient penalty on dataset samples with w_gp = 10; the ablation names it the most vital component for stable training [§5.4, §8.1, §8.5].
4. Policy at 30 Hz outputs PD targets; simulation at 1.2 kHz. No synchronization or phase with the reference clips is needed [§6, §8.1]. Limitation: still prone to mode collapse on large, diverse datasets [§9].

## 12. Chi et al. 2023: Diffusion Policy

- Title: Diffusion Policy: Visuomotor Policy Learning via Action Diffusion
- Authors (RSS version): Cheng Chi, Siyuan Feng, Yilun Du, Zhenjia Xu, Eric Cousineau, Benjamin Burchfiel, Shuran Song. The arXiv v5 (14 Mar 2024) is "an extended journal version of the original RSS2023 paper" with 8 authors, adding Russ Tedrake.
- Venue as shown: Robotics: Science and Systems XIX (RSS 2023), paper p026 (https://roboticsproceedings.org/rss19/p026.html, opened 2026-11-02).
- arXiv: 2303.04137 (v5; v1 7 Mar 2023). Locations below refer to v5.
- Read it when: doing imitation learning from demonstrations with multimodal behavior.

Facts:
1. The policy is a conditional denoising diffusion model (DDPM: start from Gaussian noise, iteratively denoise into an action sequence). Average 46.9% improvement over prior methods across 15 tasks from 4 benchmarks [Abstract].
2. Takes the last T_o observations, predicts T_p future actions, executes T_a of them, then replans (receding horizon) [§2.3]. Typical values: T_o = 2, T_a = 8, T_p = 16, position control, 100 training diffusion steps, 100 eval steps in sim and 16 on real tasks [Table 7]. Action horizon 8 was best for most tasks [§4.3].
3. With DDIM, 100 training iterations and 10 inference iterations give about 0.1 s latency on an RTX 3080 [§3.4]. Holds peak performance with up to 4 steps of simulated latency [§4.3].
4. Position control worked better than velocity control, the opposite of most prior BC work [§4.2, Fig. 4]. Vision: ResNet-18 without pretraining, BatchNorm replaced by GroupNorm [§3.2].
5. Limitation: inherits behavior-cloning limits, e.g. poor results with too few demonstrations [§9].

## 13. Zhao et al. 2023: ACT and ALOHA

- Title: Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware
- Authors: Tony Z. Zhao, Vikash Kumar, Sergey Levine, Chelsea Finn (4)
- Venue as shown: Robotics: Science and Systems XIX (RSS 2023), paper p016 (https://roboticsproceedings.org/rss19/p016.html, opened 2026-11-02).
- arXiv: 2304.13705 (v1, 23 Apr 2023)
- Read it when: action chunking, low-cost teleoperation, or a transformer BC baseline.

Facts:
1. ALOHA hardware costs under $20k: two ViperX follower arms mirror two smaller WidowX leader arms backdriven by the operator (joint-space mapping); four webcams; teleop and recording at 50 Hz [§III, Fig. 1].
2. Actions are target joint positions for both arms taken from the leader arms (not the followers) [§IV].
3. ACT (Action Chunking with Transformers): predict the next k actions; trained as a conditional VAE whose encoder is discarded at test time [§IV-A, §IV-B]. Success rose from 1% at k = 1 to 44% at k = 100 in the chunk-size ablation [§VI].
4. Temporal ensembling: query every step and average overlapping chunk predictions with weights wᵢ = exp(−m·i) [§IV-A, Algorithm 2].
5. 80 to 90% success on fine tasks (e.g. slotting a battery, opening a condiment cup) from 10 minutes, or 50 demonstrations [Abstract, §I].

## 14. Brohan et al. 2023: RT-2

- Title: RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control
- Authors (arXiv order): Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, ... (54 listed). The CoRL proceedings page lists the authors in a different order starting with Brianna Zitkovich, so the PMLR citation is "Zitkovich et al."
- Venue as shown: "Proceedings of The 7th Conference on Robot Learning, PMLR 229:2165-2183, 2023" (https://proceedings.mlr.press/v229/zitkovich23a.html, opened 2026-11-02).
- arXiv: 2307.15818 (v1, 28 Jul 2023)
- Read it when: background on VLAs; not relevant to low-level locomotion control.

Facts:
1. Robot actions (6-DoF end-effector delta position and rotation, gripper extension, terminate flag) are discretized into 256 uniform bins and written as 8 integer tokens, so a vision-language model can output them as text [§3].
2. Co-fine-tuning on robot data mixed with the original web data generalizes better than fine-tuning on robot data alone [§3].
3. Instantiations: PaLI-X 5B and 55B, PaLM-E 12B. Run from a multi-TPU cloud service: 55B at 1 to 3 Hz, 5B at about 5 Hz [§3, §4].
4. About 6,000 real evaluation trials; best model more than 3× the average success of the next baseline on emergent-skill evaluations [§4].
5. Limitation: it gains semantic generalization but no new motions; physical skills stay limited to those in the robot data [§5].

## 15. Kim et al. 2024: OpenVLA

- Title: OpenVLA: An Open-Source Vision-Language-Action Model
- Authors: Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, ... (17 on the PMLR page; 18 on arXiv)
- Venue as shown: "Proceedings of The 8th Conference on Robot Learning, PMLR 270:2679-2713, 2025" (https://proceedings.mlr.press/v270/kim25c.html, opened 2026-11-02). CoRL 2024 proceedings, published 2025.
- arXiv: 2406.09246 (v3; v1 13 Jun 2024)
- Read it when: you want an open VLA to fine-tune.

Facts:
1. 7B parameters: Prismatic VLM with fused DINOv2 + SigLIP vision encoders and a Llama 2 7B backbone [§3].
2. Trained on 970k robot episodes from Open X-Embodiment; beats RT-2-X (55B) by 16.5% absolute success across 29 tasks with 7× fewer parameters [Abstract].
3. Each action dimension is discretized into 256 bins spanning the 1st to 99th percentile of the training actions (quantiles, not min-max, to ignore outliers) [§3].
4. Training: 64 A100s for 14 days. Inference: about 6 Hz on one RTX 4090 in bfloat16; supports LoRA fine-tuning and quantization [§3.5, §4, §5].
5. Limitation: single image input only, no proprioception or observation history [§6].

## 16. Black et al. 2024: π0

- Title: π0: A Vision-Language-Action Flow Model for General Robot Control
- Authors: Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, ... (24) (Physical Intelligence)
- Venue as shown: arXiv comment "Published in RSS 2025". The preprint is dated 31 Oct 2024; the proceedings page was not opened.
- arXiv: 2410.24164 (v4)
- Read it when: action-chunked VLA control at high frequency.

Facts:
1. PaliGemma (3B) VLM plus a 300M "action expert" that outputs continuous actions by flow matching (a diffusion variant that learns a velocity field from noise to actions), 3.3B total [§IV].
2. Predicts action chunks of H = 50; integrates the flow with 10 steps; supports control up to 50 Hz on dexterous tasks [§I, §IV].
3. Pre-trained on over 10,000 hours of robot data from 7 robot configurations and 68 tasks, plus Open X-Embodiment; then post-trained on curated task data [§I, §V].

## 17. Radosavovic et al. 2024: Real-world humanoid locomotion with RL

- Title: Real-world humanoid locomotion with reinforcement learning (arXiv title capitalizes it as "Real-World Humanoid Locomotion with Reinforcement Learning")
- Authors: Ilija Radosavovic, Tete Xiao, Bike Zhang, Trevor Darrell, Jitendra Malik, Koushil Sreenath (6)
- Venue as shown: published PDF header "SCIENCE ROBOTICS | RESEARCH ARTICLE ... Copyright © 2024" (https://hybrid-robotics.berkeley.edu/publications/ScienceRobotics2024_Learning_Humanoid_Locomotion.pdf, opened 2026-11-02). DOI 10.1126/scirobotics.adi9579 from a search result, not from an opened publisher page.
- arXiv: 2303.03381 (v2; v1 6 Mar 2023). Locations below refer to arXiv v2.
- Read it when: the closest published analog to a humanoid walking project; teacher-student with PPO in a GPU simulator.

Facts:
1. Policy is a causal transformer over the last 16 observation-action pairs (4 blocks, 1.4M parameters); teacher is an MLP [512, 512, 256, 128] on full state [Methods: "Model architecture"].
2. Training: teacher state policy with PPO, then student trained with L = L_RL + λ·KL(π_student ‖ π_teacher), λ annealed to 0 by mid-training so the student can surpass the teacher. Everything on-policy, no offline dataset [Methods: "Joint optimization with reinforcement learning"]. Ablation: the joint objective beats RL alone or imitation alone; the transformer beats other architectures; longer context helps [Fig. 8].
3. Action: PD setpoints for 16 joints plus predicted PD gains for 8 leg joints; toe motors held at default with fixed gains. Hardware: policy at 50 Hz, joint PD at 1 kHz [Methods: "Joint optimization with reinforcement learning"; "Sim-to-real transfer"].
4. Simulation: Isaac Gym across four A100s and thousands of environments; Digit's closed kinematic chains approximated with stiff "virtual springs". DR over robot dynamics, control parameters, environment physics, observation noise and delay; terrains are smooth planes, rough planes, smooth slopes. Policies are screened in Agility's high-fidelity simulator before hardware (no weight changes) [Methods: "Simulation", "Domain randomization", "Sim-to-real transfer"].
5. Limitation: gait is not perfectly symmetric; large steps can trap the feet [Discussion: "Limitations"].

## 18. He et al. 2025: ASAP (delta action model)

- Title: ASAP: Aligning Simulation and Real-World Physics for Learning Agile Humanoid Whole-Body Skills
- Authors: Tairan He, Jiawei Gao, Wenli Xiao, Yuanhang Zhang, Zi Wang, ... (18)
- Venue as shown: arXiv comment "RSS 2025"; proceedings page not opened.
- arXiv: 2502.01143 (v3; v1 3 Feb 2025)
- Read it when: your G1 policy works in sim but fails on hardware in a way DR does not fix; the paper uses the same robot.

Facts:
1. Four stages [§II-III, Fig. 2]: (a) pre-train phase-conditioned motion-tracking policies in IsaacGym on human video motions retargeted to the Unitree G1; (b) roll them out on the real robot and record trajectories; (c) train a delta action model Δaₜ = π_Δ(sₜ, aₜ) with PPO so that replaying real actions in sim, s_{t+1} = f_sim(sₜ, aₜʳ + Δaₜ), reproduces the real states; (d) freeze π_Δ inside the simulator and fine-tune the policy, then deploy without π_Δ.
2. Training details for the tracking policy: asymmetric actor-critic (critic gets privileged global body positions; actor gets proprioception plus time phase), a termination curriculum that tightens tracking tolerance from 1.5 m, and reference state initialization [§II-B].
3. Evaluated IsaacGym → IsaacSim, IsaacGym → Genesis (as stand-ins for reality) and IsaacGym → real G1; lower tracking error than SysID (system identification: fit simulator parameters to real data), DR and learned delta dynamics baselines [§IV, tables in §IV-A, §IV-B].
4. Real-world constraint: motor overheating broke two G1 robots during data collection, so the real delta model covers only the 4 ankle DoF (the ankle is a parallel linkage with a large sim-to-real gap) instead of all 23 DoF [§IV-C].
5. Limitations: hardware stress, and dependence on motion capture to record real trajectories [§VIII].

---

## Recipe: what a sim-to-real locomotion pipeline usually contains

Each line: component, then the papers above that support it. Items marked [inference] are my synthesis across papers.

1. GPU-parallel on-policy RL (PPO or TRPO) with thousands of simulated robots. Radosavovic (PPO, Isaac Gym, 4 A100s), Miki (PPO), Lee and Hwangbo (TRPO, CPU-era), plus `legged-gym.md` in this folder for the massively parallel PPO recipe. mjlab's G1 task uses rsl_rl PPO (see `../training/simulators-and-tooling.md`).
2. Joint position targets tracked by a PD controller, not raw torques. Hwangbo (position targets, kp = 50, kd = 0.1; trains faster than torque), RMA (Kp = 55, Kd = 0.8), Radosavovic (PD setpoints plus learned gains), Peng 2018 DR (relative joint targets), DeepMimic and AMP (PD targets). Lee and Miki add a periodic foot trajectory and inverse kinematics on top.
3. Policy at 30 to 200 Hz, PD loop several times faster. Hwangbo 200 Hz; RMA 100 Hz; Lee 50 Hz policy / 400 Hz PD; Miki 50 Hz; Radosavovic 50 Hz / 1 kHz; OpenAI hand 12 Hz / 1 kHz.
4. Proprioceptive observations with short history (joint positions and velocities, base angular velocity, gravity direction in the body frame, previous action, velocity command). Hwangbo (gravity vector, joint history at −0.01 s and −0.02 s), RMA (50-step history), Lee (TCN over up to 2 s), Radosavovic (16-step context).
5. Dynamics randomization of mass, center of mass, friction, PD gains or motor strength, and latency. Ranges: Peng 2018 [Table I], OpenAI 2018 [Table 1], RMA [Table I], Hwangbo (inertia noise). Ablations show each class matters: Peng [Table III], OpenAI [Table 4].
6. Observation noise, and delays on observations or actions. Peng (no obs noise: 0.25 vs 0.89 real success), OpenAI (correlated plus per-step noise; action delay), Radosavovic (noise and delay), Miki (structured height-map noise).
7. Explicit actuator modeling when the motor is not ideal. Hwangbo (learned actuator network from real data); ASAP (learned delta actions on the G1 ankle); for mjlab's G1, armature and gains come from motor specs (see tooling file) [inference: this is the simulator-side counterpart].
8. Random pushes and external forces. Miki (external force and torque), OpenAI (random object forces), Lee (external disturbance in teacher's privileged input).
9. A critic that sees privileged state (asymmetric actor-critic). Pinto (origin), OpenAI 2018 [Table 2], Peng 2018 (value gets μ), ASAP.
10. Teacher-student distillation when the deployed policy lacks key information. Lee (action plus latent loss, DAgger), Miki (behavior cloning plus reconstruction), RMA (regress the latent from history), Radosavovic (KL to teacher plus RL loss, annealed).
11. Memory to infer hidden dynamics online. Peng (LSTM 0.89 vs feedforward 0.67), OpenAI 2018 (LSTM median 13 vs 3.5), Rubik's (meta-learning under ADR), RMA, Radosavovic, Lee.
12. Curriculum on terrain, disturbances, commands or randomization width. Lee (particle-filter terrain curriculum), Hwangbo (cost curriculum), OpenAI 2019 (ADR widens randomization automatically), `legged-gym.md` (game-inspired terrain curriculum).
13. Reward: tracking terms plus regularizers; motion priors for natural style. DeepMimic (imitation reward, RSI, early termination), AMP (discriminator style reward), ASAP (motion tracking). Hwangbo notes too-high regularization yields standing still [Materials and Methods].
14. Correct handling of time-limit resets (bootstrap the value on time-outs, do not treat them as failures). `legged-gym.md` [§2.2.2]; Gymnasium's terminated vs truncated API and rsl_rl's time-out bootstrapping are covered in the tooling file.
15. Sim-to-sim validation before hardware. Radosavovic (Agility simulator screen), ASAP (IsaacGym to IsaacSim and Genesis). [inference] For an mjlab-trained policy, replaying in a second simulator or plain CPU MuJoCo is the cheap analog.
16. Close the loop with real data when DR is not enough. Hwangbo (actuator net), ASAP (delta action model), Rubik's ADR starts from calibrated values [§5.1].

Not part of a locomotion pipeline, but in the canon for imitation and manipulation: Diffusion Policy, ACT, RT-2, OpenVLA, π0 (behavior cloning from demonstrations, action chunking, VLAs). Action chunking (ACT, Diffusion Policy, π0) is the imitation-learning answer to compounding error, the way DR is the RL answer to the reality gap [inference].

## Unverified or inconsistent items

- Tobin 2017 at IROS: only "Submitted to IROS 2017" was seen; the IEEE page did not load.
- OpenAI 2018 and 2019: arXiv preprints; later journal versions not checked.
- π0 and ASAP RSS 2025: from arXiv comments only.
- Radosavovic DOI: from a search result; the opened PDF confirms Science Robotics 2024 but does not show the DOI in the extracted first page.
- RMA: sim timestep (0.025 s) conflicts with 100 Hz control in the same paragraph.
- RMA Table I prints center-of-mass range in cm as ±0.15, which is very small; the unit may be a typo for m or the range may be intended.
- Hwangbo, Lee, Miki, Radosavovic give full randomization tables only in their supplements, which were not read.
