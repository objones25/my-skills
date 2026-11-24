# Humanoid motion tracking (reference-motion RL)

Recipe facts verified in the BeyondMimic paper (arXiv 2508.08241, G1 whole-body tracking); other papers named here come from the findings report and were not re-opened (unverified). Reward and observation basics: `reward-and-observation-design.md`; DR and deployment: `sim-to-real.md`; SuperMimic's world-model alternative: `papers/supermimic.md`.

## BeyondMimic recipe [2508.08241]

- Observation: anchor pose error as the first two columns of the rotation error matrix (Rot6D), no history; the paper says continuous rotation forms such as Rot6D gave markedly better sim-to-real than quaternion or axis-angle [lines 553, 789-790].
- Termination when the anchor height error exceeds 0.25 m or the anchor orientation error exceeds 0.8 rad [line 1874].
- Actuators: PD gains from a natural frequency of 10 Hz with damping ratio 2 (overdamped); 10 Hz was best among values tried and was compared with ASAP's gains [lines 1829-1840, 1915-1921].
- PPO (rsl_rl style) [Table S5, lines 2259-2273 region]: actor and critic MLP [512, 256, 128], ELU, 24 steps per env, 30,000 max iterations, lr 1e-3, clip 0.2, entropy 0.005, value coef 1.0, γ 0.99, λ 0.95, desired KL 0.01, 5 epochs, 4 minibatches. Note 24 steps is the community default, not a floor (`on-policy-at-scale.md`).
- Domain randomization kept deliberately small: ground friction and restitution, default joint positions (joint-offset calibration error), torso COM position, plus random velocity pushes [lines 584-590].
- Adaptive sampling of the start phase: the reference is cut into one-second bins; each bin's failure rate is an EMA (0.999/0.001), offset by a uniform floor 0.1/S so low failure everywhere reverts to uniform, then convolved with a decaying kernel (ρ = 0.8, look-back 0 to 2) and sampled multinomially; resets use small pose and velocity perturbations [lines 1875-1896]. Ablation: without it three of four motions failed on hard segments even after 30k iterations; on an easy motion it halved iterations (2k vs 4k) [lines 818-823 region].

## History and delay on hardware [2508.08241 lines 795, 810]

- Adding observation history degraded performance "in our setting" (the paper cites other work where history helps), so do not assume history helps motion tracking.
- Measured latency sensitivity (3 real trials): a 2 ms delay raises velocity error, 5 ms causes one failure, 10 ms causes failures in two of three trials. Randomizing delay in training could mitigate this but "makes learning more difficult". Contrast the locomotion-oriented advice in `sim-to-real.md`.

## Comparison to read next (not verified here)

Adaptive or failure-weighted start sampling recurs across motion-tracking papers (SONIC, GMT, EGM, KungfuBot, SuperMimic's return-based sampler, 50% uniform / 40% softmax / 10% prior per `papers/supermimic.md`); treat them as one family. Two DR camps exist on G1 (minimal as above vs broad: PD gain, mass, motor strength, 0 to 40 ms delay) and no paper compares them head to head, per the findings report (unverified).
