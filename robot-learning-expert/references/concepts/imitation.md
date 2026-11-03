# Learning from demonstrations

When an expert exists, copying it is often far cheaper than discovering behavior by reward. The core tension: supervised imitation trains on the expert's state distribution but is tested on the learner's, and small errors push the learner to states the expert never showed it.

Sources: AJKS ch. 13 and BBC ch. 6 to 7 for theory; paper facts are verified in `papers/canon-algorithms.md` (DAgger, GAIL) and `papers/canon-robotics.md` (DeepMimic, AMP, Diffusion Policy, ACT, VLAs).

## Notation

| Symbol | Meaning |
|---|---|
| π_E | expert policy (queryable oracle or only a fixed dataset) |
| D_E | demonstration dataset of (s, a) pairs from π_E |
| ε | per-step error of the learned policy on the expert's own state distribution |
| T or H | horizon |

## Settings (AJKS §13.1, Notes 0's "is there an expert?" question)

| Expert access | Method family |
|---|---|
| Static dataset only | behavior cloning (BC), offline RL, inverse RL |
| Dataset plus environment interaction, no reward | GAIL / AMP (adversarial imitation), max-entropy IRL (AJKS §13.4) |
| Expert can be queried on any state | interactive imitation: DAgger (AJKS §13.5) |
| Dataset plus reward | hybrid (AJKS §13.3): demos to start, RL to improve (e.g. RLPD, DAPG-style) |

## Behavior cloning and compounding error

BC is supervised learning: minimize E_{(s,a)~D_E}[ℓ(π(s), a)] for a supervised loss ℓ (squared error, negative log-likelihood) (AJKS §13.2). The classic analysis (Ross and Bagnell; Ross et al. 2011): if π errs with probability ε per step on the expert's distribution, a mistake can put it in unfamiliar states where it keeps erring, so the total cost can grow like εT², not εT. DAgger fixes the distribution: roll out the current learner (mixed with the expert early on), ask the expert to label the states visited, aggregate into the dataset, retrain. Its guarantee is linear in T (exact statement and conditions in `papers/canon-algorithms.md`).

Robot practice uses noise injection during data collection (DART-style) and action chunking (predict k future actions, as in ACT) to reduce compounding error without a queryable expert.

## Imitation in robot learning today `[practice]`

- Motion imitation for legged and humanoid control: DeepMimic tracks reference clips with a tracking reward inside RL; AMP replaces the hand-built tracking reward with a discriminator that scores style. mjlab ships a G1 motion tracking task (`mjlab/tasks/tracking/`), and SuperMimic scales whole-body tracking with a differentiable world model (`papers/supermimic.md`).
- Visuomotor policies from teleoperation: Diffusion Policy (denoising diffusion over action sequences), ACT (transformer predicting action chunks, ALOHA hardware).
- Vision-language-action models: RT-2, OpenVLA, π0 fine-tune large pretrained models on robot demonstrations; BC at scale.
- Teacher-student (privileged learning) is imitation inside sim-to-real: a teacher trained with RL on privileged state is distilled into a student that sees only onboard sensors, often with DAgger-style relabeling (Lee et al. 2020; Kumar et al. 2021, RMA; `training/sim-to-real.md`).

## Choosing

"Is there an expert, queryable or static, optimal or just good?" is a protocol question that decides the algorithm (Notes 0): demonstrations alone point to BC or offline methods, a queryable expert to DAgger (SKILL.md decision guide).
