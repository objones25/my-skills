# Decoupling Exploration and Policy Optimization: Uncertainty Guided Tree Search for Hard Exploration (Mhammedi, Cohan)

Source: arXiv:2603.22273v4 (13 May 2026), 33 pages, Google Research. Two authors: Zakaria Mhammedi and James Cohan. The algorithm is called GowU (Go-With-Uncertainty); "UGTS" is only the title phrase. Tags like [UGTS §4.2], [UGTS Alg. 1, l. 20], [UGTS App. K] point to this version. [inference] marks statements that are mine.

## 0. Summary

- Exploration is run as a search over simulator states, with no policy optimization during exploration: a population of particles does short random rollouts, dead particles are reset to a "winner" (highest cumulative reward, ties broken by epistemic uncertainty from RND), and groups periodically collapse onto their winner [UGTS §4].
- Phase I (GowU) outputs trajectories of simulator checkpoints, not a policy. Phase II distills the best trajectory into a PPO policy with a backward-learning curriculum (Salimans and Chen 2018) [UGTS §3, §5].
- Results: much faster exploration than Go-Explore on Montezuma's Revenge, Pitfall!, Venture, and distilled policies that beat all listed baselines by a wide margin; MuJoCo Adroit (door, hammer, relocate) and AntMaze solved from pixels with sparse reward and no demonstrations [UGTS §5].
- No theorems about GowU itself. The only analysis is a toy tree example showing GWTW needs O(D log(D/δ)) node evaluations where DFS/BFS need Θ(2^D) [UGTS App. K].

## 1. Setting [UGTS §3]

- Stochastic episodic MDP M = (X, A, P, R): X states, A actions (discrete or continuous), P : X × A → Δ(X) transitions, R : X × A → ℝ a deterministic reward [UGTS §3].
- Alive / Dead partition of states. Dead is a user-defined failure notion stricter than episode termination (e.g., losing one life in Atari, the ant flipping in AntMaze). Doom states ⊂ Alive are states from which failure is inevitable under every policy. The method assumes progress stays possible from any non-Doom alive state [UGTS §3].
- Uncertainty oracle U: a scalar score per state that decreases as the state is visited more, updated online. Any estimator works; experiments use RND prediction error (RND = Random Network Distillation: error of a trained predictor network against a fixed random target network) [UGTS §3, App. A.3].
- Reset oracle: overwrite one particle's simulator state with another's (cloning). Needs full checkpoints, including the frame stack [UGTS §3, App. C].
- Resets are a training-time tool only. The final policy does not use the simulator [UGTS §2].

Relation to the PoR protocol [inference]: GowU only restores states it has visited (checkpoints in its lineage tree), so it operates inside the local simulator protocol of Power of Resets. It is non-episodic (no layer index h) and adds the Dead/Alive signal, which PoR does not have.

## 2. Algorithm

### 2.1 Data structure: state-lineage tree [UGTS §4.1, App. A.1]

Node v = (s_v, R_v, u_v): s_v a simulator checkpoint, R_v cumulative extrinsic reward up to s_v, u_v a pointer to the parent node. Nodes are added only at the end of a rollout segment, so consecutive checkpoints are "inner steps" apart [UGTS §4.2, §5.2]. Nodes no longer occupied by a particle and not referenced by any descendant are pruned, so memory scales with active lineages [UGTS §4.1]. A demonstration for Phase II is the path from the highest-reward leaf to the root via parent pointers [UGTS App. A.1].

### 2.2 Algorithm 1, EvolveGroup [UGTS §4.2, Alg. 1]

```
Algorithm 1  EvolveGroup
input:  population P = {p_1, …, p_N}, lineage tree T, uncertainty U
params: outer steps, inner steps, rollback ranges: [K_min, K_max], [T_min, T_max], [k_min, k_max]
 1: sample outer steps K ~ Unif(K_min, K_max)
 2: for k = 1, …, K do
 3:   for each particle p_i ∈ P do
 4:     sample inner steps T_i ~ Unif(T_min, T_max)
 5:     /* rollout particle and update uncertainty */
 6:     for t = 1, …, T_i do
 7:       U ← p_i.Step(1, U)                                   // execute one step
 8:       if p_i is Dead or p_i.R increased then break          // halting on reward does not kill the particle
 9:     end for
10:     if p_i is Alive then
11:       T.AddChild(p_i)                                       // expand tree for surviving particles
12:     end if
13:   end for
14:   if all p_i ∈ P are Dead then
15:     /* failure recovery via rollback */
16:     v_{i,anc} ← p_i.GetAncestor(k_min, k_max), ∀i          // k_i-th ancestor with k_i ~ Unif(k_min, k_max)
17:     P ← {Reset(p_i, v_{i,anc}) | ∀i}                        // restore all particles to ancestor checkpoints
18:   else
19:     /* uncertainty-aware redistribution */
20:     p_winner ← lex-argmax_{p ∈ P, p.alive} (p.R, U(p))       // best alive: max reward, ties by uncertainty
21:     for each p_i ∈ P do
22:       if p_i is Dead then
23:         Reset(p_i, p_winner)                                // prune and respawn; p_i becomes alive
24:       end if
25:     end for
26:   end if
27: end for
28: /* group consolidation (lexicographic ranking) */
29: p_winner ← lex-argmax_{p ∈ P} (p.R, U(p))
30: P ← {Reset(p_j, p_winner) | ∀j}                             // reset all particles to winner's checkpoint
31: return P, T, U, p_winner
```

lex-argmax (a, b): maximize a first, break ties by b [UGTS Alg. 1, l. 20 comment]. Non-winner alive particles keep their own trajectories during the K outer steps to preserve diversity; only dead ones are respawned [UGTS §4.2]. Stopping a rollout as soon as reward increases "secures" the reward before a possible death [UGTS §4.2].

### 2.3 Algorithm 2, GowU [UGTS §4.3, Alg. 2]

```
Algorithm 2  GowU: Go-With-Uncertainty
input: env E, iterations N_iter, number of groups M, particles per group N
 1: initialize lineage tree T with root node s_0 (initial state)
 2: initialize groups {G_1, …, G_M}, each with N particles, all starting at s_0
 3: initialize shared uncertainty estimator U
 4: initialize p_winner at state s_0 with cumulative reward 0
 5: for i = 1, …, N_iter do
 6:   for each group G_m in parallel do
 7:     /* global synchronization */
 8:     p_m ← lex-argmax_{p ∈ G_m} (p.R, U(p))
 9:     if p_m.R < p_winner.R then
10:       reset G_m to the state of p_winner                   // all particles to global winner's checkpoint
11:     end if
12:     /* evolve group */
13:     G_m, T, U, p_m ← EvolveGroup(G_m, T, U)                 // Algorithm 1
14:   end for
15:   /* update global winner */
16:   p_winner ← lex-argmax_{p ∈ {p_1, …, p_M, p_winner}} (p.R, U(p))
17: end for
```

Groups share one uncertainty estimator, so one group's visits lower novelty for the others [UGTS §4.3]. A group whose best reward ties the global winner keeps searching independently [UGTS §4.3].

### 2.4 Action selection and RND details [UGTS §4.4, App. A]

- Each particle commits to one random action for the whole rollout segment. Discrete: that action with probability 1 − ε (ε = 0.2), uniform random otherwise. Continuous: the committed action every step [UGTS §4.4, App. A.2].
- RND: target and predictor use the standard Atari conv torso plus an MLP head of two 1024-unit layers with ReLU; orthogonal init; inputs normalized by running mean/std and clipped to [−5, 5]; predictor trained by MSE [UGTS App. A.3].
- Shared replay buffer of 512,000 (observation, action) entries; batch 128; Adam, lr 3×10⁻⁴; coordinator periodically pulls predictor weights [UGTS App. A.3].
- Architecture: central coordinator (population, lineage tree, winner logic), stateless rollout workers (restore checkpoint, roll out, report endpoint), asynchronous RND learner [UGTS App. A.1 to A.3].

### 2.5 Hyperparameters [UGTS App. A.4, Table 5]

| Parameter | Montezuma | Pitfall | Venture | MuJoCo |
|---|---|---|---|---|
| Groups M | 4 | 4 | 4 | 16 |
| Particles per group N | 32 | 32 | 32 | 8 |
| Outer steps [K_min, K_max] | [10, 20] | [10, 20] | [6, 16] | [20, 40] |
| Inner steps [T_min, T_max] | [5, 15] | [10, 20] | [10, 20] | [3, 4] |
| Rollback [k_min, k_max] | [1, 3] | [1, 4] | [1, 12] | [1, 20] |

K, T, and rollback depth are resampled every iteration by default; fixing one draw per run gives high variance across seeds [UGTS App. G].

### 2.6 Phase II: backward learning [UGTS §5, App. B]

- Take the highest-reward trajectory from each GowU run. Start PPO episodes from checkpoints near the end of the demonstration and move the start backward as the agent succeeds [UGTS §5, App. B.1].
- Start index sampled uniformly in [t_curr − Δ, t_curr]. Success: R_agent ≥ R_demo(t_start) − ε_tol. If the success rate S over the last N_update = 32 rollouts is ≥ S_req = 0.2, move back by a random step in [α_dec Δ, β_dec Δ]; otherwise move forward by a step in [α_inc Δ, β_inc Δ]. At t_curr = 0 require S ≥ 0.95 [UGTS App. B.2, Table 8].
- Atari: up to 10 segments split at reward events, Δ = 25, (α_dec, β_dec) = (0.25, 0.75), (α_inc, β_inc) = (0.5, 1.0); Adroit: one segment, Δ = 3, multipliers 1/3; AntMaze: one segment, Δ = 25 [UGTS App. B.3].
- PPO with separate IMPALA ResNet encoders for policy and value; lr 1×10⁻⁴, γ = 0.999, GAE λ = 0.95, clip 0.1 [UGTS App. B.4, Table 7].
- Phase I's dead-state rules and intermediate rewards are not used in Phase II [UGTS §5].

## 3. Experiments

Benchmarks [UGTS §5]:
- Atari (ALE): Montezuma's Revenge, Pitfall!, Venture. 84×84×4 grayscale frame stacks, sticky actions p = 0.25, action repeat 4. Pitfall!'s timer is disabled in Phase I by writing RAM and re-enabled for Phase II and evaluation [UGTS §5, App. D, App. E.1].
- MuJoCo: antmaze-large-diverse-v0 and Adroit door-v0, hammer-v0, relocate-v0 (24-DoF ShadowHand) with the sparse_reward flag. Dynamics are deterministic but goal/object placements are randomized across seeds. Final policies see only 120×120 grayscale images [UGTS §5, App. E].
- Phase I exploration signals: Atari dead = life loss (Pitfall!: also any negative reward; Montezuma: an extra rule to block a reward bug). AntMaze dead = flipped (torso height z < 0.2 or up-vector z-component < 0). Adroit: +1 on first contact with the object, dead if contact is lost. These use privileged state [UGTS §5, App. F, App. H, App. I].

Baselines: Go-Explore without domain knowledge, RND, MEME, BYOL-Hindsight on Atari, with numbers taken from those papers; Go-Explore's exploration used a deterministic environment. No direct MuJoCo baselines exist for this setting [UGTS §5, Table 1].

Atari results (mean ± std; exploration over 100 seeds, Phase II over 10 runs × 5 seeds) [UGTS Table 1, Table 3]:

| Game | GowU exploration | GowU distilled policy | Go-Explore | RND | MEME | BYOL-Hindsight (approx.) |
|---|---|---|---|---|---|---|
| Montezuma | 98,249 ± 18,102 | 196,312 ± 49,670 | 43,791 | 8,152 | 9,429 | ~14,517 |
| Pitfall | 54,440 ± 14,817 | 101,972 ± 2,226 | 6,945 | −3 | 7,821 | ~16,211 |
| Venture | 5,012 ± 2,678 | 6,436 ± 1,504 | 2,281 | 1,859 | 2,583 | ~2,328 |

Exploration budget: 400M frames per seed on Montezuma, 600M on Pitfall! and Venture [UGTS §5.1]. Best Montezuma Phase II run: 263,111; worst runs still exceed every baseline [UGTS §5.2, Table 3]. Some top Montezuma policies exploit the first-room key respawn, which the authors call a game feature, not a bug [UGTS App. J].

MuJoCo [UGTS Table 2, Table 4]:

| Task | Frames to first success, Phase I (mean / median, 100 seeds) | Phase II success rate (mean ± std) |
|---|---|---|
| Hammer | 26,797 / 24,142 | 1.000 ± 0.001 |
| Door | 272,688 / 224,338 | 0.981 ± 0.006 |
| Relocate | 275,372 / 69,546 | 0.952 ± 0.017 |
| AntMaze | 10,161,272 / 8,517,186 | 0.961 ± 0.019 |

Phase II costs far more than Phase I: Atari 15B to 20B frames; MuJoCo medians from about 9.9M (hammer) to 221.7M (relocate) frames [UGTS §5.2, App. B.6, Table 9].

Compute: Montezuma Phase I with 128 CPU rollout workers, 1 coordinator CPU, 1 TPU for RND: 10.71 ± 3.35 hours and about 2.28M RND SGD steps for 400M frames, comparable to the ~3M updates of BYOL-Explore/BYOL-Hindsight [UGTS App. A.5, Table 6].

Ablations on Montezuma (8 seeds per variant) [UGTS §5.3, App. G]:
- Winner chosen uniformly at random instead of by uncertainty: exploration fails (does not finish level 1).
- No group consolidation: worse, still better than no-uncertainty.
- Groups M ∈ {1, 2, 4, 8} at 128 total particles: 2 and 4 similar and best, 1 worst, 8 in between.
- Per-iteration hyperparameter randomization beats a fixed draw per run in robustness.

Stated limitation: RND can keep assigning high novelty to stochastic observations (the "noisy TV" problem); the authors suggest uncertainty measures that separate aleatoric from epistemic uncertainty, or temporal contrastive features [UGTS §5.4, App. H].

## 4. Relation to RVFS / Power of Resets

| Aspect | RVFS (PoR) | GowU (UGTS) |
|---|---|---|
| Uses resets to visited states | Yes [PoR §2.1] | Yes [UGTS §3] |
| Learns a value function during exploration | Yes, V̂_h per layer [PoR Alg. 3] | No; only an uncertainty model U [UGTS §2] |
| What decides where to revisit | Confidence-set test on value backups [PoR Alg. 3, l. 14] | Max cumulative reward, ties by RND uncertainty [UGTS Alg. 1, l. 20] |
| Stored states | Core sets C_h of (state, action) pairs [PoR §4.2] | Lineage tree of checkpoints [UGTS §4.1] |
| Exploration policy | Greedy backup policy of V̂ [PoR Eq. 15] | Random committed actions [UGTS §4.4] |
| Executable policy | Behavior cloning of the backup policy [PoR Alg. 4] | Backward-learning PPO on discovered demos [UGTS §5] |
| Guarantees | Polynomial sample complexity under pushforward coverability + realizability [PoR Thm 4.1] | None for GowU; toy GWTW example only [UGTS App. K] |
| Action space | Finite (enumerated in tests) [PoR Alg. 3, l. 12] | Discrete or continuous [UGTS App. H] |
| Experiments | None [PoR §5] | Atari and MuJoCo [UGTS §5] |

UGTS cites Power of Resets for the claim that local simulator access enables efficient learning where online RL is provably inefficient, and for the importance of using resets well in simulation [UGTS §2, §5.4]. It does not build on RVFS's core sets or confidence sets, and it explicitly produces no value function [UGTS §2]. [inference] The shared idea is "reset to states worth revisiting"; PoR picks them by value-function disagreement, GowU by reward plus novelty.

## 5. What a reset-based study can reuse [inference]

- Practical baseline: GowU + backward learning is a reset-based method with strong empirical results, so it is the natural "practical" comparison for an RVFS implementation on the same toy MDPs. On small discrete environments the RND oracle can be replaced by visit counts.
- Noisy-observation stress test: GowU's winner selection depends on U(s). With observation noise, RND novelty will stay high on noisy observations (the paper's own stated limitation [UGTS §5.4]), so GowU may chase noise. That is a direct, testable contrast with RVFS, whose test uses value disagreement rather than novelty.
- Reset engineering: App. C is a checklist of what a checkpoint must contain (emulator or physics state, RNG seed for randomized goals, task flags, frame stack) [UGTS App. C]. For RVFS's P̂ you also need fresh transition randomness after restore, which App. C does not discuss.
- Lineage tree with parent pointers and pruning is a reasonable storage layer for RVFS's snapshots too.
- Distillation alternative: backward learning from checkpoints could replace RVFS.bc's behavior cloning when the RVFS backup policy is too expensive to query for many BC trajectories.
- Toy testbed: the GWTW tree of App. K (spine of length D, trap subtrees with branching 6 where 4 children are dead ends) has a known exponential separation between GWTW and DFS/BFS [UGTS App. K]. A layered-MDP version of it is a cheap sanity check for any reset-based explorer, including RVFS.
- Environments: AntMaze and Adroit with sparse rewards are reset-friendly continuous-control testbeds, but RVFS needs a finite action set, so they would need action discretization.

## 6. Ambiguities and notes

1. The paper's name for the method is GowU; "Uncertainty Guided Tree Search" appears only in the title.
2. Table 1's caption says "at most 600M frames per seed"; the text gives 400M for Montezuma and 600M for Pitfall! and Venture [UGTS Table 1, §5.1]. Consistent, just stated two ways.
3. BYOL-Hindsight numbers are read off a figure in the original paper (marked approximate) and Go-Explore curves in Fig. 5 are approximated from its paper's figure [UGTS Table 1, Fig. 5].
4. Phase I on AntMaze uses only the top-down view; Phase II adds four egocentric frames [UGTS App. E.3].
5. The "order of magnitude more efficient than intrinsic motivation baselines" claim [UGTS abstract, §1] is supported in the main text by the Go-Explore curve comparison and final-score table; there is no frames-to-score comparison against RND/MEME/BYOL in the paper.
