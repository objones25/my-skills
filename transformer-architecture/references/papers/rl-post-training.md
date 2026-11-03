# RL post-training objectives: GRPO, DAPO, Dr. GRPO, GSPO (plus DeepSeek-V3.2's stabilizers)

1. DeepSeekMath (GRPO). Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y.K. Li, Y. Wu, Daya Guo (DeepSeek-AI, Tsinghua, Peking University). arXiv:2402.03300 (v3, 27 Apr 2024; v1 5 Feb 2024). Checked: §4 (RL) and §5.2 from the arXiv HTML.
2. DAPO: An Open-Source LLM Reinforcement Learning System at Scale. Qiying Yu, Zheng Zhang, Ruofei Zhu, ... (ByteDance Seed, Tsinghua AIR, HKU, SIA-Lab). arXiv:2503.14476 (v2, 20 May 2025; v1 18 Mar 2025). Checked: full text §1-4.
3. Understanding R1-Zero-Like Training: A Critical Perspective (Dr. GRPO). Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, Min Lin (Sea AI Lab, NUS, SMU). arXiv:2503.20783 (v2, 6 Oct 2025; v1 26 Mar 2025). Checked: §1, §3 (RL analysis); §2 (base models) skimmed; appendices (derivation, hyperparameters) not read.
4. Group Sequence Policy Optimization (GSPO). Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, Jingren Zhou, Junyang Lin (Qwen Team, Alibaba). arXiv:2507.18071 (v2, 28 Jul 2025; v1 24 Jul 2025). Checked: full text.
5. DeepSeek-V3.2 §3.1 "Scaling GRPO" (arXiv:2512.02556, v1, 2 Dec 2025). Checked: §3.1 full text.

## The idea

All four are critic-free PPO variants for RL with verifiable (rule-based) rewards. They sample G responses per prompt, use the group's rewards as the baseline, and differ in four choices: how the advantage is normalized, how token losses are averaged, where the importance ratio and clip live (token or sequence), and whether there is a KL term.

- GRPO: advantage (r_i − mean)/std over the group; per-response mean over tokens, then mean over responses; token-level PPO clip; KL to a reference model added to the loss with the "k3" estimator [DSMath §4.1].
- DAPO: drop KL; asymmetric clip (ε_low 0.2, ε_high 0.28) to stop entropy collapse; drop prompts whose G responses are all right or all wrong and resample; average over all tokens in the group rather than per response; soft length penalty near the generation limit. Qwen2.5-32B base: AIME24 30 (naive GRPO) to 50 [DAPO Table 1].
- Dr. GRPO: GRPO's 1/|o_i| makes long wrong answers cheap and short right answers expensive, and dividing by the group std overweights too-easy and too-hard prompts. Remove both (and normalize by a constant like the generation budget); this recovers the PPO objective with a Monte Carlo return and mean baseline [Dr. GRPO §3.1-3.2].
- GSPO: a per-token importance ratio uses a single sample per next-token distribution and only adds noise; use one length-normalized sequence-level ratio s_i = (π_θ(y_i|x)/π_old(y_i|x))^{1/|y_i|}, clipped per response. It made MoE RL converge without "Routing Replay" [GSPO §3-5].

## Notation

| Symbol | Meaning |
|---|---|
| q (or x) | Prompt |
| o_i (or y_i) | i-th sampled response, i = 1..G |
| G | Group size (responses per prompt) |
| \|o_i\| | Length of response i in tokens |
| o_{i,t} | t-th token of response i |
| π_θ, π_old, π_ref | Current policy, policy that sampled the batch, frozen reference (usually the SFT model) |
| r_{i,t}(θ) or w_{i,t} | Token ratio π_θ(o_{i,t}\|q, o_{i,<t}) / π_old(o_{i,t}\|q, o_{i,<t}) |
| s_i(θ) | GSPO sequence ratio, exp((1/\|y_i\|) Σ_t log w_{i,t}) [GSPO Eq. 7] |
| R_i | Scalar reward of response i |
| Â_{i,t} | Advantage assigned to token t of response i (same for all t under outcome reward) |
| ε, ε_low, ε_high | Clip range(s) |
| β | KL coefficient |
| D_KL (k3) | Per-token estimator π_ref/π_θ − log(π_ref/π_θ) − 1 [DSMath Eq. 4] |
| L_max, L_cache | DAPO maximum length and the soft-penalty band below it |
| μ | Gradient steps per rollout batch |

## The objectives side by side

GRPO [DSMath Eq. 3, §4.1.2]:

J = E[ (1/G) Σ_i (1/|o_i|) Σ_t { min(r_{i,t} Â_{i,t}, clip(r_{i,t}, 1−ε, 1+ε) Â_{i,t}) − β D_KL[π_θ || π_ref] } ], with Â_{i,t} = (R_i − mean(R)) / std(R).

DAPO [DAPO Eq. 8-13]:

J = E[ (1/Σ_i |o_i|) Σ_i Σ_t min(r_{i,t} Â_{i,t}, clip(r_{i,t}, 1−ε_low, 1+ε_high) Â_{i,t}) ], subject to 0 < #correct < G; same Â as GRPO; β = 0; R = ±1 correctness plus R_length = 0 if |y| ≤ L_max − L_cache, ((L_max − L_cache) − |y|)/L_cache inside the band, −1 beyond L_max.

Dr. GRPO [§3.2]:

J = E[ (1/G) Σ_i Σ_t min(r_{i,t} Ã_i, clip(r_{i,t}, 1−ε, 1+ε) Ã_i) ] / C, with Ã_i = R_i − mean(R) and C a constant (e.g. the generation budget MAX_TOKENS) instead of the response or batch length; β = 0.

GSPO [Eq. 5-7]:

J = E[ (1/G) Σ_i min(s_i Â_i, clip(s_i, 1−ε, 1+ε) Â_i) ], Â_i = (r_i − mean)/std. GSPO-token (Eq. 13-14) allows per-token advantages with s_{i,t} = sg[s_i] · π_θ(y_{i,t})/sg[π_θ(y_{i,t})], numerically equal to s_i.

| Choice | GRPO | DAPO | Dr. GRPO | GSPO | DeepSeek-V3.2 |
|---|---|---|---|---|---|
| Advantage | (R − mean)/std | (R − mean)/std | R − mean | (R − mean)/std | R − mean [DSV3.2 §3.1] |
| Token averaging | per response (1/\|o_i\|) | over all group tokens (1/Σ\|o_i\|) | constant | per response, inside s_i | per response (1/\|o_i\|) |
| Ratio and clip | token | token, asymmetric | token | sequence (geometric mean) | token, plus masking |
| KL | β·k3 to π_ref | none | none (β = 0) | not used in the paper | k3 corrected by r_{i,t}; weak or none for math |
| Filtering | none | drop all-correct / all-wrong groups | none | none | mask negative-advantage sequences whose mean log(π_old/π_θ) > δ |

## Hyperparameters

| Setting | GRPO (DeepSeekMath) | DAPO | GSPO | Source |
|---|---|---|---|---|
| Policy LR | 1e-6 | 1e-6 constant, 20-step warmup | not given in the paper | DSMath §4.2; DAPO §4.1 |
| β (KL) | 0.04 | 0 | not stated | same |
| G | 64 | 16 | not given | same |
| Batch | 1024 | 512 prompts per rollout; mini-batch 512 → 16 updates per rollout | 4 mini-batches per rollout | DSMath §4.2; DAPO §4.1; GSPO §5.1 |
| Updates per rollout (μ) | 1 | 16 | 4 | same |
| Clip | not given in §4 | ε_low 0.2, ε_high 0.28 | 3e-4 / 4e-4 (GRPO baseline 0.2 / 0.27) | DAPO §4.1; GSPO §5.1 |
| Max length | 1024 | 16,384 + 4,096 penalty band = 20,480 | not given | DSMath §4.2; DAPO §4.1 |
| Model | DeepSeekMath-Instruct 7B, ~144K GSM8K/MATH questions | Qwen2.5-32B base, DAPO-Math-17K | cold-start SFT of Qwen3-30B-A3B-Base | same |

Dr. GRPO's hyperparameters are in its Appendix G (not read).

## Key results

- GRPO on DeepSeekMath-Instruct 7B: GSM8K 82.9 to 88.2, MATH 46.8 to 51.7, CMATH 84.6 to 88.8 [DSMath §1.1]. RL improved Maj@K but not Pass@K, which the authors read as sharpening the output distribution rather than adding capability [§5.2.2].
- DAPO ablation, AIME24 avg@32 on Qwen2.5-32B: naive GRPO 30; + overlong filtering 36; + clip-higher 38; + soft overlong punishment 41; + token-level loss 42; + dynamic sampling 50; DeepSeek-R1-Zero-Qwen-32B 47 [DAPO Table 1]. Reached 50 in half of R1-Zero-Qwen-32B's steps [§4.2].
- Dr. GRPO vs GRPO (Qwen2.5-1.5B, MATH questions): similar reward; GRPO keeps growing response length after reward plateaus and produces longer incorrect answers; Dr. GRPO does not [§3.2, Fig. 5; plot only]. Minimal recipe: Qwen2.5-Math-7B, MATH levels 3-5, 27 hours on 8×A100 [§1].
- Dr. GRPO reports that trl, OpenRLHF, verl, SimpleRL-Zero and Open-Reasoner-Zero PPO losses all normalize by response or batch length [Table 2].
- GSPO vs GRPO on Qwen3-30B-A3B: better training reward and AIME24 / LiveCodeBench / CodeForces curves at equal compute [Fig. 1; plots only, no table]. GSPO clips about two orders of magnitude more tokens yet learns faster [§5.2]. About 10% of activated experts change after one gradient update on the 48-layer Qwen3-30B-A3B, which destabilizes token ratios; GRPO needed Routing Replay to converge, GSPO did not [§5.3].
- DeepSeek-V3.2 stabilizers: an importance-weighted k3 KL estimate that is unbiased under π_old sampling; Off-Policy Sequence Masking; "Keep Routing" (reuse inference-time expert routes in training, called crucial for MoE RL and used since DeepSeek-V3-0324); "Keep Sampling Mask" (apply the top-p/top-k truncation mask from sampling to π_θ) [DSV3.2 §3.1]. Post-training compute exceeded 10% of pretraining [§1].

## Gotchas when reproducing

- Loss normalization is an implementation detail with real effects: per-response mean (GRPO), per-batch token mean (DAPO, verl), and constant normalization (Dr. GRPO) give different length dynamics [Dr. GRPO §3.1].
- With a token-level ratio and a sampling engine whose numerics differ from the trainer, π_old should be recomputed by the trainer (GSPO §5.4) or the mismatch handled explicitly (DeepSeek-V3.2 masking, truncated importance sampling in Kimi Linear §5.4.2).
- MoE: either keep routing fixed between sampling and training (DeepSeek "Keep Routing", Qwen "Routing Replay") or use a sequence-level ratio (GSPO).
- GSPO's clip range is around 1e-4 scale because s_i is a geometric mean; do not reuse GRPO's 0.2 [GSPO §4.1, §5.1].
- DAPO's dynamic sampling makes each batch cost a variable number of rollouts; the paper says wall-clock was not much affected because long-tail generation dominates [§3.2].
- Base-model and template effects can masquerade as RL gains: Qwen2.5-Math base models answer well without any template, and templates can destroy the ability RL then "recovers" [Dr. GRPO §2.2, §3.3].

## Connections

- The skill's post-training file covers SFT and DPO only. These objectives belong in a new RL section: GRPO as the base, Dr. GRPO's unbiased normalization and DAPO's clip-higher and dynamic sampling as the standard fixes, GSPO or routing replay for MoE.
- Architecture choices reach into RL: MoE routing volatility (GSPO §5.3, DSV3.2 Keep Routing), and Kimi Linear reports faster RL improvement than full MLA at matched recipe [KL Fig. 6].
- DPO comparison: DeepSeekMath §5.2.1 places RFT, DPO, PPO and GRPO in one gradient-coefficient framework (not read in detail here).

## Disagreements and errors

- DeepSeekMath calls the k3 estimator "unbiased" [Eq. 4]; DeepSeek-V3.2 §3.1 says that under sampling from π_old the gradient of k3 is biased and adds the ratio r_{i,t} to fix it, noting k3's gradient gives unbounded weight to tokens with π_θ ≪ π_ref.
- DAPO presents the 1/Σ|o_i| token-level loss as the fix for GRPO's length weighting [§3.3]; Dr. GRPO lists exactly this per-batch length normalization (verl, trl) as still biased because Σ|o_i| varies between batches, and normalizes by a constant [§3.1, code listing].
- DAPO keeps std normalization; Dr. GRPO and DeepSeek-V3.2 drop it. No head-to-head ablation of std normalization in DAPO's paper.
- GSPO says GRPO "necessitates" Routing Replay for MoE [§5.1]; DeepSeek-V3.2 keeps token-level GRPO and treats Keep Routing as the fix. Different labs, different models; neither runs the other's alternative.
- DeepSeekMath Algorithm 1 says to maximize "the GRPO objective (Equation 21)"; the objective in §4.1.1 is Eq. 3.
