# Post-training: SFT, preference tuning, and RL

The base model predicts likely continuations. SFT makes it respond to
instructions. Preference tuning changes how it responds. Each stage answers a
question the previous one structurally cannot.

| Stage | Signal | What it can express |
|---|---|---|
| Pretraining | next token over a corpus | what text looks like |
| SFT | next token over curated (prompt, response) pairs | "produce this" |
| Preference tuning | pairwise (chosen, rejected) | "produce this, not that" |

SFT has no way to say "do not generate that." It can only demonstrate. That is
the structural reason preference tuning is a separate stage rather than more
SFT data, and it is worth stating in exactly those terms when justifying the
pipeline.

Preference tuning does not add knowledge. If the model is factually wrong, the
SFT data is wrong. Reaching for DPO to fix facts is a category error.

## Contents

- [SFT](#sft)
- [DPO: the derivation](#dpo-the-derivation)
- [DPO hyperparameters](#dpo-hyperparameters)
- [DPO failure modes](#dpo-failure-modes)
- [Variants](#variants)
- [RL with verifiable rewards](#rl-with-verifiable-rewards-grpo-and-its-fixes)
- [Evaluation](#evaluation)
- [Ordering](#ordering-and-the-reference-model)

## SFT

**Loss masking.** Standard practice is to zero the loss on prompt tokens and
train only on the completion. Llama 2: "we utilize an autoregressive objective
and zero-out the loss on tokens from the user prompt." TRL's `SFTConfig` does
this by default only for prompt-completion datasets (`completion_only_loss`); for
chat `messages` datasets set `assistant_only_loss=True`, which needs a chat template
that marks assistant spans.

It is not as settled as it looks. Huerta-Enochian and Ko ran a controlled study
and found a statistically significant negative quadratic relationship between
prompt-loss-weight and performance on *short-completion* data: a small non-zero
weight (0.01 to 0.5) beat both full masking and full weighting. On
long-completion data, which is what modern chat SFT is, the relationship is
weak and masking is fine. So: mask by default for chat data, and if your
completions are short (classification-style, multiple choice), try a small
prompt weight before assuming masking is optimal.

**Packing.** Concatenating short examples to fill the sequence avoids wasting
compute on padding, which can be up to 50% of tokens. The risk is attention
leaking across unrelated documents. The fix is a block-diagonal document mask
so each token attends only within its own example. Llama 3 does this at
pretraining too and calls it "important" for long-context continued
pretraining. TRL exposes `packing=True` with `packing_strategy='bfd'`.

**The EOS bug.** If `pad_token == eos_token` and padding is masked from the
loss, every EOS in the batch gets masked, so the model is never trained to emit
EOS at the end of a response. Symptom: the fine-tuned model never stops, and
rambles into hallucinated follow-up turns. Fix: use a distinct pad token, or a
dedicated end-of-turn token. LIMA introduced an explicit EOT token for exactly
this reason, "to avoid conflation with any other meaning that the pretrained
model may have imbued into the preexisting EOS token." TRL exposes an
`eos_token` override on `SFTConfig` for the same reason (for Qwen2.5 you set
`eos_token="<|im_end|>"`).

**How much data.** LIMA got a 65B model to be judged equal-or-better than GPT-4
in 43% of a 300-prompt study using 1000 curated examples and no RLHF, and
argued for the Superficial Alignment Hypothesis: knowledge comes from
pretraining, alignment selects a format. Its own ablations show the result is
about quality and diversity, not quantity.

Real recipes use much more. Llama 2 stopped at 27,540 annotations and reports
"SFT annotations in the order of tens of thousands was enough." Tulu 3 uses
866K to 939K prompts. This tension is unresolved: LIMA's result holds for its
specific setup, and revealed practice across every major open recipe is
inconsistent with 1000 examples being enough for a general-purpose assistant.

**Hyperparameters.**

| Source | LR | Batch | Epochs | Notes |
|---|---|---|---|---|
| Llama 2 | 2e-5 cosine, wd 0.1 | 64 | 2 | seq 4096, full FT |
| Llama 3 405B | 1e-5 | | | |
| Zephyr | 2e-5 cosine, 10% warmup | | 1 (optimal before DPO) | |
| OLMo 2 7B | 1e-5 (swept 1e-5 to 3e-5) | 128 | 2 | sum-reduction loss |
| OLMo 2 13B | 5e-6 (swept 1e-6 to 8e-6) | 128 | | |
| Tulu 3 8B | 5.0e-6 | 128 | | full FT |
| Tulu 3 70B | 2.0e-6 | 128 | | |
| TRL default | 2e-5 | 8/device | 3.0 | `max_length=1024` |

**LoRA versus full fine-tune.** Tulu 2 tested QLoRA at the SFT stage and found
it underperformed full fine-tuning by 10 to 20% on AlpacaEval, narrowing to 3%
by 70B, and did not attempt QLoRA for the preference stage at all. Zephyr,
Llama 2/3, Tulu 3 and OLMo 2 all use full fine-tuning for their headline
models. LoRA facts worth knowing, from LoRA Without Regret (Thinking Machines,
2025, a vendor blog rather than a paper): it wants roughly 10x the full
fine-tuning learning rate, it pays a larger penalty at large batch sizes, and
applying it to all layers, especially the MLP, beats attention-only at matched
parameters. Biderman et al. (TMLR 2024) add that LoRA forgets less but learns less.

## DPO: the derivation

Worth being able to reproduce, because it is the cleanest argument in the
alignment literature and it comes up in interviews.

**Step 1.** The RLHF objective is reward maximization with a KL leash to the
reference policy:

```
max_pi  E_{x~D, y~pi}[ r(x,y) ]  -  beta * KL( pi(y|x) || pi_ref(y|x) )
```

`pi_ref` is the frozen SFT model. `beta` sets how far the policy may drift. The
leash exists because the reward model is imperfect, so unconstrained
maximization finds the proxy's flaws rather than the objective.

**Step 2.** For a fixed reward, this objective has a closed-form maximizer:

```
pi*(y|x) = (1/Z(x)) * pi_ref(y|x) * exp( r(x,y) / beta )
```

`Z(x) = sum_y pi_ref(y|x) exp(r(x,y)/beta)` is a partition function, a sum over
every possible completion. Intractable, which is why this form cannot be
optimized directly.

**Step 3.** Take logs and solve for the reward:

```
r(x,y) = beta * log( pi*(y|x) / pi_ref(y|x) )  +  beta * log Z(x)
```

**Step 4.** Substitute into Bradley-Terry, which models a pairwise preference
as a sigmoid of a reward difference: `P(y1 > y2 | x) = sigmoid(r(x,y1) - r(x,y2))`.
Because this is a *difference* of two rewards at the same `x`, and `Z(x)`
depends only on `x`, the `beta log Z(x)` terms cancel exactly. This is the whole
trick.

**Step 5.** What remains is a preference probability written entirely in terms
of the policy and the reference. Fit it by maximum likelihood:

```
L_DPO = - E[ log sigmoid( beta*log(pi(y_w|x)/pi_ref(y_w|x))
                        - beta*log(pi(y_l|x)/pi_ref(y_l|x)) ) ]
```

No reward model, no rollouts, no value function. Two model copies instead of
PPO's four (policy, value, reward, reference). Hence the paper's title: your
language model is secretly a reward model.

The **implicit reward** is `r_hat(x,y) = beta * (log pi(y|x) - log pi_ref(y|x))`.
It is what you log to see whether training is working.

## DPO hyperparameters

Learning rates are 5 to 20 times lower than SFT at the same scale. This is the
single most common thing to get wrong.

| Source | LR | beta | Batch | Notes |
|---|---|---|---|---|
| DPO paper | 1e-6 (RMSprop, 150-step warmup) | 0.1, 0.5 for TL;DR | 64 | beta not tuned |
| Llama 3 | 1e-5 | 0.1 | | plus NLL term, alpha 0.2 |
| Zephyr | 5e-7 linear, 10% warmup | 0.1 | 32 | 3 epochs optimal |
| Tulu 2 | 5e-7 | 0.1 | 32 | 3 epochs, seq 8192 |
| Tulu 3 8B | 5.0e-7 | | 128 | length-normalized variant |
| OLMo 2 7B | 1e-6 (swept 5e-7 to 1e-6) | | | |
| OLMo 2 13B | 8e-7 | | | |
| KTO paper's note | "default LR for DPO is 5e-7" | 0.1 optimal in most settings | 32 | |
| TRL default | 1e-6 | 0.1 | 8/device | |

Same-model comparisons: OLMo 2 7B is SFT 1e-5 and DPO 1e-6, a 10x drop.
Tulu 3 8B is 5e-6 and 5e-7, also 10x. TRL's defaults are 2e-5 and 1e-6, 20x.

`beta = 0.1` is the near-universal starting point. Higher keeps the policy
closer to the reference; lower lets it move further at the cost of KL.

SmolTulu found that for small models (135M to 1.7B) a *higher* LR-to-batch
ratio than Tulu 3's large-model recipe improved reasoning benchmarks, while a
lower one was better for pattern-recognition benchmarks. So scaling down a
large-model recipe is not automatically right.

## DPO failure modes

**Likelihood displacement.** The log-probability of the *chosen* response
often falls during training, even while the margin correctly increases. Razin
et al. (2024) show this can be catastrophic rather than benign: training a
model to prefer "No" over "Never" sharply increases the probability of "Yes."
In a real case, DPO-training Llama-3-8B-Instruct to prefer refusals over
compliance on unsafe prompts drove its refusal rate from 74.4% down to 33.4%.
DPO unaligned a safety-tuned model.

They trace severity to a centered hidden embedding similarity (CHES) score
between chosen and rejected: the more similar their internal representations,
the more the chosen likelihood drops. Filtering high-CHES pairs prevented the
unalignment better than adding an SFT term did.

**Both logprobs falling.** Yan et al. ("3D-Properties") show the gradient ratio
is `-pi_chosen/pi_rejected`, so as the rejected probability approaches zero its
gradient explodes relative to the chosen side, and the chosen-side gradient
vanishes. DPO degenerates from "steer toward good" into "suppress bad," and
because probability mass is conserved, it lands on responses outside the
training pairs entirely. The effect is mildest when both chosen and rejected
are on-policy, because on-policy rejected responses start with higher
probability and the collapse is delayed.

**Length exploitation.** DPO's implicit reward is an unnormalized sum of token
log-probabilities, so more tokens means more terms and longer responses are
structurally favored. SimPO's fix is to divide by response length. Their
ablation shows this is data-dependent, not universal: adding length
normalization to DPO helped a lot on Mistral-Base (LC win rate 15.1% to 21.0%)
and *hurt* on Mistral-Instruct (26.8% to 21.7%).

**Mitigations, with sources.**

| Fix | Mechanism | Source |
|---|---|---|
| Add an NLL term on the chosen response | `L = L_DPO + alpha * NLL(y_w)` | Llama 3 uses alpha 0.2; RPO / Pang et al. call it "crucial" for reasoning |
| On-policy pair generation | sample completions from the current or recent policy | Llama 3 (6 rounds), Tulu 3, OLMo 2 |
| Length normalization | reward = mean, not sum, of token logprobs | SimPO |
| CHES filtering | drop pairs whose chosen and rejected have similar hidden embeddings | Razin et al. |
| Asymmetric beta | separate beta for the chosen and rejected terms | 3D-Properties |

## Variants

| Method | Difference from DPO | When |
|---|---|---|
| IPO | replaces the logistic transform with the identity, so the optimum stays regularized toward the reference even under near-deterministic preferences | small or noisy preference sets |
| KTO | unpaired binary desirable/undesirable labels via a Kahneman-Tversky loss, no pairs needed | data is thumbs up/down, or the classes are imbalanced |
| ORPO | no reference model at all: SFT loss plus an odds-ratio contrastive penalty, one stage | skip the two-stage pipeline; half the forward passes |
| SimPO | reference-free, length-normalized average logprob, plus a target margin | maximize win rate with minimal length exploitation |
| cDPO | label smoothing on the DPO logit | known-noisy crowd labels |
| RPO | DPO plus an NLL term on the chosen chain of thought | reasoning tasks, where plain DPO degrades correctness |

The 2025-2026 direction is not a new pairwise loss. Tulu 3 and OLMo 2 keep
on-policy DPO and add RLVR (PPO or GRPO against a verifiable 0/1 correctness
reward) as a *subsequent* stage for math, code, and constrained instruction
following.

## RL with verifiable rewards: GRPO and its fixes

**The idea.** For each prompt, sample a group of G responses, score each with a verifier (0/1 for a correct final answer or passing tests), and push up the tokens of responses that beat the group's average. The group mean replaces PPO's learned value function, which is the expensive part of PPO for LLMs [Dr. GRPO §3].

**Notation.** `G` responses per prompt; `o_i` response i with `|o_i|` tokens; `R_i` its reward; `r_it = pi_theta(o_it) / pi_old(o_it)` the per-token importance ratio against the policy that sampled it; `A_i` the advantage given to every token of response i; `eps` the clip range.

**GRPO** [2402.03300]: `A_i = (R_i - mean R) / std R`, loss `(1/G) sum_i (1/|o_i|) sum_t min(r_it A_i, clip(r_it, 1-eps, 1+eps) A_i)`, minus a KL penalty to a reference model estimated per token with the k3 estimator (`ratio - log ratio - 1`, ratio = ref/policy). k3 is unbiased for samples from the current policy; with samples from pi_old its gradient is biased, and DeepSeek-V3.2 multiplies it by the importance ratio to correct this [DSV3.2 §3.1; `references/papers/rl-post-training.md`]. For MoE policies, DeepSeek-V3.2 also reuses the inference-time expert routes during training ("Keep Routing"), which it calls crucial for MoE RL.

Five documented problems, and the fix each paper proposes:

| Problem | Mechanism | Fix | Source |
|---|---|---|---|
| Length bias | `1/|o_i|` spreads a wrong answer's penalty over its own length, so long wrong answers are cheap and length grows without accuracy | divide by a constant (the generation budget) | Dr. GRPO [2503.20783 §3.2]; common PPO libraries are length-normalized too, OpenRLHF per response and trl / verl per batch, both biased (its Table 2, Listing 1) |
| Difficulty bias | dividing by the group's std gives near-solved and near-hopeless prompts the largest weights | drop the std (Dr. GRPO) | [2503.20783 §3.1] |
| Zero-gradient groups | all-correct or all-wrong groups have `A_i = 0` for every response | over-sample and keep only groups with `0 < correct < G` ("dynamic sampling") | DAPO [2503.14476 Eq. 10] |
| Entropy collapse | the symmetric clip caps how fast low-probability tokens can rise | asymmetric clip, eps_low 0.2, eps_high 0.28 ("clip-higher") | DAPO [§3.1, §4.1] |
| Token-level ratios, sequence-level reward | per-token ratios add noise that accumulates with length; reported to collapse MoE RL runs | sequence ratio `s_i = exp(mean_t log r_it)`, clipped per response | GSPO [2507.18071 Eqs. 5-7] |

DAPO also averages the loss over all tokens in the group (`1 / sum_i |o_i|`), which weights every token equally, and shapes rewards for over-long responses; its ablation table credits each piece with a few AIME points on Qwen2.5-32B (30 for naive GRPO, 50 with all of them) [DAPO Table 1]. GSPO notes its clip range differs from GRPO's by orders of magnitude because the ratio is defined differently [§4.2]; do not reuse eps = 0.2.

**DAPO's training setup** [§4.1]: AdamW, constant LR 1e-6 with 20-step warmup, 512 prompts x 16 responses per rollout step, minibatch 512 (16 gradient updates per rollout step), max generation 20,480 tokens (16,384 expected plus a 4,096-token soft-penalty zone).

Most of these papers drop the KL term (beta = 0) for verifier rewards, since there is no learned reward model to drift away from [Dr. GRPO §3]. With a learned reward model, keep it.

`scripts/post_training_losses.py --verify` checks each property above (the 10x per-token penalty gap, equal weights under Dr. GRPO and DAPO, clip-higher keeping a ratio of 1.25, whole-sequence clipping in GSPO, k3 being non-negative and unbiased). `--explain` prints the mechanisms.

**Two cross-stage rules from the 2025-2026 papers.** Keep the optimizer the same through post-training when the base was pretrained with Muon (Moonlight Table 6, MMLU after SFT: Muon base with Muon SFT 55.7, the same base with AdamW SFT 50.2, an AdamW base with AdamW SFT 52.0; other benchmarks in the table are mixed). And judge architecture shortcuts after post-training, not on pretraining loss alone: Qwen3.8-Next's NoPE global layers and sparse residual reads looked fine in pretraining and failed only after post-training [§2.1.1, §2.2].

**What to log.** Mean response length and accuracy together (length up with flat accuracy is the length bias), the fraction of zero-variance groups, policy entropy, clip fraction, and for MoE models the router load.

## Evaluation

**Training health.** Log the implicit reward accuracy (fraction of pairs where
chosen scores above rejected) and the reward margin. TRL emits these as
`rewards/accuracies`, `rewards/chosen`, `rewards/rejected`, `rewards/margins`.
Also log both raw logprobs, because the failure mode above is invisible in the
margin alone: the margin can look healthy while both are falling.

**Quality.** The DPO paper's own evaluation tool is the reward-versus-KL
frontier, because "achieving slightly higher reward but with much higher KL is
not necessarily desirable." Sweep beta, plot both.

**Win rate.** AlpacaEval 2 length-controlled and Arena-Hard. Use the
length-controlled number. The raw one conflates quality with verbosity, which
the DPO paper itself hit in 2023: GPT-4 "prefers longer, more repetitive
summaries than humans do," and they had to switch to a "concise" judging prompt.

## Ordering and the reference model

**Do not over-train SFT first.** Zephyr's ablation found 1 epoch of SFT then 3
epochs of DPO was optimal, and that "if the SFT model is trained for more than
one epoch, the DPO step actually induces a performance regression with longer
training."

**Do not skip SFT.** Zephyr tested DPO straight on the base model: MT-Bench
4.76 versus 7.00 for SFT-then-DPO. "Without an initial SFT step, models are not
able to learn at all from feedback and perform terribly," and the model never
learns the chat template.

**Do not run DPO on pairs whose chosen response the SFT stage already saturated.**
This is the 3D-Properties mechanism applied to pipeline design: if the chosen
response already has probability near 1, the chosen-side gradient vanishes and
the update degenerates into pure suppression, which is the engine of likelihood
displacement.

**The reference model is conventionally the SFT checkpoint**, and the DPO
paper's own dialogue experiment constructs it that way so the completions are
in-distribution. It is a strong default, not a theoretical requirement. TR-DPO
updates the reference on an exponential moving average every k steps and Llama
3 used this for its vision post-training, reporting better human evaluations.
TRL exposes it as `sync_ref_model` / `ref_model_mixup_alpha` /
`ref_model_sync_steps`.

**Iterative is the frontier default.** Llama 3 runs six full SFT-then-DPO
rounds, each collecting fresh preferences sampled from the latest model,
explicitly so the training data matches the current policy's distribution. That
is a mitigation for the off-policy failure modes above, not just a performance
optimization.
