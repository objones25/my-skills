# 05: "GRPO responses keep getting longer, accuracy is flat"

*Illustrative scenario.* A math RL run with 0/1 verifier rewards, G = 8, the loss averaged per response. Mean response length doubled over 300 steps; accuracy did not move.

## Verdict

Length bias from per-response averaging. A wrong answer's penalty is spread over its own length, so long wrong answers are cheap. Switch the loss to a constant denominator (Dr. GRPO) or token-level averaging over the group (DAPO), then re-check whether length still grows.

## The check

```
$ python scripts/post_training_losses.py --verify
Length bias (gradient per token at ratio 1, one short and one long wrong answer)
  [PASS] GRPO: a token in the long wrong answer is penalized 10x less  per-token grad +0.0500 vs +0.00500; long wrong answers are cheap, so length grows
  [PASS] dr_grpo: every token weighted the same  +0.00391 vs +0.00391
  [PASS] dapo: every token weighted the same  +0.00909 vs +0.00909
```

Common PPO libraries are length-normalized too: OpenRLHF averages per response, trl and verl per batch, and Dr. GRPO counts both as biased [§3.1, Table 2, Listing 1]. Check what your trainer does, not what the paper's equation says.

## Two more things to look at in the same run

- **Zero-variance groups.** Prompts where all 8 answers are right (or all wrong) have zero advantage and add no gradient. If that fraction grows, the effective batch shrinks; DAPO keeps only groups with 0 < correct < G.
- **Std normalization.** With 0/1 rewards, a 7/8-correct group gives its one wrong answer |A| = 2.47 against 0.94 in a 4/8 group (`--explain grpo-std-bias`). Easy prompts dominate.

## What would change it

If length grows with a constant-denominator loss too and accuracy rises with it, the task may reward longer reasoning; cap with an overlong penalty rather than fighting it.
