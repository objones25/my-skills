# Do New Attention Mechanisms Actually Fix Attention Sinks at Million-Token Context?

Sara Rizwan, Samaanah Abdus Salam (Shadan Women's College of Engineering and Technology), Mohammed Mudassir Uddin (Muffakham Jah College of Engineering & Technology), Hyderabad. arXiv:2609.08574 (v2, 18 Sep 2026; v1 8 Sep 2026). Preprint, no venue. Posted after June 2026: treat with extra care. Checked: full text of §I-VI.B from the PDF (pdftotext); §VII-VIII (protocol and limitations) skimmed.

## The idea

A small controlled study asking whether gated attention and a Kimi-K3-style hybrid remove attention sinks. Four ~1M-parameter models (8 layers, width 96) are trained on a synthetic key-value retrieval task for 1500 steps on a laptop, differing only in token/depth mixing: softmax, softmax with a channel-wise output gate (Qiu et al.'s mechanism), a 3:1 linear-to-global hybrid, and the hybrid with attention residuals [Rizwan §IV.B, Table V]. Three findings: the training objective creates the sink (predicting at every position vs only at the answer: sink mass 0.316 vs 0.049); the gate's sink reduction (0.316 to 0.261) was inside seed noise (baseline seeds 0.089-0.455), so "gating did not reproduce its published effect at our scale"; and sink mass, massive activation and position bias moved independently [Abstract, Tables VII-IX].

The key caveat is in the paper itself: their gate's mean score was 0.686 after 1500 steps, versus 0.116 in Qiu et al. after trillions of tokens, so the gate never became sparse [§VI.C]. This is evidence that the sink fix needs a trained, sparse gate, not evidence that the gate fails at scale.

## Notation

| Symbol | Meaning |
|---|---|
| σ (sink mass) | Mean attention on position 0, over softmax layers, heads and query positions t ≥ 2 [Eq. 1] |
| µ | Mean over layers of the largest absolute hidden value [Eq. 2] |
| H | Mean attention entropy (nats) [Eq. 3] |
| p(d) | Recall when the answer sits at relative depth d ∈ [0, 1] |
| Δr | Recency gap: mean recall for d in (0.75, 1] minus mean recall for d in [0, 0.25] [Eq. 4] |
| a_t | Channel-wise decay of their simplified linear layer (no delta rule) [Eq. 5] |

## The method

1. Four models, identical except mixing: 8 layers, width 96, 4 heads of width 24, SwiGLU FFN 256, RMSNorm pre-norm, vocab 70, AdamW (0.9, 0.95) wd 0.01, LR 2e-3 one-cycle with 8% warmup, clip 1.0, batch 16, 1500 steps, float32, 3 seeds [Table V].
2. Task: haystack of a repeated 24-symbol motif with half the positions randomized, one planted key-value pair plus 3 distractor pairs, question at the end; train length 96, evaluated to 768 [§IV.E, §V.B].
3. Loss: answer position plus an auxiliary next-token term at weight 0.3 (0.0 for the control) [Table V].
4. The linear layer omits the delta rule, so it is not KDA or Gated DeltaNet [§IV.C].

## Key results

- Objective control (same softmax stack): sink mass 0.049 with answer-only loss vs 0.316 with every-position loss; worst layer 0.067 vs 0.633; recall unchanged (33.3% vs 32.0%) [Table VII].
- At train length 96, sink mass: softmax 0.316 ± 0.198, + gate 0.261 ± 0.175, hybrid 0.183, hybrid + attention residuals 0.202 [Table IX].
- Gate halved massive activation (µ 74.9 to 36.2) without removing the sink [Table VIII].
- Both hybrids never learned the task (recall 5.7-6.0% vs 8.3% chance), so their sink numbers come from models that do not retrieve [§VI.B].
- Haystack predictability changed the baseline sink mass from 0.095 (15% randomized) to 0.438 (50%) in a single-seed calibration [§V.A].

## Gotchas when reading it

- Four orders of magnitude smaller than Qiu et al., synthetic data, 1500 steps; the gate stayed open (0.686 mean) [§VI.C].
- The hybrid rows are confounded by failed training (no delta rule) [§VI.B, §VIII].
- The paper cites "Kimi K3" (arXiv 2607.24653) and "Attention Residuals" (arXiv 2603.15031) for the large-model design it mimics; neither was read for this digest.
- It states Qiu et al.'s gated attention won a NeurIPS 2025 best paper award [§I, ref. 2]; not verified here.

## Connections

- Gated attention (`gated-attention.md`): supports the paper's own mechanism story (sparsity matters, NS-sigmoid with a 0.5 floor brought the sink back), and adds that a gate which has not learned to close does nothing for sinks. Do not expect sink removal in short proxy runs.
- Stability proxies: the skill uses small proxies (Wortsman et al.) for instabilities. This paper is a warning that sink and gate behavior may not show up at proxy scale or short horizons.
- Objective matters: next-token loss at every position is what creates sinks [Table VII], consistent with StreamingLLM's explanation (`attention-sinks.md`).

## Disagreements and errors

- Abstract: "Gating did not reproduce its published effect at our scale." Accurate as stated, but the paper's own §VI.C shows why (gate never became sparse), which makes it weak evidence against Qiu et al. at the scale Qiu et al. tested.
