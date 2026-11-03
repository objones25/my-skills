# ST-MoE: Designing Stable and Transferable Sparse Expert Models (router z-loss)

Barret Zoph, Irwan Bello, Sameer Kumar, Nan Du, Yanping Huang, Jeff Dean, Noam Shazeer, William Fedus (Google). arXiv:2202.08906 (v2, 29 Apr 2022; v1 17 Feb 2022). No venue printed. Checked: full text of §3 (stability) and §5 (design); §4 (fine-tuning), §6-7 and appendices skimmed only.

## The idea

MoE routers add an exponential (the router softmax) whose inputs can grow large, and in bfloat16 large inputs carry large roundoff error [ST-MoE §3.4]. The router z-loss, L_z = (1/B) Σ_i (log Σ_j exp x_ij)², squared log-partition of the router logits averaged over tokens, pushes those logits toward small magnitude. In a stability study where the baseline diverged in 2 of 6 seeds, z-loss made 3 of 3 seeds stable and slightly improved quality, while the other fixes tested (dropout, input jitter, tight update clipping, removing GEGLU or RMSNorm gains) bought stability at a quality cost [Tables 2-4]. It is the router-level version of PaLM's output-logit z-loss.

## Notation

| Symbol | Meaning |
|---|---|
| B | Number of tokens in the batch |
| N | Number of experts |
| x | Router logits, shape (B, N) |
| L_z | Router z-loss [Eq. 5] |
| L_B | Auxiliary load-balance loss (from Switch Transformer) |
| c_B, c_z | Weights of L_B and L_z in L_tot = L_CE + c_B L_B + c_z L_Z [Eq. 6] |
| CF | Capacity factor: tokens an expert may take, as a multiple of an even split; overflow is dropped [§5.2] |
| Quality | Negative log perplexity on English (higher is better) [§3] |

## The method

1. Compute router logits in float32 (already standard, "selective precision") [§3.3].
2. Add c_z · L_z to the loss with c_z = 0.001, chosen by a sweep on pre-training quality [§3.3].
3. Cast every exponentiated tensor to float32 [§3.4].
4. Do not replace this with logit clipping: clipping acts after roundoff has already happened and adds a discontinuity [§3.4].

## Hyperparameters

| Setting | Value | Source |
|---|---|---|
| c_z | 0.001 | §3.3 |
| c_B | not given in the sections read | |
| Stability study model | sparse, FLOP-matched to T5-XL, 32 experts, MoE every 4th FFN, train CF 1.25, eval CF 2.0, masked LM on mC4, 20k steps, Adafactor, batch 1M tokens | §3, Fig. 1 |
| Seeds | 6 for baseline, 3 per variant | §3 |
| Routing recommendation | top-2, train CF 1.25, at most one expert per core (their TPU setup) | §5 |
| ST-MoE-32B | 269B parameters, FLOP-matched to a 32B dense model, encoder-decoder, d_kv 128 | §6.2 |

## Key results

Stability study, fraction of seeds stable and quality [Tables 2-4]:

| Change | Stable | Quality |
|---|---|---|
| Baseline | 4/6 | -1.755 ± 0.02 |
| Remove GEGLU (dense ReLU at equal FLOPs) | 3/3 | -1.849 |
| Remove RMSNorm scale parameter | 3/3 | -2.020 |
| Input jitter 1e-2 | 3/3 | -1.777 |
| Dropout 0.1 | 3/3 | -1.822 |
| Update clipping 0.1 | 3/3 | -4.206 |
| Router z-loss | 3/3 | -1.741 ± 0.02 |

- Z-loss on attention logits also improved stability without hurting quality (footnote, no numbers) [§3.3 fn. 4].
- Raising capacity factor helps (top-1: +0.011 neg. log perp. from CF 1.0 to 1.25), and top-2 beats top-1 by 0.004 at CF 1.25; tripling a dense model gives +0.090 for scale [§5.2].
- Gains from more experts diminish beyond about 256 experts, or below about 1% of experts active [§5.1].

## Gotchas when reproducing

- Encoder-decoder, masked-LM, Adafactor, TPU. The authors say decoder-only models were more stable in their hands and conjecture fewer exponentials as the reason [§3.4, §9]. Transfer to a decoder-only AdamW MoE is plausible but not shown here.
- Three seeds per variant is a small sample for "3/3 stable"; the baseline's instability rate was about 1/3.
- The CF and top-n recommendations predate fine-grained experts (DeepSeekMoE) and dropless routing; DeepSeek-V3 drops no tokens [DSV3 §2.1.2].

## Connections

- Stability: extends the skill's z-loss section from the output softmax to the router softmax. Same functional form as PaLM's, different coefficient (1e-3 here vs PaLM's 1e-4 on output logits).
- The Qwen gated-attention MoE baseline uses z-loss and a global-batch balance loss [GatedAttn §3.1]; DeepSeek-V3 reports neither z-loss nor QK-norm and balances with a bias instead (`deepseek-v3.md`).
- Note against the skill's defaults: removing GEGLU or RMSNorm gains improved stability here; SwiGLU-type gates and norm gains are multiplicative interactions that cost some stability for quality [§3.1].
