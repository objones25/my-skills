# Muon is Scalable for LLM Training (Moonlight)

Jingyuan Liu, Jianlin Su, Xingcheng Yao, Zhejun Jiang, Guokun Lai, Yulun Du, ... , Zhilin Yang (Moonshot AI; full list on arXiv). arXiv:2502.16982 (v1, 24 Feb 2025, the only version). Technical report. Checked: full text of §1-5 from the arXiv HTML; appendices (proof of Lemma 1, AdamW grid search, Moonlight architecture details) not read.

## The idea

Muon updates each weight matrix with an orthogonalized version of its momentum (all singular values pushed toward 1 by a Newton-Schulz iteration). Two changes made it work at billions of parameters and trillions of tokens [Moonlight §2.2]: add AdamW-style decoupled weight decay, because without it weights and layer outputs kept growing past bf16's precise range; and rescale each matrix's update by 0.2·sqrt(max(A, B)), which makes every matrix's update RMS about 0.2, close to AdamW's typical 0.2-0.4. With that scaling, Muon reuses AdamW's tuned learning rate and weight decay with no new sweep. In a compute-optimal scaling study on dense Llama-style models from 399M to 1.5B, Muon matched AdamW at about 52% of the training FLOPs [§3.2, Fig. 1a]. Moonlight, a 16B-total / 3B-active MoE trained on 5.7T tokens with Muon, beat an otherwise identical AdamW run at 1.2T tokens on most benchmarks [Table 4].

## Notation

| Symbol | Meaning |
|---|---|
| W_t | A weight matrix at step t, shape (A, B) |
| ∇L_t | Gradient of the loss with respect to W |
| M_t | Momentum buffer, M_t = μ M_{t-1} + ∇L_t |
| μ | Momentum coefficient (0.95) |
| O_t | Orthogonalized momentum, about U Vᵀ where U Σ Vᵀ = M_t (SVD) [§2.1] |
| X_k | Newton-Schulz iterate; X_0 = M_t / ||M_t||_F |
| a, b, c | Newton-Schulz polynomial coefficients (3.4445, -4.7750, 2.0315) |
| N | Newton-Schulz steps (5) |
| η_t | Learning rate |
| λ | Weight decay coefficient |
| RMS(·) | Root mean square of a matrix's entries |
| H | Model hidden size |

## The method

### 1. Orthogonalize the momentum

X_k = a X_{k-1} + b (X_{k-1} X_{k-1}ᵀ) X_{k-1} + c (X_{k-1} X_{k-1}ᵀ)² X_{k-1}, five steps, starting from the Frobenius-normalized momentum [Eq. 2]. Ten steps orthogonalize more accurately but did not train better [§2.2].

### 2. Add decoupled weight decay

W_t = W_{t-1} − η_t (O_t + λ W_{t-1}) [Eq. 3]. On an 800M model trained 100B tokens (about 5x compute-optimal), vanilla Muon converged faster early but was overtaken; Muon with decay beat both vanilla Muon and AdamW in the over-trained regime [§2.2, Fig. 2; plot only].

### 3. Match update RMS across shapes and to AdamW

Lemma 1: a full-rank (A, B) matrix's orthogonalized update has RMS sqrt(1/max(A, B)) [§2.2]. So wide MLP matrices get too-small updates and small matrices (e.g. a per-head KV projection treated separately) get too-large ones. Final rule: W_t = W_{t-1} − η_t (0.2 · O_t · sqrt(max(A, B)) + λ W_{t-1}) [Eq. 4]. Non-matrix parameters (RMSNorm gains, embeddings, LM head) stay on AdamW with the same η and λ [§2.2].

### 4. Distributed Muon

ZeRO-1 style: reduce-scatter gradients, apply momentum on the local shard, gather the full matrix over data-parallel ranks, run Newton-Schulz in bf16, keep only the local shard of the update, all-gather parameters [Alg. 1]. One momentum buffer instead of AdamW's two; communication 1x to 1.25x of distributed AdamW; optimizer latency 1-3% of a step [§2.3].

## Hyperparameters

| Setting | Value | Source |
|---|---|---|
| Momentum μ | 0.95 (not tuned further; no consistent gain) | §2.2 |
| Newton-Schulz steps, coefficients | 5; (3.4445, -4.7750, 2.0315) from Jordan et al. | §2.1-2.2 |
| Update scale | 0.2 · sqrt(max(A, B)) per matrix | Eq. 4 |
| LR and weight decay | reused from the AdamW optimum (no separate Muon sweep) | §2.2, §3.2 |
| Scaling-law models | dense Llama-style, 399M-1.5B non-embedding params, 8.92B-38.91B tokens, LR 9.503e-4 to 8.305e-4, batch 96-256 sequences of 8K | Table 2 |
| Moonlight architecture | DeepSeek-V3-small style MoE, 2.24B active / 15.29B total excluding embeddings (3B / 16B with) | §3.3 |
| Moonlight schedule | warmup to 4.2e-4 over 2K steps (to 33B tokens); cosine to 4.2e-5 until 5.2T; cooldown: up to 1e-4 in 100 steps then linear to 0 over 500B tokens | §3.3 |
| Moonlight batch | 2048 sequences to 200B tokens, then 4096; 8K context | §3.3 |
| Moonlight weight decay | 0.1 all stages | §3.3 |
| Aux-free bias update rate | 1e-3 in stages 1-2, 0 in stage 3 | §3.3 |
| β for the AdamW part, eps, grad clip | not given in the sections read | |

## Key results

- Scaling fit (LM loss at 8K): Muon 2.506·C^-0.052, AdamW 2.608·C^-0.054; Muon needs about 52% of AdamW's FLOPs at compute-optimal settings [Table 3, §3.2].
- Update-RMS control at 800M (with a 4H plain MLP to exaggerate shape differences), validation loss after 4B of 20B tokens: single 0.2·sqrt(H) scale 2.812; per-matrix normalization 2.789; per-matrix 0.2·sqrt(max(A, B)) 2.789 [Table 1].
- Moonlight vs Moonlight-A (AdamW), same data, 1.2T tokens, before cooldown: MMLU 60.4 vs 60.2, HumanEval 37.2 vs 29.3, MATH 19.8 vs 16.1, but BBH 43.2 vs 45.3 [Table 4].
- Optimizer mismatch at SFT: Muon-pretrained then AdamW-SFT scored MMLU 50.2 vs 52.0 for AdamW/AdamW, and Muon/Muon scored 55.7. Muon SFT of AdamW-pretrained Qwen2.5-7B gave no gain (GSM8K 85.8 vs 89.8 for Adam SFT) [Tables 6-7].
- Muon-trained matrices have higher SVD entropy than AdamW-trained ones in over 90% of matrices, most so for router weights [§3.4].

## Gotchas when reproducing

- Treat attention projections as full matrices, or apply the per-matrix scale to each piece you orthogonalize; per-head pieces with small max(A, B) get oversized updates otherwise [§2.2].
- Without weight decay Muon loses its lead in long runs [§2.2, Fig. 2].
- The 0.2 factor assumes AdamW-like update RMS; if your AdamW baseline differs, the LR transfer is approximate [§2.2].
- Keep the optimizer the same between pretraining and SFT where possible [§3.5].
- Muon is applied only to 2D hidden matrices; embeddings, LM head and norm gains stay on AdamW [§2.2].
- At Kimi K2 scale, Muon needed QK-clip to cap attention logits (already in the skill's stability file); Moonlight itself reports no such fix.
- Self-reported by the lab that uses Muon. The scaling study used AdamW-tuned hyperparameters for both optimizers, which if anything favours AdamW.

## Connections

- Stability and optimizer defaults: the skill's default is AdamW (0.9, 0.95), wd 0.1. Muon with RMS matching is a drop-in alternative that keeps those LR and wd values for transfer. Kimi Linear (MuonClip) and Qwen3.8-Next (Muon) both train with it (`kimi-linear.md`, `qwen38-next-design.md`).
- Qwen3.8-Next reports that its architecture plus Muon shift the optimal LR and batch size upward; this paper's "reuse AdamW's LR" is a transfer heuristic, not a claim that the optimum is the same.
- Weight decay: the skill notes PyTorch AdamW multiplies decay by LR. Moonlight's Eq. 3 does the same (η_t multiplies λ W).
- Muon halves optimizer-state memory relative to AdamW (one buffer), which would take the skill's 16-bytes-per-parameter budget to about 12 for Muon-managed matrices with fp32 state [inference: removes one 4-byte buffer; the paper gives no byte count].

## Disagreements and errors

- Moonlight vs DSV3-Small: §3.3 calls DeepSeek-V3-Small "a 2.4B/16B-parameter MoE", while Table 4 lists it as 2.24B active / 15.29B total (embedding-excluded, matching Moonlight). The two counting conventions are mixed in the text.
- The abstract says "~2x computational efficiency"; the body says "approximately 52% of the training FLOPs" (1.92x). Same result, rounded.
- Later work disputes simple LR transfer: "no principled scaling rule for Muon works consistently across training settings" [arXiv 2610.08975, abstract only], and Qwen3.8-Next reports a higher optimal LR with Muon (`qwen38-next-design.md`).
