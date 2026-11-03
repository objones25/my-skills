# DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models

Damai Dai, Chengqi Deng, Chenggang Zhao, R.X. Xu, Huazuo Gao, Deli Chen, Jiashi Li, Wangding Zeng, Xingkai Yu, Y. Wu, Zhenda Xie, Y.K. Li, Panpan Huang, Fuli Luo, Chong Ruan, Zhifang Sui, Wenfeng Liang (DeepSeek-AI; Peking University). arXiv:2401.06066 (v1, 11 Jan 2024, the only version). No venue on the arXiv page. Checked: full text of §2-7 from the arXiv HTML; appendices A-C not read.

## The idea

Two changes to a standard top-K MoE feed-forward layer, both at fixed total and active parameters [DSMoE §3]. Fine-grained segmentation: split each expert into m smaller experts (FFN hidden size divided by m) and route each token to m times as many, which makes far more expert combinations possible. Shared experts: K_s experts that every token always uses, with K_s fewer routed picks, so common knowledge does not get copied into every routed expert. At 2B total / 0.3B active and 100B tokens it matched GShard with 1.5x the expert parameters and compute, and came close to a dense model with 16x the FFN parameters [DSMoE Table 2]. The 16B model (2.8B active) matched DeepSeek 7B dense at 40.5% of the FLOPs, except on multiple-choice tasks [DSMoE Table 3, §5.2.1].

## Notation

| Symbol | Meaning |
|---|---|
| T | Sequence length (tokens in the batch for the balance loss) |
| d | Hidden size |
| u_t^l | Hidden state of token t after layer l's attention, shape (d,) [§2] |
| N | Number of experts in the coarse baseline; total expert parameters equal N standard FFNs |
| K | Experts activated per token in the coarse baseline |
| m | Segmentation factor: each expert becomes m experts of 1/m the FFN hidden size [§3.1] |
| K_s | Number of shared (always-on) experts [§3.2] |
| mN - K_s | Number of routed experts; mK - K_s of them are active per token [Eq. 9-10] |
| e_i^l | Learned centroid of expert i in layer l, shape (d,) |
| s_{i,t} | Token-to-expert affinity, softmax over experts of u_t^T e_i [Eq. 11] |
| g_{i,t} | Gate value: s_{i,t} if expert i is in the token's top (mK - K_s), else 0 [Eq. 10] |
| N', K' | Shorthand: N' = mN - K_s, K' = mK - K_s [§3.3] |
| f_i | Load of expert i: (N'/(K'T)) times the count of tokens that select it [Eq. 13] |
| P_i | Mean affinity of expert i over the T tokens [Eq. 14] |
| α_1, α_2 | Expert-level and device-level balance-loss weights [Eq. 12, 15] |
| D | Number of device groups for the device-level loss |

## The method

### 1. Fine-grained expert segmentation

Output h_t = Σ_{i=1}^{mN} g_{i,t} FFN_i(u_t) + u_t, with top-mK selection [Eq. 6-8]. Example from the paper: 16 experts, top-2 gives 120 combinations; splitting each into 4 and taking top-8 of 64 gives 4,426,165,368 [§3.1].

### 2. Shared expert isolation

h_t = Σ_{i=1}^{K_s} FFN_i(u_t) + Σ_{i=K_s+1}^{mN} g_{i,t} FFN_i(u_t) + u_t [Eq. 9]. Shared experts bypass the router and carry no gate. The idea is credited to Rajbhandari et al. (2022, DeepSpeed-MoE), arrived at there for engineering reasons [§3.2].

### 3. Load balance

Expert-level loss L_ExpBal = α_1 Σ_i f_i P_i guards against routing collapse [Eq. 12-14]. Device-level loss L_DevBal = α_2 Σ_d f'_d P'_d, where f'_d is the mean load and P'_d the summed affinity of the experts on device d, balances compute across devices without forcing every expert to be equal [Eq. 15-17]. The paper's rule: small α_1, larger α_2 [§3.3].

### 4. First layer stays dense

At 16B and 145B, all FFNs except the first layer are MoE, because the first layer's load balance converged slowly [§5.1.2, §7.1].

## Hyperparameters

| Setting | 2B validation | 16B | 145B (preliminary) | Source |
|---|---|---|---|---|
| Layers / hidden / heads (head dim 128) | 9 / 1280 / 10 | 28 / 2048 / 16 | 62 / 4096 / 32 | §4.1.3, §5.1.2, §7.1 |
| Attention | MHA | MHA | MHA | same |
| Experts (shared + routed), active | 1 + 63, 1 + 7 | 2 + 64, 2 + 6 | 4 + 128, 4 + 12 | §4.2, §5.1.2, §7.1 |
| Expert size vs standard FFN | 0.25 | 0.25 | 0.125 | same |
| Total / active params | 2.0B / 0.3B | 16.4B / 2.8B | 144.6B / 22.2B | same |
| Init std | 0.006 | 0.006 | 0.006 | same |
| Optimizer | AdamW (0.9, 0.95), wd 0.1, clip 1.0 | same | same | §4.1.3, §5.1.2, §7.1 |
| Peak LR | 1.08e-3 | 4.2e-4 | 3.0e-4 | same |
| Schedule | 2K warmup, ×0.316 at 80% and 90% of steps | same | 2K warmup, then constant | same |
| Batch | 2K seqs × 2K tokens = 4M tokens | 4.5K × 4K = 18M tokens | 18M tokens | same |
| Tokens | 100B (25,000 steps) | 2T (106,449 steps) | 245B (13,000 steps) | same |
| Vocab | 8K | 100K | 100K | §4.1.1, §5.1.1 |
| α_1 (expert balance) | 0.01 | 0.001 | 0.003 | §4.1.3, §5.1.2, §7.1 |
| α_2 (device balance) | not used | not used | 0.05 | same |
| Token dropping | none | none | not stated | §4.1.3, §5.1.2 |
| Dropout | none | none | none | same |

## Key results

- 2B / 100B tokens, equal total params: Pile loss Dense(0.2B) 2.060, Hash Layer 1.932, Switch 1.881, GShard (0.3B active) 1.867, DeepSeekMoE (0.3B active) 1.808 [Table 1].
- DeepSeekMoE 2B (1.808) equals GShard×1.5 (1.808, 1.5x expert params and 5.8T vs 4.3T FLOPs per 2K tokens) and nearly equals Dense×16 (1.806, 24.6T FLOPs) [Table 2].
- Shared-expert count barely matters at fixed active experts: 1, 2, 4 shared of 64 give Pile 1.808, 1.806, 1.811; the authors kept a 1:3 shared-to-routed-active ratio afterwards [§4.4].
- Disabling the shared expert at inference (and activating one more routed expert) raised Pile loss from 1.808 to 2.414 [§4.5].
- A from-scratch model with 1 shared + 3 of 63 routed active (half GShard's active expert params) still beat GShard [§4.5, Fig. 6; plot only].
- 16B vs DeepSeek 7B dense, same 2T tokens: TriviaQA 64.8 vs 59.7, NaturalQuestions 25.5 vs 22.2, but MMLU 45.0 vs 48.2 and CEval 40.6 vs 45.0. The authors blame the 16B model's 0.5B attention parameters vs 2.5B in the 7B [Table 3, §5.2.1].
- 145B at 245B tokens: Pile 1.876 vs DeepSeek 67B dense 1.905 and GShard 137B 1.961; MMLU 39.4 vs 45.1 (67B) [Table 6].

## Gotchas when reproducing

- The comparisons fix total and active parameters, not wall-clock time. Many small experts cost more in routing and kernel efficiency; the authors stopped at 0.25x expert size for 16B partly for this reason [§5.1.2].
- Multiple-choice scores (MMLU, CEval) lag a dense model of similar quality on LM loss. If those benchmarks matter, do not starve attention: at 16B attention was about 0.5B of 16.4B parameters [§5.2.1].
- Expert-level balance weight was kept tiny (0.001 at 16B) because a larger one hurt quality without helping throughput when all experts sat on one device [§5.1.2]. With expert parallelism, add the device-level term instead [§7.1].
- GShard 137B scores 26.3 on MMLU and 26.2 on CEval at 245B tokens, near chance for 4-way multiple choice, which makes it a weak baseline at that scale [Table 6].
- DeepSeek benchmarks its own models on an internal evaluation framework [§5.1.3]; the comparisons with LLaMA2 7B are not on identical data.

## Connections

- Sizing: MoE breaks the skill's dense parameter budget. Total parameters set memory; active parameters set FLOPs per token. The `d_ff = (2/3)·4·d_model` rule applies per expert only as a starting point; here experts are 1/4 or 1/8 of a standard FFN.
- Components: shared + fine-grained experts became the DeepSeek-V2/V3 FFN (see `deepseek-v2-mla.md`, `deepseek-v3.md`). Qwen's gated-attention MoE baseline uses the same fine-grained design with 128 experts, top-8 [GatedAttn §3.1].
- Stability: the balance loss is a second auxiliary loss next to z-loss; DeepSeek-V3 later replaces most of it with a bias term (see `deepseek-v3.md`). ST-MoE's router z-loss is the other standard router stabilizer (`st-moe.md`).

## Disagreements and errors

- §4.4 says "the ratio of 1:3 yields a marginally better Pile loss", but the best of the three tested values is 2 shared experts (1.806); with 8 active experts that is 2 shared + 6 routed, which is 1:3. The 2B model actually reported in Tables 1-2 uses 1 + 7, not the 1:3 ratio.
- Not an error, but easy to misread: "1.5 times both expert parameters and expert computation" for GShard×1.5 [§4.3] refers to experts only; total FLOPs per 2K tokens go from 4.3T to 5.8T (1.35x) because attention cost is unchanged [Table 2].
