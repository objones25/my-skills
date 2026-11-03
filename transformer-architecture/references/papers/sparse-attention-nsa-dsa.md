# Trainable sparse attention: NSA and DeepSeek Sparse Attention (DSA)

Two DeepSeek papers, digested together because DSA is the production successor of NSA's idea.

1. Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention. Jingyang Yuan, Huazuo Gao, Damai Dai, Junyu Luo, Liang Zhao, Zhengyan Zhang, Zhenda Xie, Y.X. Wei, Lean Wang, Zhiping Xiao, Yuqing Wang, Chong Ruan, Ming Zhang, Wenfeng Liang, Wangding Zeng (DeepSeek-AI, Peking University, University of Washington). arXiv:2502.11089 (v2, 27 Feb 2025; v1 16 Feb 2025). No venue on the arXiv page. Checked: full text of §2-6 from the arXiv HTML.
2. DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models. DeepSeek-AI. arXiv:2512.02556 (v1, 2 Dec 2025). Technical report. Checked: full text of §1-3.1 from the arXiv HTML; the agentic sections (§3.2-4) and Appendix A not read.

## The idea

Both replace each query's attention over all previous tokens with attention over a small, query-dependent subset, and both train the sparsity in rather than applying it only at inference.

- NSA runs three attention branches per query and mixes them with learned sigmoid gates: attention over compressed block summaries (an MLP squeezes each block of 32 keys into one), attention over the original tokens of the top-16 blocks (scored for free from the compressed branch's attention weights), and a 512-token sliding window [NSA §3.2-3.3]. At 27B total / 3B active and 270B tokens it matched or beat full attention on general benchmarks (average 0.456 vs 0.443) and LongBench (0.469 vs 0.437), with up to 9.0x forward / 6.0x backward kernel speedup and an expected 11.6x decode memory-access reduction at 64K [Tables 1-2, 4; §5.1].
- DSA keeps one branch: a cheap "lightning indexer" (few heads, ReLU, FP8) scores every previous token, and the main MLA attention reads only the top-2048 tokens' latent entries [DSV3.2 §2.1]. It was added to an existing 128K DeepSeek-V3.1 checkpoint by continued training: 2.1B tokens to warm up the indexer against dense attention, then 943.7B tokens of sparse training [§2.1.1]. DeepSeek reports no substantial regression on short or long context and lower serving cost at long positions [§2.2-2.3].

## Notation

| Symbol | Meaning |
|---|---|
| q_t, k_i, v_i | Query at position t; key and value at position i |
| C = {cmp, slc, win} | NSA's three branches: compression, selection, sliding window |
| g_t^c | Gate for branch c, in [0, 1], from an MLP and sigmoid on the input [NSA Eq. 5] |
| l, d | Compression block length (32) and stride (16) |
| φ | MLP (with intra-block position encoding) mapping a block of keys to one compressed key [Eq. 7] |
| l' | Selection block size (64) |
| n | Number of selected blocks (16, including 1 initial and 2 local blocks always kept) |
| w | Sliding window (512) |
| p_t^cmp, p_t^slc | Attention over compressed keys; block importance for selection derived from it [Eq. 8-9] |
| N_t | Total keys a query attends to in NSA [Eq. 6] |
| h_t | DSA query token's hidden state, shape (d,) |
| H^I, d^I | Number and dimension of indexer heads (values not given in the sections read) |
| q^I_{t,j}, w^I_{t,j} | Indexer query and per-head weight for head j, from h_t |
| k^I_s | Indexer key for earlier token s (shared across indexer heads) |
| I_{t,s} | Index score, Σ_j w^I_{t,j} ReLU(q^I_{t,j} · k^I_s) [DSV3.2 Eq. 1] |
| k (DSA) | Selected tokens per query (2048) |
| c_s | MLA latent KV entry of token s |
| p_{t,:} | Target distribution: main attention summed over heads, L1-normalized [Eq. 3] |

## The method

### NSA

1. Compression: K̃^cmp = {φ(k_{id+1 : id+l})} for overlapping blocks (d < l) [Eq. 7].
2. Selection: reuse the compressed-branch softmax p_t^cmp as block importance, sum over the query heads of a GQA group so all heads pick the same blocks, take the top-n blocks of size l' and attend to their raw tokens [Eq. 8-12].
3. Window: separate keys/values for the last w tokens, so local patterns do not "shortcut" the other branches [§3.3.3]. Each branch has its own keys and values.
4. Combine: o_t = Σ_c g_t^c Attn(q_t, K̃^c, Ṽ^c) [Eq. 5].
5. Kernel: for each query position, load all heads of one GQA group and their shared sparse KV blocks (group-centric loading), Triton implementation [§3.4]. Designed for GQA/MQA; MHA is called inefficient for decoding.

### DSA

1. Indexer: I_{t,s} = Σ_j w^I_{t,j} ReLU(q^I_{t,j} · k^I_s) [Eq. 1].
2. Select top-k (2048) tokens per query; run MLA (in its MQA mode, latent shared across all query heads) over only those entries [Eq. 2, §2.1].
3. Warm-up: freeze everything but the indexer, keep dense attention, train the indexer with KL(p_{t,:} || softmax(I_{t,:})) [Eq. 3].
4. Sparse stage: train all parameters with sparse attention; indexer still trained by KL on the selected set only, with its input detached so the LM loss trains the model and the KL trains the indexer [Eq. 4, §2.1.1].
5. Post-training (SFT distillation and RL) also uses sparse attention [§3].

## Hyperparameters

| Setting | NSA | DSA | Source |
|---|---|---|---|
| Backbone | 27B total / 3B active, 30 layers, width 2560, GQA 4 groups, 64 heads, d_q = d_k = 192, d_v = 128, DeepSeekMoE 72 routed + 2 shared, top-6, first layer dense | DeepSeek-V3.1-Terminus (MLA + MoE, 671B-class) continued | NSA §4.1; DSV3.2 §2.1 |
| Blocks / budget | l = 32, d = 16, l' = 64, n = 16, w = 512 | top-k = 2048 tokens | NSA §4.1; DSV3.2 §2.1.1 |
| Training | 270B tokens at 8K, then continued training and SFT at 32K with YaRN | warm-up: 1000 steps × 16 × 128K = 2.1B tokens at LR 1e-3; sparse: 15,000 steps × 480 × 128K = 943.7B tokens at LR 7.3e-6 | NSA §4.1; DSV3.2 §2.1.1 |
| Indexer heads / dims | n/a | not given in the sections read (open-source code is referenced as the specification) | DSV3.2 §2.1 |

## Key results

- NSA vs full attention, 27B MoE pretrained: average 0.456 vs 0.443; GSM8K 0.520 vs 0.486; DROP 0.545 vs 0.503; lower on MMLU (0.565 vs 0.567) and MBPP (0.466 vs 0.482) [NSA Table 1].
- LongBench (selected subsets), average: NSA 0.469, full attention 0.437, Exact-Top 0.423, Quest 0.392, H2O 0.303, with inference-only baselines given the same 2560-token budget [Table 2].
- AIME 24 after 10B tokens of R1-distilled SFT at 32K: NSA-R 0.121 vs 0.046 (8K generation), 0.146 vs 0.092 (16K) [Table 3].
- Decode memory access at 64K: 5,632 token-equivalents vs 65,536, expected 11.6x [Table 4]. Kernel speedup at 64K: 9.0x forward, 6.0x backward vs Triton FlashAttention-2 [§5.1].
- Learned (auxiliary-loss) and heuristic (Quest-style) block selection both trained to higher loss than NSA on a 3B model [§6.1; plot only].
- DSA: attention cost O(L k) for the main model, the indexer still O(L²) but cheap [§2.3]; parity with V3.1-Terminus claimed on benchmarks and ChatbotArena Elo; AA-LCR 4 points higher in reasoning mode [§2.2]. The parity tables are not in the text read; cost curves are a figure based on H800 serving at 2 USD per GPU hour [Fig. 3].

## Gotchas when reproducing

- NSA needs GQA/MQA so heads in a group share the selected blocks; DSA runs MLA in MQA mode for the same reason [NSA §3.4, DSV3.2 §2.1].
- NSA gives each branch its own keys and values; sharing them invited shortcut learning [§3.3.3].
- DSA's indexer is trained only by its KL loss (input detached). Training it through the LM loss is not what the paper does [§2.1.1].
- DSA was retrofitted at 128K. Prefill at short lengths uses a masked dense MHA mode because it is faster there [§2.3].
- Both are DeepSeek self-reports. NSA's full-attention baseline is a 27B MoE trained 270B tokens, small relative to production. DSA's evidence for "no regression" is mostly qualitative in the text.

## Connections

- Long context and KV cache: sparse attention does not shrink the KV cache (all entries are kept; only reads are sparse). It cuts compute and decode bandwidth. Hybrids with linear attention shrink the cache itself (`kimi-linear.md`). Qwen3.8-Next combines both: GDN hybrid plus "Qwen Sparse Attention" in the full-attention layers (`qwen38-next-design.md`).
- Components: NSA's per-branch sigmoid gates are another case of gating the attention output (`gated-attention.md`).
- Post-training: DeepSeek-V3.2's RL stabilizers (unbiased KL, off-policy masking, keep routing, keep sampling mask) are covered in `rl-post-training.md`.

## Disagreements and errors

- NSA §6.1 says "As shown in Figure 8, both methods exhibited inferior loss", but Figure 8 is the attention-map visualization; the loss comparison is Figure 7.
- NSA's LongBench table excludes subsets "due to their low scores across all models" [§4.3]; the average is over a chosen subset.
