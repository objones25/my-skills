# Sources and how each was checked

Every API fact in this skill comes from one of four places, in this order of trust:

1. **The installed library**, read with Python's `inspect` (`inspect.signature`, and the
   source for behavior such as Muon's `_adjust_lr`), or reproduced by running it. Versions:
   torch 2.15.0 and 2.14.1 (CPU builds), torchvision 0.29.2 and 0.29.1, torchao 0.18.0,
   Python 3.13. `scripts/verify_api.py` re-runs the pinned part of this on any install, and
   `--docs` re-checks every name and signature claim in these references.
2. **Release notes** for behavior that needs hardware not available here (multi-GPU NCCL,
   CUDA kernels, MPS). Marked "not run here" where used.
3. **Current documentation**, read through Context7 (docs.pytorch.org 2.14 pages,
   github.com/pytorch/ao docs).
4. **Papers and vendor posts**, for the few claims that are not about an API.

When the docs and the installed code disagree, the skill follows the code and says so.

## Release information

| Source | Read through | Supports |
|---|---|---|
| github.com/pytorch/pytorch/releases/tag/v2.15.0 | WebFetch (truncated in distributed bug fixes) and Firecrawl (direct quotes), 2026-11-02 | every "2.15" claim in `00-module-map.md`, `12-distributed.md`, `04`, `09`, `11`, `13` |
| dev-discuss.pytorch.org/t/pytorch-release-2-15-key-dates/3434 | WebFetch | 2.15 release date, 2026-10-28; Stable/Unstable labels |
| discuss.pytorch.kr/t/pytorch-2-15/12351 (PyTorchKR summary of the 2.15 blog) | WebFetch | corroborates the release-notes items; the official blog page itself could not be fetched (the Firecrawl request timed out) |
| PyTorch 2.14 release blog (Sept 2, 2026) | read by the previous version of this skill | 2.14 rows in `00-module-map.md`; the `dynamic_spec` and `torch.accelerator` RNG locations were then corrected by import on 2.14.1 and 2.15.0 |
| PyTorch 2.13 release blog (July 8, 2026) | read by the previous version | 2.13 rows |
| CPython gh-84559 (GH-101556), "Change the multiprocessing start method default to forkserver" | web search result | the Python 3.14 DataLoader worker note in `05-data-training-loop.md` |

## Documentation pages (Context7)

| Page | Supports | Disagreement with the installed code |
|---|---|---|
| docs.pytorch.org/docs/2.14/checkpoint.html | `checkpoint` keyword list, curried form | docs: "In version 2.9 we will raise an exception if `use_reentrant` is not passed"; torch 2.15.0 only warns (`tests/test_claims.py`) |
| docs.pytorch.org/docs/2.14/generated/torch.optim.Muon.html | Muon algorithm, `adjust_lr_fn` options | one rendered formula for `match_rms_adamw` drops the square root; the source (`torch/optim/_muon.py`) has `0.2 * math.sqrt(max(A, B))`, which the skill quotes |
| github.com/pytorch/ao docs (inference workflow table, first quantization example, int4 by device) | `14-quantization.md` config names and device notes | none found; the CPU int4 `ImportError` is not in the docs and was reproduced |

## Papers and posts

| Source | Supports | How it was checked |
|---|---|---|
| Hu et al. 2021, "LoRA," arXiv:2106.09685 | LoRA parameter count and low-rank argument (`07`) | parameter arithmetic recomputed in the walkthrough |
| Biderman et al. 2024, "LoRA Learns Less and Forgets Less," TMLR, arXiv:2405.09673 | LoRA forgetting claim (`07`) | carried from the previous version, not re-read |
| Schulman et al. 2025, "LoRA Without Regret," thinkingmachines.ai (vendor blog) | ~10x LR for LoRA (`07`) | carried; labeled vendor, not peer reviewed |
| Goyal et al. 2017, arXiv:1706.02677 | linear LR scaling for SGD (`06`) | carried |
| Malladi et al. 2022, NeurIPS, arXiv:2205.10287 | square-root LR scaling for Adam (`06`) | carried |
| Bergstra and Bengio 2012, JMLR | random beats grid search (`06`) | carried |
| Su et al. 2021, "RoFormer," arXiv:2104.09864 | RoPE relative-position property (`09`) | carried |
| Liang et al. 2024, "TorchTitan," arXiv:2410.06511 | per-block compile, checkpointing and FSDP order (`11`) | carried |

"Carried" means the claim was checked by the previous version of this skill and was not
re-read in this audit. None of these claims is about an API that changed between 2.13 and
2.15.

## What was run, by file

| File | Executed on 2.15.0 | Executed on 2.14.1 |
|---|---|---|
| every `## Walkthrough` in `references/` | yes (`tests/test_walkthroughs.py`) | yes |
| `assets/train_template.py` | yes, including bit-exact resume | yes |
| `assets/ddp_template.py` | yes, 2 gloo ranks, DDP and FSDP2 | yes |
| `assets/test_model_template.py` | yes | yes |
| `scripts/*.py` | yes (one test file each) | yes |
| torchao walkthrough (`14`) | yes, torchao 0.18.0 | skipped (torchao not installed there) |

## Not verified

- Anything needing CUDA, MPS, or multiple GPUs: the rewritten NCCL backend, FSDP2's 2.15
  `reduce_dtype` change, FlashAttention backend selection, fp16 `GradScaler` on MPS, int4
  CUDA kernels, CUDA graph capture. These are from release notes or docs and say so.
- Speedup and memory-saving figures quoted from release blogs (FlexAttention on MPS,
  `LinearCrossEntropyLoss`'s ~4x) were not measured.
- The official 2.15 blog post on pytorch.org could not be fetched; the GitHub release notes
  and the PyTorchKR summary were used instead.
