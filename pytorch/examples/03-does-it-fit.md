# Does a 7B model train on 8 x 80 GB, and with which wrapper?

*Illustrative scenario; the script output is real.*

**Question.** "We have one node with eight 80 GB GPUs. Can we full-fine-tune a 7B model with
AdamW, and do we need FSDP or will DDP do?"

## Verdict

FSDP2. With DDP every GPU holds the full weights, gradients and Adam state, 104 GiB before
any activation, which does not fit 80 GiB. Sharded across 8 ranks it is 13 GiB per GPU,
leaving most of each card for activations.

## Evidence

```
$ python scripts/memory_budget.py --params 7e9 --optimizer adamw
params 7,000,000,000 | adamw | fp32 | ddp x 1
term        bytes/param  GiB per rank
weights               4         26.08
grads                 4         26.08
master                0          0.00
optimizer             8         52.15
total                16        104.31
does NOT fit in 80 GiB before activations

$ python scripts/memory_budget.py --params 7e9 --optimizer adamw --world 8
params 7,000,000,000 | adamw | fp32 | fsdp x 8
...
total                16         13.04
fits in 80 GiB before activations
```

Check by hand: 16 bytes per parameter (4 weight, 4 gradient, 8 for Adam's two moments)
times 7e9 is 1.12e11 bytes, and 1.12e11 / 2^30 = 104.3 GiB. bf16 autocast does not change
this, because autocast casts on the fly and the parameters stay fp32.

The 16 bytes/param figure is not taken on faith. On a real model the script takes one
optimizer step and sums what PyTorch allocated:

```
$ python scripts/memory_budget.py --model examples/models/healthy.py --optimizer adamw --measure
...
optimizer state, one real CPU step in fp32: measured 10,784 bytes, formula 10,784 bytes (match)
```

(`examples/models/healthy.py` has 1,348 parameters; 1,348 x 8 = 10,784.)

## What would change the answer

- **Activations.** Not counted. At long sequence lengths they can exceed the 67 GiB left per
  GPU; measure with `torch.accelerator.max_memory_allocated()` after one step, and add
  activation checkpointing if needed (`references/13-performance-profiling.md`).
- **LoRA instead of full fine-tuning.** Only the adapters get gradients and Adam state; the
  frozen base needs weights only, and DDP becomes viable.
- **A leaner optimizer.** SGD with no momentum in pure bf16 is 4 bytes/param, 26 GiB for 7B,
  which fits one GPU (`--optimizer sgd --precision bf16`), at a cost in training quality
  that has to be measured.
- **Interconnect.** FSDP2 all-gathers each layer every step; on a slow interconnect the
  communication, not memory, becomes the limit (`references/12-distributed.md`).
