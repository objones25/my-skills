# Worked examples

Each file is one question answered end to end: the verdict, the evidence that decides it,
the script run that produced the evidence, and what would change the answer. Script output
is from real runs (torch 2.15.0 CPU build, Python 3.13, 2026-11-02, plus torch 2.14.1 for
the upgrade example). The scenarios and people are invented; the models in `models/` and
`buggy_train.py` are small planted-bug programs written for these examples. Never present
an example's scenario as a fact about the user's code.

| File | Question | Scripts | References |
|---|---|---|---|
| `01-model-wont-learn.md` | "Loss goes down a bit, then stalls. What is wrong?" | `diagnose.py` | `10-debugging.md` |
| `02-nan-loss.md` | "Loss is NaN from the first step." | `diagnose.py nan` | `10-debugging.md` |
| `03-does-it-fit.md` | "Can I train a 7B model on 8 x 80 GB, and with DDP or FSDP2?" | `memory_budget.py` | `12-distributed.md` |
| `04-upgrading-to-2-15.md` | "We are on 2.14.1. What breaks if we move to 2.15?" | `verify_api.py` | `00-module-map.md` |
| `05-reviewing-a-training-script.md` | "Review this training script before I run it overnight." | `lint_pytorch.py`, `diagnose.py` | `05-data-training-loop.md` |
| `06-int8-on-cpu.md` | "Quantize this for CPU inference." | torchao walkthrough | `14-quantization.md` |

Planted-bug programs, used by the examples and by `tests/test_diagnose.py`:

| File | Bug | Caught by |
|---|---|---|
| `models/healthy.py` | none (the control) | passes every check |
| `models/plain_list.py` | layers in a plain list | `diagnose.py grads` (HIDDEN), `lint_pytorch.py` (PLAIN_LIST_OF_MODULES) |
| `models/detached.py` | stray `.detach()` | `diagnose.py grads` (NONE flags) only; invisible to the linter |
| `models/softmax_head.py` | softmax before CrossEntropyLoss | `diagnose.py grads` (WARNING), `lint_pytorch.py` (SOFTMAX_HEAD) |
| `models/nan_log.py` | `log` of a negative value | `diagnose.py nan` |
| `buggy_train.py` | eight bugs in one script | `lint_pytorch.py` finds seven; the eighth needs `diagnose.py` |

The shape every answer keeps: the verdict first, then the output that proves it, then what
to change and what would change the verdict.
