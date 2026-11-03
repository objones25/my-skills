# Reviewing a training script before an overnight run

*Illustrative scenario; the script output is real (torch 2.15.0).*

**Question.** "Can you review `examples/buggy_train.py` before I leave it running overnight?
It runs fine on a small test."

## Verdict

Do not run it. It has eight bugs, none of which raises; together they guarantee a run that
looks normal and learns little. Seven are found statically in a second; the eighth needs
the model to run.

## Evidence

```
$ python scripts/lint_pytorch.py examples/buggy_train.py
examples/buggy_train.py:15: PLAIN_LIST_OF_MODULES [error]: self.blocks holds modules in a plain list comprehension; ...
examples/buggy_train.py:17: UNREGISTERED_TENSOR [warning]: self.temperature = torch.ones(...) is a plain attribute; ...
examples/buggy_train.py:21: F_DROPOUT_NO_TRAINING [error]: F.dropout defaults to training=True and ignores model.eval(); ...
examples/buggy_train.py:30: SET_OF_PARAMS [warning]: optimizer params given as a set, whose order changes between runs; ...
examples/buggy_train.py:35: DISCARDED_TO [error]: xb.to(...) result is thrown away; tensors move by copy, so reassign: ...
examples/buggy_train.py:39: MISSING_ZERO_GRAD [warning]: backward() and optimizer step() with no zero_grad() in this function; ...
examples/buggy_train.py:40: LOSS_TENSOR_ACCUMULATED [error]: loss accumulated into history as a tensor; use loss.item() ...
examples/buggy_train.py:42: EVAL_NO_GRAD_GUARD [warning]: model(...) runs after model.eval() with autograd on; ...

8 finding(s): ...
```

Seven of the eight lines marked `BUG` are found. The eighth finding (`EVAL_NO_GRAD_GUARD`)
was not planted, and it is real: the final evaluation builds an autograd graph for the whole
dataset.

The missed bug is `forward` returning `F.softmax(...)` into `CrossEntropyLoss`. The linter
traces softmax into a loss only within one function, and here the softmax is inside
`forward`. Running the model catches it. The same bug in `examples/models/softmax_head.py`:

```
$ python scripts/diagnose.py grads examples/models/softmax_head.py
...
WARNING model output rows are non-negative and sum to 1: the head applies softmax, but the loss expects raw logits (it applies its own)
```

Its overfit test plateaus at 0.7466. That floor can be checked by hand: with softmax
outputs `p` fed to `CrossEntropyLoss`, the best the model can do is `p = (1, 0, 0, 0)`, which
the loss's own log-softmax turns into `-log(e / (e + 3)) = log(1 + 3/e) = 0.7437`. With
hard labels and no label smoothing, a loss that cannot go below 0.74 on four classes points
at a double softmax (heavy label smoothing or the same input appearing with different
labels can also put a floor near there).

## The fixes, in the order they matter

1. `nn.ModuleList` for `self.blocks` (otherwise two of three layers never train).
2. Return raw logits from `forward`.
3. `optimizer.zero_grad()` each step (otherwise every step adds all previous gradients).
4. `xb = xb.to(device)` (harmless on CPU, a device-mismatch crash on GPU).
5. `F.dropout(..., training=self.training)` or `nn.Dropout`.
6. `history.append(loss.item())` (otherwise memory grows every step until OOM overnight).
7. `register_buffer("temperature", ...)` or `nn.Parameter`.
8. A list of parameters, and `torch.inference_mode()` around evaluation.

## What would change the answer

Nothing about the verdict: each bug is independently enough to waste the night. If the
script had passed both tools, the next check is `diagnose.py overfit` on a real batch.
