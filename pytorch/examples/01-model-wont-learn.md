# The model learns a little, then stalls

*Illustrative scenario; the script output is real (torch 2.15.0, CPU).*

**Question.** "My classifier's loss drops from 1.40 to about 1.32 and then barely moves,
whatever learning rate I try. The code runs without errors." The model is
`examples/models/plain_list.py`.

## Verdict

Three of the four layers never train: they are kept in a plain Python list, so
`model.parameters()` does not contain them and the optimizer never sees them. Only the
head learns. Fix: `self.hidden = nn.ModuleList([...])`.

## Evidence

Overfit one batch first (`references/10-debugging.md`). A correct model, loss and step
drive one batch's loss toward zero, so a failure here rules out data, schedule and
generalization in one run:

```
$ python scripts/diagnose.py overfit examples/models/plain_list.py
== overfit
verdict: stuck (loss 1.4121 -> 1.3212 in 200 steps; constant-prediction baseline 1.3750; no better than predicting one constant for every example, ...)
curve: 1.4121 1.3860 1.3700 1.3600 1.3530 1.3471 1.3415 1.3362 1.3310 1.3259 1.3212
```

The constant-prediction baseline is the loss of one learned output shared by every
example, which ignores the inputs entirely. This model does no better, so whatever it
learned (the class frequencies) did not use the inputs. The control model with the same
data reaches 0.0081 in the same 200 steps
(`python scripts/diagnose.py overfit examples/models/healthy.py`). So the bug is in the model,
loss or step. The gradient table names it:

```
$ python scripts/diagnose.py grads examples/models/plain_list.py
parameter                                shape              grad norm  grad/weight  flag
head.weight                              (4, 16)            7.732e-02    7.887e-02  ok
head.bias                                (4,)               1.380e-01    3.547e-01  ok
HIDDEN hidden: modules in a plain list (their parameters are invisible to the optimizer; use nn.ModuleList)
```

Only two parameters are listed for a four-layer network. The linter finds the same bug
without running anything:

```
$ python scripts/lint_pytorch.py examples/models/plain_list.py
examples/models/plain_list.py:15: PLAIN_LIST_OF_MODULES [error]: self.hidden holds modules in a plain list comprehension; ...
```

## The same symptom, a different cause

`examples/models/detached.py` stalls the same way (1.4016 to 0.7646), but every layer is
registered. The grads table shows the encoder's gradients are `None`:

```
encoder.0.weight                         (32, 16)                   -            -  NONE
encoder.0.bias                           (32,)                      -            -  NONE
head.weight                              (4, 32)            2.527e-01    2.039e-01  ok
```

`NONE` means the parameter never reached the loss: a `.detach()`, a `torch.no_grad()` block,
or a branch the forward never takes. The linter cannot see this one; only running the model
can. That is why the debugging protocol runs the overfit test before reading code.

## What would change the answer

If overfit had reported `memorized`, the model would be fine and the problem would be
downstream: labels misaligned with inputs, too-strong augmentation, or an LR wrong for the
full data. If grads had shown `ZERO` rather than `NONE`, the parameters are connected but
the gradient vanished (dead ReLUs, saturated activations).
