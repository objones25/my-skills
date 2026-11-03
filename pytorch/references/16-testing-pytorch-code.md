# Testing PyTorch code

Tolerances, shape and gradient tests, overfit-one-batch, determinism, checkpoints,
train/eval mode, devices, custom ops, and property tests.

## Overview

Most bugs in model code do not crash. A softmax left on the head, a layer cut off from
the loss by a stray `.detach()`, or a tensor stored as a plain attribute instead of a
buffer all let training run and quietly learn less. The tests in this file are cheap
(milliseconds on CPU with a tiny model) and each one targets a bug of that kind.

This file is the PyTorch side: which torch APIs to test with and how they behave on
torch 2.15.0. The pytest side (fixtures, markers, the audit that catches tests that
cannot fail, mutation testing) lives in the pytest-expert skill; its
`pytest-expert/references/ml-testing.md` covers data-pipeline tests, behavioral tests, and what not
to unit test, which this file does not repeat.

Terms used below:

- **Tolerance**: how far two floats may differ and still count as equal. `atol`
  (absolute tolerance) is a fixed allowance; `rtol` (relative tolerance) scales with the
  expected value. `assert_close` passes element-wise when
  `|actual - expected| <= atol + rtol * |expected|`.
- **Fixture**: a pytest function whose return value is passed to any test that names it
  as a parameter.
- **Overfit one batch**: train on a single fixed batch until the loss is near zero. A
  model with enough capacity must be able to memorize a few examples; if it cannot, the
  model, loss, or step is wired wrong.

## Recommended approach

Start from the two assets instead of writing these by hand:

- `assets/test_model_template.py`: ten behavior tests (output shape, raw logits, every
  parameter gets a finite gradient, loss drops when overfitting one batch, same seed
  gives the same run, eval is deterministic, train is stochastic with dropout,
  `state_dict` round trip with `weights_only=True`, row independence in eval,
  `torch.compile` matches eager). Replace `make_model()`, `make_batch()` and `loss_fn()`
  at the top.
- `assets/conftest_torch.py`: copy it next to the template as `conftest.py`. It seeds
  every test, provides the `device` fixture (CPU, plus CUDA and MPS with skip reasons),
  an opt-in `deterministic` fixture, and `compile_ready`. Its docstring lists where it
  differs from pytest-expert's `conftest_ml.py`.

`tests/test_assets.py` in this skill runs the template against the placeholder model and
against six planted bugs, and checks that each bug fails the test meant to catch it.

Compare tensors with `torch.testing.assert_close`, never `==` and never `torch.allclose`
in a test. `assert_close` also checks shape, dtype, and device, and its failure message
names the worst element. `torch.allclose` broadcasts, so a `(3, 1)` tensor "matches" a
`(3,)` one (walkthrough step 2).

Run on every available device by requesting the `device` fixture; absent hardware is
reported as skipped with a reason, not silently passed:

```python
def test_forward_shape(device):
    model = make_model().to(device)
    x, _ = make_batch()
    assert model.eval()(x.to(device)).shape == (16, 3)
```

For a custom op registered with `torch.library.custom_op` (see
`15-custom-ops-autograd.md`), `opcheck` is the test. It checks the schema, the autograd
registration, the fake kernel, and tracing with dynamic shapes:

```python
def test_silu_np_registration_is_consistent():
    results = torch.library.opcheck(silu_np, (torch.randn(4, requires_grad=True),))
    assert set(results.values()) == {"SUCCESS"}
```

For shapes that should hold across many sizes, a property test with Hypothesis (a
library that generates inputs and shrinks a failure to the smallest one) covers more
than a hand-picked list. Keep it in its own file, since `importorskip` at module level
skips the whole module when Hypothesis is missing:

```python
import pytest
import torch
from torch import nn

hypothesis = pytest.importorskip("hypothesis", reason="property tests need the hypothesis package")
from hypothesis import given, settings, strategies as st

encoder = nn.TransformerEncoderLayer(d_model=8, nhead=2, dim_feedforward=16, batch_first=True).eval()

@settings(max_examples=25, deadline=None)
@given(batch=st.integers(min_value=1, max_value=4), length=st.integers(min_value=1, max_value=12))
def test_encoder_preserves_batch_and_length(batch, length):
    with torch.no_grad():
        out = encoder(torch.randn(batch, length, 8))
    assert out.shape == (batch, length, 8)
```

`deadline=None` turns off Hypothesis's per-example time limit, which the first call
into a torch kernel can exceed. This snippet was run on torch 2.15.0 with Hypothesis
6.170.14 (passed), and on the 2.14.1 environment without Hypothesis (skipped with the
reason).

## API reference

Introspected from torch 2.15.0 (identical on torch 2.14.1).

| API | Signature | Notes |
|-----|-----------|-------|
| `torch.testing.assert_close` | `assert_close(actual, expected, *, allow_subclasses=True, rtol=None, atol=None, equal_nan=False, check_device=True, check_dtype=True, check_layout=True, check_stride=False, msg=None)` | `rtol` and `atol` must be given together or not at all (`ValueError` otherwise); `None` means the dtype default below |
| `torch.allclose` | `allclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False) -> bool` | from the docstring (`inspect.signature` cannot read this builtin); broadcasts, no shape or dtype check |
| `torch.testing.make_tensor` | `make_tensor(*shape, dtype, device, low=None, high=None, requires_grad=False, noncontiguous=False, exclude_zero=False, memory_format=None)` | random test input; `noncontiguous=True` catches code that assumes contiguous memory |
| `torch.load` | `load(f, map_location=None, pickle_module=None, *, weights_only=None, mmap=None, **pickle_load_args)` | `weights_only=None` resolves to `True`: loading a pickled full model raised `UnpicklingError` with and without the argument |
| `Module.load_state_dict` | `load_state_dict(state_dict, strict=True, assign=False)` | returns `missing_keys` and `unexpected_keys` |
| `torch.use_deterministic_algorithms` | `use_deterministic_algorithms(mode, *, warn_only=False)` | read back with `are_deterministic_algorithms_enabled()` and `is_deterministic_algorithms_warn_only_enabled()` |
| `torch.autograd.gradcheck` | see `15-custom-ops-autograd.md` | float64 inputs |
| `torch.library.opcheck` | see `15-custom-ops-autograd.md` | returns `{test_name: 'SUCCESS' or exception}` |

Default `assert_close` tolerances, measured on torch 2.15.0 and 2.14.1 by provoking a
failure and reading the limits from the message (walkthrough step 1):

| dtype | `atol` | `rtol` |
|-------|--------|--------|
| `float64` | `1e-07` | `1e-07` |
| `float32` | `1e-05` | `1.3e-06` |
| `float16` | `1e-05` | `0.001` |
| `bfloat16` | `1e-05` | `0.016` |
| integer types | exact | exact |

The half-precision `atol` is the same `1e-05` as float32, far below the spacing between
neighbouring float16 values near 1 (about `1e-03`). Near-zero half-precision values
therefore need an explicit `atol`.

Which test catches which bug:

| Test | Catches |
|------|---------|
| output shape, batch size 1 and full | a hardcoded dimension, `squeeze()` without `dim` |
| raw logits | softmax or log_softmax on the head before `cross_entropy` |
| every parameter gets a finite gradient | `.detach()` or `torch.no_grad()` inside `forward`, unused layers, NaN from a bad op |
| loss drops when overfitting one batch | wrong loss, wrong target format, optimizer over the wrong parameters, missing `zero_grad` |
| same seed, same run | unseeded generators, nondeterministic kernels |
| eval deterministic, train stochastic | `F.dropout` without `training=self.training`, a forgotten `model.eval()` |
| `state_dict` round trip | a tensor stored as a plain attribute, a missing buffer |
| row independence in eval | batch statistics used at inference (BatchNorm left in train mode, a mean over dim 0) |
| compiled matches eager | ops that compile differently, graph breaks that change behavior |

## Walkthrough

A placeholder classifier (the same one as in `assets/test_model_template.py`) goes
through each check. Output executed on torch 2.15.0 and identical on torch 2.14.1.

**1. Measure `assert_close`'s default tolerances instead of recalling them.** A failed
comparison reports the limits it used ("up to ... allowed").

```python
import re
import torch

def default_tolerances(dtype):
    try:
        torch.testing.assert_close(torch.full((1,), 3, dtype=dtype),
                                   torch.full((1,), 2, dtype=dtype))
    except AssertionError as e:
        found = re.findall(r"up to (\S+) allowed", str(e))
    return found or ["exact", "exact"]     # integer dtypes report no tolerance

for dtype in (torch.float64, torch.float32, torch.float16, torch.bfloat16, torch.int64):
    atol, rtol = default_tolerances(dtype)
    print(f"{str(dtype):15s} atol={atol:8s} rtol={rtol}")
```
```
torch.float64   atol=1e-07    rtol=1e-07
torch.float32   atol=1e-05    rtol=1.3e-06
torch.float16   atol=1e-05    rtol=0.001
torch.bfloat16  atol=1e-05    rtol=0.016
torch.int64     atol=exact    rtol=exact
```

**2. `torch.allclose` broadcasts; `assert_close` does not.** A `(3, 1)` result compared
with a `(3,)` expectation is a shape bug that `allclose` hides.

```python
a, b = torch.ones(3, 1), torch.ones(3)
print("allclose:    ", torch.allclose(a, b))
try:
    torch.testing.assert_close(a, b)
except AssertionError as e:
    print("assert_close:", e)
```
```
allclose:     True
assert_close: The values for attribute 'shape' do not match: torch.Size([3, 1]) != torch.Size([3]).
```

**3. A placeholder model, its output shape, and raw logits.** Rows of raw logits do not
sum to 1; after a softmax they would.

```python
import torch.nn as nn
import torch.nn.functional as F

class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(8, 32), nn.BatchNorm1d(32), nn.ReLU(),
                                 nn.Dropout(p=0.2), nn.Linear(32, 3))
    def forward(self, x):                  # (B, 8) -> (B, 3)
        return self.net(x)

torch.manual_seed(0)
model = TinyClassifier()
x, y = torch.randn(16, 8), torch.randint(0, 3, (16,))
model.eval()
with torch.no_grad():
    out = model(x)
print("shape:", tuple(out.shape))
print("row sums:", torch.round(out.sum(dim=1)[:4], decimals=3))
```
```
shape: (16, 3)
row sums: tensor([0.7250, 0.6040, 0.3870, 0.3770])
```

**4. Gradient flow: list the parameters that got no finite gradient.** The list form
puts the culprit's name in the failure message. A stray `.detach()` after the first
layer cuts off its weight and bias.

```python
def params_without_finite_grad(model, x, y):
    model.zero_grad(set_to_none=True)
    model.train()
    F.cross_entropy(model(x), y).backward()
    return [n for n, p in model.named_parameters()
            if p.grad is None or not torch.isfinite(p.grad).all()]

class Detached(TinyClassifier):
    def forward(self, x):
        return self.net[1:](self.net[0](x).detach())

print("healthy: ", params_without_finite_grad(model, x, y))
print("detached:", params_without_finite_grad(Detached(), x, y))
```
```
healthy:  []
detached: ['net.0.weight', 'net.0.bias']
```

**5. Overfit one batch.** 100 Adam steps on the same 16 examples.

```python
def train(model, x, y, steps, lr=1e-2):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses

torch.manual_seed(0)
losses = train(TinyClassifier(), x, y, steps=100)
print(f"first loss {losses[0]:.3f} | last below 5% of first: {losses[-1] < 0.05 * losses[0]}")
```
```
first loss 1.141 | last below 5% of first: True
```

**6. Determinism: same seed, same losses; different seed, different losses.** The
seed covers initialization and every dropout mask.

```python
def run(seed):
    torch.manual_seed(seed)
    return train(TinyClassifier(), x, y, steps=5)

print("seed 0 twice equal:", run(0) == run(0))
print("seed 0 vs 1 equal: ", run(0) == run(1))
```
```
seed 0 twice equal: True
seed 0 vs 1 equal:  False
```

**7. `state_dict` round trip with `weights_only=True`.** One train-mode forward first,
so BatchNorm's running statistics (buffers, not parameters) differ from their initial
values and the round trip has to carry them.

```python
import io

torch.manual_seed(0)
model = TinyClassifier()
model.train()
with torch.no_grad():
    model(x)
buf = io.BytesIO()
torch.save(model.state_dict(), buf)
buf.seek(0)
fresh = TinyClassifier()
print(fresh.load_state_dict(torch.load(buf, weights_only=True)))
model.eval()
fresh.eval()
with torch.no_grad():
    print("identical outputs:", torch.equal(model(x), fresh(x)))

whole = io.BytesIO()
torch.save(nn.Linear(2, 2), whole)         # a whole module object: what not to do
whole.seek(0)
try:
    torch.load(whole, weights_only=True)
except Exception as e:
    print("loading a pickled model:", type(e).__name__)
```
```
<All keys matched successfully>
identical outputs: True
loading a pickled model: UnpicklingError
```

**8. Train versus eval mode.** Dropout makes train mode stochastic. BatchNorm in train
mode normalizes with the batch's own statistics, so row 0's output depends on the other
rows; in eval mode it uses the stored running statistics and does not. Turning only the
dropout submodule to eval isolates the BatchNorm effect.

```python
x_other = torch.cat([x[:1], torch.randn(15, 8)])   # row 0 kept, rows 1-15 replaced
with torch.no_grad():
    model.train()
    print("train: same input twice equal:", torch.equal(model(x), model(x)))
    model.eval()
    print("eval:  same input twice equal:", torch.equal(model(x), model(x)))

    model.train()
    model.net[3].eval()                    # dropout off, BatchNorm still in train mode
    print("train BN: row 0 ignores other rows:", torch.allclose(model(x)[0], model(x_other)[0]))
    model.eval()
    print("eval BN:  row 0 ignores other rows:", torch.allclose(model(x)[0], model(x_other)[0]))
```
```
train: same input twice equal: False
eval:  same input twice equal: True
train BN: row 0 ignores other rows: False
eval BN:  row 0 ignores other rows: True
```

## Gotchas

- **`torch.allclose` broadcasts and ignores dtype.** `(3, 1)` vs `(3,)` returns `True`.
  Mixed float32/float64 raises `RuntimeError: Float did not match Double` rather than
  reporting a dtype mismatch.
- **`assert_close` treats NaN as unequal by default.** Pass `equal_nan=True` only when
  NaN is the expected value; otherwise a NaN in the output is the bug.
- **`assert_close` wants both tolerances or neither.** `rtol=1e-3` alone raises
  `ValueError: Both 'rtol' and 'atol' must be either specified or omitted`.
- **Mixed dtypes fail before values are compared** (`The values for attribute 'dtype' do
  not match: torch.float32 != torch.float64`). Cast deliberately or pass
  `check_dtype=False` with a comment saying why.
- **A fixed tolerance across dtypes is a bug.** pytest-expert's `conftest_ml.py`
  `assert_close` fixture defaults to the float32 values (`rtol=1.3e-6, atol=1e-5`) for
  every dtype, which is looser than torch's float64 default and far tighter than its
  bfloat16 default (`rtol=0.016`). Leave `rtol`/`atol` unset to get the dtype-aware
  defaults, or state them per test.
- **BatchNorm cannot run a batch of 1 in train mode**: `ValueError: Expected more than 1
  value per channel when training, got input size torch.Size([1, 32])`. Test batch size
  1 in eval mode, which is how inference runs.
- **A train-mode forward updates BatchNorm running statistics even under
  `torch.no_grad()`.** A test that runs train mode before an eval comparison changes the
  model it then checks.
- **`gradcheck` needs float64**; on float32 it warns and fails a correct backward (see
  `15-custom-ops-autograd.md`).
- **Do not assert exact loss values.** Assert direction and size of change, as in step 5.
  Exact values move with torch versions, BLAS libraries, and CPU instruction sets.
- **Determinism holds on one machine and one build.** CPU and GPU results differ even
  with the same seed, and PyTorch does not promise identical results across releases.
  The `deterministic` fixture only makes repeated runs on one setup agree.
- **`torch.compile` changes rounding.** Compare compiled to eager with a stated
  tolerance (the template uses `rtol=1e-4, atol=1e-5`), not exact equality. The first
  compile on CPU took about 18 s here, so the template marks that test `slow`.
- **`importorskip` at module level skips everything in the file.** Put Hypothesis tests
  in their own module.
- **Hypothesis and torch's RNG are separate.** Hypothesis controls which sizes it
  generates; tensor contents still come from torch's generator, seeded once per test
  function by the autouse fixture, not once per generated example.

## Alternatives

| Need | Use | Instead of |
|------|-----|------------|
| float comparison in a test | `torch.testing.assert_close` | `torch.allclose`, `==` |
| scalar float comparison | `pytest.approx` or `assert_close` on Python floats | `==` |
| random test tensors with controlled range or layout | `torch.testing.make_tensor` | hand-rolled `torch.rand` plus casts |
| shape properties over many sizes | Hypothesis `@given` | a long `parametrize` list |
| custom autograd | `gradcheck`, `gradgradcheck` | comparing to a hand-derived number |
| registered custom op | `torch.library.opcheck` | testing only eager mode |
| model quality | a scheduled benchmark (pytest-expert: `integration` marker) | a unit test with an accuracy threshold |
| data loaders, transforms, splits | the pytest-expert skill, `pytest-expert/references/ml-testing.md` | (not covered here) |

## Related

- `15-custom-ops-autograd.md` - what `gradcheck` and `opcheck` check, and why.
- `06-tuning-reproducibility.md` - seeding and deterministic algorithms in training runs.
- `10-debugging.md` - the overfit-one-batch protocol when a model will not learn.
- `05-data-training-loop.md` - checkpoints, which the round-trip test protects.
- `11-compile-export.md` - why compiled output can differ from eager.

## Sources

- `inspect.signature` against torch 2.15.0 and 2.14.1 (identical): `assert_close`,
  `make_tensor`, `torch.load`, `Module.load_state_dict`,
  `use_deterministic_algorithms`. `torch.allclose` defaults read from its docstring on
  both versions.
- Default tolerances measured on both versions by provoking failures (walkthrough
  step 1); they also match the private `torch.testing._comparison.default_tolerances`
  helper on both.
- Every error message quoted under Gotchas was produced by running the code on both
  versions. The compile timing is one CPU run of `assets/test_model_template.py`'s model
  on torch 2.15.0.
- Docs pages (each URL checked to resolve; the claims above come from introspection
  and runs, not from these pages):
  docs.pytorch.org/docs/2.15 - `testing.html`, `generated/torch.allclose.html`,
  `generated/torch.load.html`, `notes/randomness.html`, `library.html`.
- pytest-expert skill - `pytest-expert/references/ml-testing.md` and `pytest-expert/assets/conftest_ml.py`, read
  for consistency; hard gates followed by the template, and its
  `pytest-expert/scripts/audit_tests.py` reports "clean: no findings" on `assets/`.
