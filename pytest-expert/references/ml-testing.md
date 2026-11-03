# Testing ML code

Testing a model is not testing the model. Almost everything worth testing in an ML codebase is
ordinary software (shapes, dtypes, devices, data transforms, checkpoint round-trips, the
training step) and it is testable with the same rules as any other code. Model *quality* is a
benchmark, not a unit test, and it belongs behind a marker.

Verified against PyTorch 2.15 and pytest 9.2.0. Setup fixtures are in `assets/conftest_ml.py`.

## Contents

- [The layers](#the-layers)
- [Determinism first](#determinism-first)
- [Shape and dtype contracts](#shape-and-dtype-contracts)
- [Comparing tensors](#comparing-tensors)
- [Gradient tests](#gradient-tests)
- [The training step](#the-training-step)
- [Overfit one batch](#overfit-one-batch)
- [Data pipeline tests](#data-pipeline-tests)
- [Checkpoints](#checkpoints)
- [Device and GPU markers](#device-and-gpu-markers)
- [Behavioral tests](#behavioral-tests)
- [What not to unit test](#what-not-to-unit-test)

## The layers

| Layer | Runs in | Marker | Asserts |
|---|---|---|---|
| **Plumbing** | milliseconds | none (default suite) | Shapes, dtypes, devices, gradient flow, no NaN |
| **Step** | milliseconds | none | One optimizer step changes parameters and lowers loss on a fixed batch |
| **Overfit** | seconds | `slow` | The model can drive loss to ~0 on a batch of 4 |
| **Pipeline** | seconds–minutes | `slow` / `integration` | End to end: data loads, trains, checkpoints, reloads |
| **Quality** | minutes–hours | `integration` + scheduled | Metric on a held-out set clears a threshold |

Everything in the first two rows runs on every commit. The rest run on a schedule or before a
release. Getting this split right is what keeps a test suite usable: an ML suite that takes
twenty minutes gets skipped, and a skipped suite is not a suite.

## Determinism first

Nothing else works without it. A flaky ML test is usually four separate problems:

```python
import os, random
import numpy as np
import torch

# 1. Python's RNG (used by shuffling, augmentation, dropout in some libraries)
random.seed(SEED)

# 2. NumPy's global RNG (many datasets and transforms use it)
np.random.seed(SEED)

# 3. Torch, CPU and all CUDA devices at once
torch.manual_seed(SEED)

# 4. Nondeterministic kernels
torch.backends.cudnn.benchmark = False       # stop re-picking a conv algorithm per run
torch.backends.cudnn.deterministic = True    # force the deterministic conv
torch.use_deterministic_algorithms(True, warn_only=True)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # before the first CUDA context
```

What each one is for, from PyTorch's Reproducibility note:

- `torch.manual_seed` seeds CPU *and* every CUDA device. You do not need `cuda.manual_seed_all`
  separately.
- `cudnn.benchmark = True` (the performance default) times several conv algorithms on first use
  and keeps the fastest. Benchmark noise means it can pick a different one on the next run, on
  the same machine. Turning it off costs speed and buys reproducibility.
- `cudnn.deterministic = True` controls only conv determinism.
  `torch.use_deterministic_algorithms(True)` is broader: it makes *every* op deterministic where a
  deterministic implementation exists, and raises `RuntimeError` where none does, so
  nondeterminism becomes an error you can see rather than a drift you cannot.
- `warn_only=True` downgrades that error to a warning. Under `filterwarnings = ["error"]` the
  warning still fails the test, which is usually the behavior you want: it tells you exactly
  which op is the problem.
- `CUBLAS_WORKSPACE_CONFIG` must be set before the first CUDA context is created, or cuBLAS
  reductions stay nondeterministic.

DataLoader workers get their own RNG state, so they need separate handling: straight from the
PyTorch docs:

```python
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(SEED)

DataLoader(dataset, batch_size=8, num_workers=2, worker_init_fn=seed_worker, generator=g)
```

Two caveats worth stating up front, because they cause "but I seeded it" confusion:

- **CPU and GPU results differ even with identical seeds.** Different kernels, different
  reduction order. Do not assert bitwise equality across devices.
- **PyTorch does not guarantee reproducibility across versions, commits, or platforms.** Pin the
  version if you assert on exact numbers, or assert on properties instead.

Attention has its own determinism story. `scaled_dot_product_attention` dispatches to several
backends; the Flash, Efficient, and cuDNN backends have **non-deterministic backward passes** by
default. `torch.use_deterministic_algorithms(True, warn_only=False)` forces deterministic
backward for Flash and Efficient and disables the cuDNN backend entirely. To pin the backend:

```python
from torch.nn.attention import sdpa_kernel, SDPBackend

with sdpa_kernel(SDPBackend.MATH):     # always deterministic, both directions
    out = F.scaled_dot_product_attention(q, k, v)
```

## Shape and dtype contracts

The highest-value ML test there is, and the cheapest. Most real ML bugs are a transposed
dimension, a missing `unsqueeze`, or a broadcast that silently did the wrong thing.

```python
@pytest.mark.parametrize(
    ("batch", "channels", "size"),
    [(1, 3, 8), (4, 3, 8), (4, 1, 16)],
    ids=["single-sample", "standard-batch", "grayscale-larger"],
)
def test_forward_output_shape(tiny_model, device, batch, channels, size):
    x = torch.randn(batch, channels, size, size, device=device)
    out = tiny_model(x)
    assert out.shape == (batch, 2)
    assert out.dtype == torch.float32
    assert out.device.type == device.type
```

Batch size 1 deserves its own case: it is where a hardcoded dimension or a `squeeze()` without an
explicit `dim` blows up, and it is what inference does in production.

Assert on `.shape` as a tuple, not on `len(out)`. `len` on a tensor gives you dimension 0 only,
so it passes for a completely wrong shape.

## Comparing tensors

Never `==` on float tensors (rule 8). `torch.testing.assert_close` checks values, shape, dtype,
and device, and its failure message reports the largest absolute and relative deviation and where
it occurred.

```python
torch.testing.assert_close(actual, expected)                      # dtype-aware defaults
torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-5)
torch.testing.assert_close(actual, expected, check_dtype=False)   # only when you mean it
```

Default tolerances scale with dtype: float32 gets roughly `rtol=1.3e-6, atol=1e-5`; float16 and
bfloat16 get much looser ones. State them explicitly in a test that cares: it documents the
precision you are claiming.

For a plain boolean check, `torch.allclose` returns a bool with no diagnostic message. Prefer
`assert_close`, which explains itself when it fails.

Non-obvious properties worth asserting outright:

```python
assert torch.isfinite(out).all(), "output contains NaN or inf"
assert out.requires_grad
assert not torch.equal(before, after), "the parameters did not move"
```

## Gradient tests

Gradient bugs are silent: the model trains, loss goes down a bit, and one branch never learns.

```python
def test_all_parameters_receive_gradients(tiny_model, tiny_batch):
    x, y = tiny_batch
    loss = torch.nn.functional.cross_entropy(tiny_model(x), y)
    loss.backward()

    dead = [name for name, p in tiny_model.named_parameters()
            if p.requires_grad and (p.grad is None or torch.all(p.grad == 0))]
    assert dead == [], f"parameters with no gradient: {dead}"


def test_gradients_are_finite(tiny_model, tiny_batch):
    x, y = tiny_batch
    torch.nn.functional.cross_entropy(tiny_model(x), y).backward()
    for name, p in tiny_model.named_parameters():
        assert torch.isfinite(p.grad).all(), f"non-finite gradient in {name}"
```

The `dead == []` form matters: it puts the offending parameter names in the failure message.
`assert not dead` tells you only that something is wrong.

For a hand-written `autograd.Function`, check the analytic gradient against numerical
differentiation:

```python
from torch.autograd import gradcheck

def test_custom_op_gradient_is_correct():
    x = torch.randn(4, 3, dtype=torch.double, requires_grad=True)   # float64 required
    assert gradcheck(MyOp.apply, (x,), eps=1e-6, atol=1e-4)
```

`gradcheck` needs `float64`; in `float32` the numerical estimate is noisier than the tolerance.

## The training step

One optimizer step, on a fixed batch. Milliseconds, and it catches a genuinely long list of
bugs: a frozen parameter, a missing `zero_grad`, an optimizer built over the wrong parameter
list, a loss that does not connect to the graph.

```python
def test_one_step_updates_parameters_and_lowers_loss(tiny_model, tiny_batch):
    x, y = tiny_batch
    optimizer = torch.optim.SGD(tiny_model.parameters(), lr=0.1)
    before = {n: p.detach().clone() for n, p in tiny_model.named_parameters()}

    optimizer.zero_grad()
    loss_before = torch.nn.functional.cross_entropy(tiny_model(x), y)
    loss_before.backward()
    optimizer.step()

    with torch.no_grad():
        loss_after = torch.nn.functional.cross_entropy(tiny_model(x), y)

    for name, p in tiny_model.named_parameters():
        assert not torch.equal(before[name], p), f"{name} did not change"
    assert loss_after.item() < loss_before.item()
```

Assert the *direction*, not a specific value. `assert loss_after < loss_before` is a property that
survives version changes; `assert loss_after == pytest.approx(0.6931)` breaks the next time
anything upstream shifts.

Related one-liners worth their own tests:

```python
def test_eval_mode_disables_dropout(model_with_dropout, tiny_batch):
    x, _ = tiny_batch
    model_with_dropout.eval()
    with torch.no_grad():
        torch.testing.assert_close(model_with_dropout(x), model_with_dropout(x))
    # in train() mode the same call would give different results
```

## Overfit one batch

The strongest single signal that a model and its training loop are wired correctly: a model with
enough capacity must be able to memorize four examples. If it cannot, something is broken: the
loss, the label alignment, the learning rate, or the gradient path.

```python
@pytest.mark.slow
def test_model_can_overfit_a_single_batch(tiny_model, tiny_batch):
    x, y = tiny_batch
    optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1e-2)

    for _ in range(200):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(tiny_model(x), y)
        loss.backward()
        optimizer.step()

    assert loss.item() < 0.01, f"could not overfit 4 samples; final loss {loss.item():.4f}"
    assert (tiny_model(x).argmax(dim=1) == y).all()
```

Marked `slow` because 200 steps is not milliseconds. Run it before a release and after any change
to the loss or the training loop.

## Data pipeline tests

The transform stack is ordinary code and deserves ordinary tests. It is also where the highest
density of real bugs sits, because a wrong normalization or a label off by one degrades the model
without ever raising.

```python
def test_normalize_produces_expected_statistics():
    x = torch.rand(100, 3, 32, 32)
    out = normalize(x, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    torch.testing.assert_close(out.mean().item(), (x.mean().item() - 0.5) / 0.5, rtol=1e-4, atol=1e-4)


def test_augmentation_preserves_shape_and_label(sample_image):
    image, label = sample_image
    augmented, aug_label = train_transform(image, label)
    assert augmented.shape == image.shape
    assert aug_label == label            # geometric augmentation must not change the class


def test_dataset_length_matches_source(tmp_path):
    write_fake_records(tmp_path, count=17)
    assert len(MyDataset(tmp_path)) == 17


def test_dataloader_is_reproducible(seeded_generator, seed_worker):
    def batches():
        g = torch.Generator(); g.manual_seed(20260826)
        loader = DataLoader(FakeDataset(32), batch_size=4, shuffle=True,
                            generator=g, worker_init_fn=seed_worker)
        return [b.clone() for b in loader]

    for first, second in zip(batches(), batches()):
        torch.testing.assert_close(first, second)
```

Build the dataset from `tmp_path` (rule 16), never from a real data directory. A unit test that
reads your actual training set is an integration test with a misleading name, and it fails on
every machine that does not have the data.

Train/test leakage is worth a test of its own:

```python
def test_train_and_val_splits_do_not_overlap():
    train, val = make_splits(list(range(100)), val_fraction=0.2)
    assert set(train).isdisjoint(set(val))
    assert len(train) + len(val) == 100
```

## Checkpoints

```python
def test_checkpoint_round_trips(tiny_model, tmp_path, tiny_batch):
    x, _ = tiny_batch
    with torch.no_grad():
        before = tiny_model(x)

    path = tmp_path / "model.pt"
    torch.save(tiny_model.state_dict(), path)

    reloaded = build_model()
    reloaded.load_state_dict(torch.load(path, weights_only=True))
    reloaded.eval()
    tiny_model.eval()

    with torch.no_grad():
        torch.testing.assert_close(reloaded(x), before)
```

`weights_only=True` is the default from PyTorch 2.6 onward and should be explicit in tests: it
refuses to unpickle arbitrary objects. Pass `eval()` on both models so dropout and batchnorm are
in the same mode; forgetting that produces a mysterious mismatch.

`load_state_dict` returns a named tuple of `missing_keys` and `unexpected_keys`, and by default
**does not raise** on either unless `strict=True` (which is the default for the method: but many
codebases pass `strict=False` and then ignore the result). Assert on it:

```python
result = reloaded.load_state_dict(state, strict=False)
assert result.missing_keys == []
assert result.unexpected_keys == []
```

## Device and GPU markers

```python
@pytest.fixture(params=["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def device(request):
    torch = pytest.importorskip("torch")
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available on this machine")
    return torch.device(request.param)
```

Every test requesting `device` runs on CPU by default and on CUDA when the `gpu` marker is
selected. The default `addopts` in `assets/pytest-config.toml` deselects `gpu`, so a laptop run
is CPU-only and a GPU CI job adds `-m gpu`.

`pytest.importorskip("torch")` skips cleanly when torch is not installed, rather than erroring at
collection and taking the whole suite down. Verified signature:
`pytest.importorskip(modname, minversion=None, reason=None, *, exc_type=None)`.

Do not assert bitwise CPU/CUDA equality. Compare with a tolerance if you must compare at all:

```python
@pytest.mark.gpu
def test_cuda_matches_cpu_within_tolerance(tiny_model):
    x = torch.randn(4, 3, 8, 8)
    cpu_out = tiny_model(x)
    cuda_out = tiny_model.cuda()(x.cuda()).cpu()
    torch.testing.assert_close(cuda_out, cpu_out, rtol=1e-4, atol=1e-5)
```

## Behavioral tests

Beyond aggregate accuracy: does the model behave correctly on cases you can name? The MLOps
literature calls these *minimum functionality tests*: specific input/output pairs that encode
domain knowledge.

```python
@pytest.mark.integration
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Natural language processing is the next big wave.", "nlp"),
        ("This is about graph neural networks.", "other"),
    ],
)
def test_classifier_handles_known_examples(trained_predictor, text, expected):
    assert trained_predictor(text) == expected


@pytest.mark.integration
def test_prediction_is_invariant_to_irrelevant_change(trained_predictor):
    """Invariance: a change that should not matter, does not."""
    assert trained_predictor("The model works.") == trained_predictor("the model works")
```

Also worth writing: **directional** tests (a change that should move the prediction a specific
way does), and **robustness** tests (adding whitespace or a typo does not flip the class).

Aggregate metrics get a tolerance, never an equality:

```python
@pytest.mark.integration
def test_accuracy_clears_the_release_threshold(trained_model, held_out_loader):
    accuracy = evaluate(trained_model, held_out_loader)
    assert accuracy == pytest.approx(0.95, abs=0.02)   # 0.95 ± 0.02
```

## What not to unit test

- **Model accuracy in the default suite.** It is slow, hardware-dependent, and it fails for
  reasons that are not regressions. Mark it `integration` and run it on a schedule.
- **Third-party layers.** `nn.Linear` works. Test *your* composition of it.
- **Exact loss values.** Assert direction and finiteness. An exact value breaks on a version bump
  and tells you nothing when it does.
- **Training on real data.** That is an integration test. Unit tests use `tiny_batch`.
- **Anything requiring a downloaded checkpoint.** Network in a unit test (rule 14). Build a tiny
  randomly-initialized model instead: you are testing the plumbing, not the weights.
