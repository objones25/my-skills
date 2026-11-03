# Negative space in Python

## `assert` is a debug statement, not a runtime check

`assert cond, msg` compiles to `if __debug__: if not cond: raise AssertionError(msg)`. Running under
`python -O`, `python -OO`, or with `PYTHONOPTIMIZE=1` sets `__debug__` to `False` and the code
generator emits **no code at all** for the statement: the condition is never evaluated and the
message is never built. (Python language reference, "The `assert` statement"; `-O` in
"Command line and environment".)

Consequences:

- Rule 7 (assertions on in production) cannot be satisfied by `assert` in any code whose runtime
  flags you do not control, which includes every library you publish.
- Any logic living inside an `assert` disappears in optimized mode. Never put a side effect there.
- Some container base images and WSGI setups enable `PYTHONOPTIMIZE`; you will not get a warning.

Use `assert` for: test bodies (pytest rewrites it for good failure output), and cheap internal
sanity checks whose disappearance is acceptable. Use `require()` for everything you actually want
running in production.

## The helpers

`assets/negative_space.py` is stdlib-only and needs Python 3.10+. Copy it into the project (do not
import from a skill directory). The API:

| Call | Raises | Use for |
|---|---|---|
| `require(cond, msg)` | `CheckFailed` (subclass of `AssertionError`) | Programmer errors. Survives `-O`. Does not narrow types. |
| `expect(x, msg)` | `CheckFailed` if `x is None` | Same as `require(x is not None)`, but returns `x` typed without `None`. |
| `unreachable(msg)` | `CheckFailed` | `else` branches and match arms that cannot happen. |
| `bounded(it, limit, name=...)` | `CheckFailed` on the `limit+1`-th item | Any loop over something whose length you did not compute. |
| `check_shape(arr, spec, name=...)` | `CheckFailed` | Tensor/array shape with named or wildcard dims: `check_shape(x, ("B", 3, 224, 224))`. |
| `check_finite(arr, name=...)` | `CheckFailed` | NaN/inf gate on a loss or gradient: torch tensors, NumPy arrays (imports numpy only then), or floats. |

`CheckFailed` subclasses `AssertionError` so existing `pytest.raises(AssertionError)` tests and
crash handlers keep working.

## Footguns

**Tuple assert.** `assert (x > 0, "x must be positive")` asserts a two-element tuple, which is
always truthy. It never fires. Python emits a `SyntaxWarning` for this (seen on 3.10 through 3.14, and not on 3.9, on 2026-11-03 on macOS arm64), which is easy to
miss in a noisy log. The audit script flags it as NSP002.

**Array truthiness.** `assert x == y` on NumPy arrays or torch tensors raises
`ValueError: The truth value of an array with more than one element is ambiguous`, so the check
fails for the wrong reason. Use explicit reductions:

```python
require(x.shape == y.shape, f"{x.shape} != {y.shape}")
require(bool(torch.equal(x, y)), "tensors differ")
require(bool(np.allclose(a, b, rtol=1e-5)), "arrays differ beyond tolerance")
require(bool(torch.isfinite(loss)), f"loss went non-finite: {loss}")
```

Wrap array predicates in `bool()` so a returned 0-d tensor cannot be silently truthy.

**`assert isinstance(...)` as a type narrower.** Type checkers accept it; `-O` deletes it, and the
`None` it was guarding flows through. Use an explicit `if not isinstance(...): raise CheckFailed(...)`
(narrows and survives `-O`) or `expect()` for `None`. Plain `require(isinstance(...))` survives
`-O` but does not narrow; see the next section.

**Expensive checks in hot loops.** `require(not (set(train) & set(val)))` is O(n) and fine once per
epoch, wrong once per batch. Hoist to the boundary where the object is created, or gate that single
check on a module-level `CHECK_EXPENSIVE` flag.

## Checks and the type checker

A type checker (pyright, which is what VS Code's Pylance runs; or mypy) proves properties before the
program runs. A runtime check proves them when it runs. They overlap, and they disagree in two
predictable ways:

1. **The checker cannot see through `require`.** It narrows a type only on syntax it understands:
   `if`, `assert`, `isinstance`, `is None`, and functions typed as `TypeGuard`/`TypeIs`. To it,
   `require(item is not None)` is a call that returns `None`, so `item` is still `Item | None` on
   the next line. The check is correct; the checker cannot prove it. Fix the code so both agree.
2. **A check that restates an annotation is filler.** `require(isinstance(path, Path))` on a
   parameter annotated `path: Path` tells a typed caller nothing. Pyright strict reports it as
   `reportUnnecessaryIsInstance`. Keep runtime type checks for values whose type the checker does
   not know: parsed JSON, `pickle`/`torch.load` output, `**kwargs`, anything typed `object`/`Any`.

Each row below is reproduced by `assets/examples/typing_before.py` and fixed in
`typing_after.py`; `scripts/test_types.py` keeps both honest.

| Pyright rule (Pylance shows the same ID) | Usual cause next to a check | Fix |
|---|---|---|
| `reportOptionalSubscript`, `reportOptionalMemberAccess` | `require(x is not None)`, then `x[...]` / `x.attr` | `x = expect(x, msg)`, or `if x is None: raise CheckFailed(msg)` |
| `reportTypedDictNotRequiredAccess` | `state["key"]` on a `total=False` / `NotRequired` key | `expect(state.get("key"), msg)`; if a key is always set after some step, give that step its own TypedDict with the key required |
| `reportGeneralTypeIssues`: "Expected mapping for dictionary unpack operator" | `{**x}` where `x` may be `None` | narrow first with `expect` |
| `reportArgumentType`: TypedDict "not assignable to `dict[...]`" | a parameter annotated `dict` receiving a TypedDict (LangChain's `RunnableConfig` is a `total=False` TypedDict) | annotate the parameter with the TypedDict itself, or `Mapping[str, object]`. A TypedDict is not a `dict` to the checker, because a `dict` parameter could gain arbitrary keys. |
| `reportPossiblyUnbound` (JSON output: `reportPossiblyUnboundVariable`) | using a loop variable after the loop, e.g. `require(attempt == MAX)` | bind it before the loop (`attempt = 0`); the check then also catches "ran zero times" |
| `reportUnnecessaryIsInstance` | `require(isinstance(arg, T))` where `arg: T` | delete it; check something the annotation cannot say (a range, a relationship, a length) |
| `reportImplicitOverride` (opt-in) | a subclass method such as `visit_Assert` without `@override` | add `@override`, so renaming the base method makes the checker fail instead of silently orphaning yours |

`cast()` turns a value of unknown type into a typed one and checks nothing at runtime.
A `cast` **after** a runtime check states a verified fact; a `cast` **instead of** one is an
unverified claim. `assets/examples/before.py` has the second kind; `after.py` replaces it with the
first.

`@override` lives in `typing` from Python 3.12. For 3.10/3.11 without a `typing_extensions`
dependency, import it from `typing_extensions` under `if TYPE_CHECKING:` (checkers ship its stubs)
and fall back to an identity decorator at runtime; `scripts/audit_negative_space.py` shows the
pattern.

### Ruff overlaps the audit for `assert`, not for `require`

Ruff's rules see `assert` statements and `except` clauses. They do not know `require()` is a check,
so they cannot do NSP001, NSP003/NSP004 on `require` calls, NSP005, or NSP009.

| NSP | Ruff equivalent | Note |
|---|---|---|
| NSP002 tuple assert | `F631` | on by default in ruff |
| NSP004 compound check | `PT018` | `assert` only |
| NSP006 bare except | `E722` | |
| NSP007 swallowed exception | `S110`, `S112` | `pass` / `continue` bodies only; `BLE001` flags `except Exception` itself |
| NSP008 plain assert | `S101` | ignore it in test files |
| NSP009 stale `# nsp: ignore` | `RUF100` for `# noqa` | `RUF100` also flags a `# noqa: BLE001` when `BLE` is not selected: enable the rule or delete the comment |

## Where each mechanism belongs

| Situation | Mechanism |
|---|---|
| Argument from a caller you own | `require(...)`; `expect(...)` when the value may be `None` |
| Argument parsed from CLI/JSON/HTTP | `raise ValueError(...)`, caught at the edge |
| Impossible `else` / unhandled enum member | `unreachable(...)` |
| Structural property of a data object | validate once in `__post_init__` on a `frozen=True` dataclass |
| Property a type can express | encode in the type: `Literal`, `Enum`, `NewType`, `tuple[int, ...]` |
| Property only tests can check | `assert` in the test |
| Loop over an unknown-length iterable | `bounded(it, MAX)` |
| Numeric health of a training step | `check_finite(loss)` |

## Types do the work assertions otherwise repeat

```python
Split = NewType("Split", tuple[tuple[int, ...], tuple[int, ...]])   # not any pair of lists
Mode = Literal["train", "val", "test"]                              # not any str
@dataclass(frozen=True)                                             # cannot be invalidated later
```

Static types are checked before the program runs and cost nothing at runtime, but they are erased
at runtime: a `Mode` annotation does not stop `"trian"` arriving from a config file. Types for
what your own code can express; `require` for what crosses a boundary.

## ML-specific checkpoints

Places where a wrong value is silent rather than loud, so they carry the highest assertion value:

- **Dataset/collate**: batch shape, dtype, label range `require(0 <= y.min() and y.max() < n_classes)`,
  and no NaNs in the inputs.
- **Split construction**: no overlap between train/val/test (the threaded example in SKILL.md); assert
  it where the split is built and again where the loaders are constructed (a pair assertion).
- **Forward pass**: `check_shape` on input and output, since a wrong broadcast changes the loss without
  changing the shape of the loss.
- **Loss**: finite, and in range for the loss type (cross-entropy ≥ 0; a "loss" of exactly 0.0 at step 1
  is a leak, not a triumph).
- **Optimizer step**: gradient norm finite and non-zero; a silently zero grad trains nothing for hours.
- **Checkpoint**: assert the key set on save and on load, and assert the loaded step is greater than
  the previous one.
- **Normalization statistics**: computed on train only. Assert the val indices were not touched. That is the
  bug that inflates every published number.

## pytest interaction

- pytest rewrites `assert` in test files for readable failures. Keep using plain `assert` there.
- `CheckFailed` subclasses `AssertionError`, so `pytest.raises(AssertionError)` catches `require`
  failures. Prefer catching `CheckFailed` explicitly when you mean a contract violation.
- Test the negative space directly: for each `require`, a test that trips it. That is what turns a
  contract into a tested contract.
- Do not run the test suite under `-O`. If CI does, the assertions in the tests themselves vanish and
  every test passes.
