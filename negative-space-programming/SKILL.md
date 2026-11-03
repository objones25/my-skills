---
name: negative-space-programming
description: "Write and review code that states what must never happen: runtime checks on arguments, returns and invariants, a bound on every loop, queue and retry, and a hard line between programmer errors that crash and operating errors that are handled. Use whenever the user asks to make code more robust, defensive, fail-fast, safer, or harder to misuse; when they mention assertions, invariants, preconditions, postconditions, contracts, defensive programming, negative space, TigerStyle, or NASA's Power of Ten; when hardening code that swallows exceptions, loops without a bound, or trusts its inputs; when pyright/Pylance/mypy flag Optional or TypedDict access right next to a runtime check; and proactively when writing a training loop, data pipeline, parser, or any function whose bad output would be wrong rather than loud. Ships an audit script that flags low assertion density, unbounded loops, swallowed exceptions and stale suppressions, and fails the build on violations."
---

# Negative Space Programming

Most code describes what should happen. The bugs live in what shouldn't. A batch tensor arrives
with 63 rows instead of 64 and the model trains anyway. A train/val split overlaps by 3 indices and
validation accuracy quietly climbs. A retry loop never terminates because the server keeps
answering 503. None of these raise. Negative space programming is the practice of writing the
*absence* down, as executable checks that crash the moment reality leaves the space you designed
for.

The name comes from drawing: you define the vase by shading everything that is not the vase. In
code, the vase is your logic and the shading is assertions.

> The golden rule is to assert the positive space you expect **and** the negative space you do not
> expect, because the interesting bugs live where data crosses the boundary between them.
> (TigerStyle, paraphrased)

Core rules are language-agnostic; every example is Python, with mappings to C++/Rust/Zig/TypeScript
in `references/other-languages.md`.

## Definitions

Defined once, used throughout.

| Term | Meaning |
|---|---|
| **Assertion** | A runtime check of a condition the programmer believes is *always* true. Failure means the code is wrong, not the input. Correct response: crash. |
| **Positive space** | The states your logic is written to handle. `len(train) == int(n * frac)`. |
| **Negative space** | The states your logic assumes never occur. `set(train) & set(val) == set()`. Usually unwritten. This skill writes it down. |
| **Precondition** | A property that must hold when a function is entered. Assert it on the arguments, first thing. |
| **Postcondition** | A property that must hold when a function returns. Assert it on the return value, last thing. |
| **Invariant** | A property that holds throughout: across a loop body, an object's lifetime, or a whole module. |
| **Programmer error** | A bug: an impossible state reached. Cannot be handled; only crashed on. Assertions detect these. |
| **Operating error** | An expected failure of the outside world: file missing, socket reset, malformed user input. Must be *handled*, never asserted. |
| **Pair assertion** | The same property asserted at two different code paths (e.g. before a write and after the matching read), so a violation is caught whichever side introduced it. |
| **Bound** | An explicit maximum on a loop, queue, buffer, retry, or recursion depth, asserted rather than assumed. |
| **Assertion density** | Assertions per function. NASA's Power of Ten sets a minimum average of two. |
| **Fail fast** | Stop at the first detected violation rather than propagating a corrupt state. Converts a silent correctness bug into a loud liveness bug. |
| **Narrowing** | A type checker shrinking a variable's type after a test, e.g. `Item \| None` to `Item` after `if item is None: raise`. `require(...)` is an ordinary call to the checker and does **not** narrow; `expect(...)` does. |
| **`__debug__` / `-O`** | Python's optimization flag. `python -O` removes every `assert` statement from the bytecode, so `assert` alone cannot be your production safety net. See `references/python.md`. |

## The threaded example

One function carries the whole skill. Before:

```python
def split_indices(n, train_frac):
    k = int(n * train_frac)
    idx = list(range(n))
    random.shuffle(idx)
    return idx[:k], idx[k:]
```

Nothing here is wrong. Nothing here is checked either. `train_frac=1.0` returns an empty validation
set. `n=0` returns two empty lists. A later refactor to `idx[:k], idx[k-1:]` overlaps by one and
nothing notices. After:

```python
from negative_space import require

def split_indices(n: int, train_frac: float) -> tuple[list[int], list[int]]:
    # Preconditions: the positive space of the arguments.
    require(n >= 2, f"need at least 2 samples to split, got {n}")
    require(0.0 < train_frac < 1.0, f"train_frac must be strictly interior, got {train_frac}")

    k = int(n * train_frac)
    require(1 <= k <= n - 1, f"split point {k} leaves an empty side for n={n}")

    idx = list(range(n))
    random.shuffle(idx)
    train, val = idx[:k], idx[k:]

    # Postconditions: positive space, then negative space.
    require(len(train) + len(val) == n, "split lost or duplicated samples")
    require(not (set(train) & set(val)), "train and val overlap")
    require(min(train + val) == 0, "index below range")
    require(max(train + val) == n - 1, "index above range")
    return train, val
```

Five of those seven checks are negative space: the empty side, the lost samples, the overlap, and
the index falling off either end. None of them can fire from bad user input, because `n` and `train_frac` are guarded at
the top. They fire only when *this function* is wrong, which is exactly what makes them worth
keeping in production.

## Rules

Full statements, rationale and sources in `references/rules.md`. Read it before a review pass or
when a rule needs justifying to someone.

| # | Rule |
|---|---|
| 1 | Assert arguments, returns, and invariants. Two runtime assertions per function, minimum average. |
| 2 | Assert positive **and** negative space. If you wrote `x > 0`, ask what else is now impossible. |
| 3 | Programmer errors assert and crash; operating errors raise a typed exception and get handled. Never mix. |
| 4 | Put a bound on everything: loops, queues, retries, buffers, recursion. Assert the bound and, where it matters, the minimum. |
| 5 | Pair assertions across a boundary: assert on write and again on read, on send and on receive. |
| 6 | Split compound assertions. `require(a); require(b)` beats `require(a and b)`: the failure names itself. |
| 7 | Keep assertions on in production. That is where the states you never imagined actually occur. |
| 8 | Handle every error path; a swallowed exception is negative space erased. |
| 9 | Make illegal states unrepresentable first. A type or enum that cannot express the bad state beats an assertion that catches it. |
| 10 | Assertions must be pure. A check with a side effect changes behaviour when checks are disabled. |

Rule 3 is the one people get wrong. `require(0 < frac < 1)` for a value hardcoded by the caller is
an assertion. The same check on a value parsed from a config file is validation and belongs in a
`raise ValueError`. Same predicate, different category, decided by where the value came from.

## Workflow

**Writing new code.** Write the signature, then the preconditions, then the body, then the
postconditions, in that order. Writing the checks before the body forces the design decision
("what is `train_frac=1.0` supposed to mean?") to happen while it is still cheap. Every loop gets
its bound at the moment it is written, not after the first hang.

**Hardening existing code.** Run the audit first so the conversation starts from findings rather
than opinion:

```bash
python scripts/audit_negative_space.py <path>            # exits 1 on any finding
python scripts/audit_negative_space.py <path> --json     # machine-readable
python scripts/audit_negative_space.py <path> --strict   # also flags bare `assert` (stripped by -O)
```

If the project is typed, run the type checker in the same pass (`pyright`, which is what Pylance
runs, or `mypy --strict`). Checks and types answer different questions, and each catches mistakes
in the other: see "Checks and the type checker" in `references/python.md`.

Then work the findings in this order, cheapest signal first:

1. Swallowed exceptions and bare `except` (NSP006, NSP007). These hide every other failure.
2. Unbounded loops (NSP005): `while True` without a bound, or `for` over `itertools.count()`,
   `cycle()`, `repeat(x)`, `iter(f, sentinel)`. A hang is the worst failure mode to debug.
3. Broken assertions: tuple asserts that are always true, side effects inside checks (NSP002, NSP003).
4. Stale suppressions (NSP009): a `# nsp: ignore` that no longer matches anything is a hole
   waiting for the next real finding. Delete it.
5. Low-density functions (NSP001), highest-traffic first.

Do not carpet-bomb a codebase to make the density number go up. An assertion that restates the line
above it is a no-op that costs a reader's attention. Each one should encode something you had to
*think* to know. `--min-assertions` is the dial for this: 2 for code that computes, 1 for
dispatch-heavy layers (visitors, routers, thin adapters) where the second check would be filler.
The audit script gates itself at 1 for exactly that reason, and where even one check would only
repeat the type annotation (a pure predicate, a visitor method) it says so with a suppression and
a reason: `# nsp: ignore NSP001 pure predicate`. An `isinstance` check that restates a parameter's
annotation is filler too, and pyright strict reports it (`reportUnnecessaryIsInstance`).

In CI, run the error-severity checks as a hard gate and the density warning as a report:

```bash
python scripts/audit_negative_space.py src/ --select NSP002,NSP003,NSP005,NSP006,NSP007  # gate
python scripts/audit_negative_space.py src/ --select NSP001,NSP009 --min-assertions 2 || true  # report
```

**Reviewing someone's code.** For each function ask: what would have to be true for this to be
wrong, and would anything notice? Then check the four usual holes: an unbounded loop, an exception
handler with no `raise`, a `-1`/`+1` on an index with no range check, and a function that trusts a
caller in another module.

## Python specifics

`assert` is not a runtime check you can rely on: `python -O` deletes it from the bytecode entirely,
condition and message both. `assert (x > 0, "msg")` (a tuple) is always truthy and never fires.
Comparison asserts on NumPy/PyTorch arrays raise `ValueError` on ambiguity instead of failing
cleanly. `assets/negative_space.py` (stdlib only, Python 3.10+) has `require()`, `expect()`,
`unreachable()`, `bounded()`, `check_shape()` and `check_finite()` and avoids all of these; copy it
into the project.

The one trap the helpers cannot hide: `require(x is not None)` checks at runtime but does not
narrow the type, so pyright still reports `reportOptionalSubscript` on `x["id"]` one line later.
Write `x = expect(x, "why it cannot be None")` instead; for an optional TypedDict key, write
`expect(d.get("key"), msg)` rather than `d["key"]`. The full error-to-fix table, plus ruff's
overlap with the NSP codes, is in `references/python.md`.

## Files

| Path | Read when |
|---|---|
| `references/rules.md` | Applying or justifying the ten rules; doing a full review pass. |
| `references/python.md` | Writing the checks in Python: `assert` vs `require`, arrays, dataclasses, typing, pytest. |
| `references/other-languages.md` | The target is C++, Rust, Zig, TypeScript, or Go. |
| `assets/negative_space.py` | Project needs the helpers; copy, do not import from the skill. |
| `assets/examples/` | `before.py` / `after.py`: one module unchecked and checked. `typing_before.py` / `typing_after.py`: checks a type checker rejects, and their fixes. |
| `scripts/audit_negative_space.py` | Any hardening or review pass on existing code. |
| `scripts/test_audit.py` | Changing the audit script; `python scripts/test_audit.py` needs no pytest. |
| `scripts/test_types.py` | Changing any `.py` here; runs pyright strict on the skill and pins the fixture's rule IDs. |
| `pyproject.toml` | The pyright / mypy / ruff settings the skill's own code passes; a starting point for a project. |
