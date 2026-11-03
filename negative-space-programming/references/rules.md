# The ten rules, with rationale

Each rule states the practice, why it holds, and where it comes from. Sources are listed at the
bottom; `[P10]` is NASA's Power of Ten, `[TS]` is TigerStyle, `[OSDI14]` is the Yuan et al. failure
study.

---

## 1. Assert arguments, returns, and invariants

Minimum average of two runtime assertions per function `[P10 rule 5]`. A function must not operate
blindly on data it has not checked `[TS]`. The purpose of a function is to raise the probability
that the program is correct; assertions are part of how it does that.

Two is an *average*, not a quota. A three-line pure helper called from one place may need none; a
function that mutates shared state or crosses a module boundary may need six. The audit script
flags functions below two so you make the decision consciously, not so you pad them.

Where to put them:

```python
def compact(entries: list[Entry], keep: int) -> list[Entry]:
    require(keep >= 1, f"keep={keep}")                    # precondition: argument
    require(entries, "compact() on empty log")            # precondition: argument
    invariant_size = sum(e.size for e in entries)

    result = _merge(sorted(entries, key=lambda e: e.key))[:keep]

    require(len(result) <= keep, "compaction exceeded budget")             # postcondition
    require(sum(e.size for e in result) <= invariant_size, "size grew")    # invariant
    return result
```

## 2. Assert positive and negative space

The golden rule `[TS]`: assert what you expect **and** what you do not expect, because bugs live
where data crosses the valid/invalid boundary.

Positive space is the easy half and the half you already write. To find the negative space, take
each positive assertion and ask what it *implies*:

| You asserted | The negative space it implies |
|---|---|
| `len(train) == k` | `len(train) + len(val) == n`: nothing was dropped |
| `k >= 1` | `k <= n - 1`: the other side is not empty |
| indices come from `range(n)` | no index repeats; `set(train) & set(val) == set()` |
| `loss` is finite | `loss >= 0` for a cross-entropy; `not math.isnan(loss)` |
| a buffer was filled | it was filled *entirely*; check for the underflow, not just the overflow `[TS]` |

The same rule extends to tests: test with invalid data and with data that becomes invalid, not only
the happy path `[TS]`.

## 3. Programmer errors crash; operating errors are handled

Assertions detect programmer errors, which are unexpected. Operating errors are expected and must
be handled. The only correct response to corrupt code is to crash `[TS]`. This downgrades a
catastrophic correctness bug (wrong answer, silently) into a liveness bug (loud stop).

The decision is about the *origin* of the value, not the predicate:

| Value's origin | Bad value means | Mechanism |
|---|---|---|
| Another function you own, in-process | your code is wrong | `require(...)` → crash |
| A literal or config your team controls | deploy is wrong | `require(...)` at startup → crash |
| User input, network, file, CLI arg, parsed JSON | the world is normal | `raise ValueError/OSError` → caught and reported |
| A hardware/OS call | expected failure | handle, retry with a bound, or propagate typed |

Validate at the boundary, assert inside it. Once a request has passed validation at the edge, every
function downstream may assert the validated properties rather than re-validating them.

## 4. Put a bound on everything

Everything in reality has a limit `[TS]`, so loops, queues, buffers, retries and recursion get an
explicit upper bound, asserted. Where a loop genuinely cannot terminate (an event loop), assert that
fact so the reader knows it is deliberate `[TS]`. NASA states this as: all loops must have a fixed
upper bound, and do not use recursion `[P10 rules 1–2]`.

Assert the **minimum** too. A retry loop that runs zero times because the collection was empty is
as much a bug as one that runs forever, and it is much quieter.

```python
MAX_RETRIES = 5

attempts = 0
for attempts in bounded(range(1, MAX_RETRIES + 1), MAX_RETRIES):
    if fetch():
        break
else:
    raise TimeoutError(f"failed after {MAX_RETRIES} attempts")
require(1 <= attempts <= MAX_RETRIES, f"retry counter corrupt: {attempts}")
```

Bounds are also a performance property: an unbounded queue is an unbounded tail latency.

## 5. Pair assertions

For every property you want to enforce, find at least two different code paths where it can be
asserted. For example, assert data is valid right before writing it to disk, and again right after
reading it back `[TS]`.

The pair catches what one cannot: a single assertion tells you the property held at that point; the
pair tells you *which side* broke it. In ML code the natural pairs are collate-then-forward
(shape asserted in the DataLoader, asserted again at the top of `forward`), and save-then-load
(checkpoint keys asserted on write and on read).

## 6. Split compound assertions

`require(a); require(b)` over `require(a and b)` `[TS]`. Simpler to read, and the traceback names
the half that failed instead of the conjunction. Use a single-line `if` for an implication:
`if a: require(b)`.

The related style rule: state invariants positively `[TS]`. `if index < length` reads correctly;
`if not (index >= length)` makes the reader do the negation. Negative *space* does not mean
negated *expressions*.

## 7. Keep assertions on in production

Production is where the states you never imagined actually occur `[TS]`, so that is where the
checks earn their keep. Turning them off for speed optimizes the case you already tested.

Concretely in Python: this means not depending on `assert`, since `-O` and `PYTHONOPTIMIZE` remove
it (`references/python.md`). Where an assertion is genuinely hot (inside a per-element loop), hoist
it out of the loop or gate that one check behind a flag, rather than disabling the whole class.

The cost argument is usually backwards: a comparison is nanoseconds, a wrong number written to a
database is permanent.

## 8. Handle every error path

An analysis of production failures in distributed data-intensive systems found that almost all
(92%) of catastrophic failures resulted from incorrect handling of non-fatal errors that the
software had already explicitly signalled `[OSDI14]`. The error was detected. The handler was wrong
or empty.

So: no bare `except:`, no handler whose body is `pass`, no `except Exception` that logs at debug
level and continues with a corrupt object. If a failure is truly ignorable, name the exception
class narrowly and write the reason in a comment. That comment is the negative space.

## 9. Make illegal states unrepresentable

The strongest assertion is the one you never have to write, because the type cannot hold the bad
value. Prefer, in order: a type that excludes the state, a constructor that validates once, an
assertion at every use.

```python
@dataclass(frozen=True)
class Split:
    train: tuple[int, ...]
    val: tuple[int, ...]

    def __post_init__(self) -> None:      # validated once, at construction
        require(not (set(self.train) & set(self.val)), "overlapping split")
```

Every function taking a `Split` now gets the no-overlap property for free, and `frozen=True` means
nobody can invalidate it later. Related: declare variables at the smallest possible scope, and
compute them close to where they are used, so there is less distance in which an invariant can rot
`[TS]`.

## 10. Assertions must be pure

A check that mutates state, such as `require(queue.pop() is not None)`, changes program behaviour
depending on whether checks run. In Python that difference appears the first time someone runs with
`-O` or a helper is wrapped in a debug flag. Read state in a check; never write it. Also avoid
checks whose evaluation is expensive enough that someone will later be tempted to strip them.

---

## Beyond the ten

These come from the same sources and cost nothing to apply once the ten are in place.

- **Pass options explicitly at the call site** rather than relying on library defaults, so a change
  of defaults upstream cannot silently change your behaviour `[TS]`.
- **Simple, explicit control flow.** Split compound boolean conditions into nested branches so the
  reader can verify every case is covered, and ask whether an `if` needs a matching `else` to
  handle or assert the negative space `[TS]`.
- **A blatantly true assertion can replace a comment** where the condition is critical and
  surprising: it is documentation the interpreter checks `[TS]`.
- **Assert relationships between constants** at import/compile time. A check that runs before the
  program does is the cheapest one available `[TS]`.
- **Assertions are a safety net, not a substitute for understanding** `[TS]`. Build the mental
  model, encode it as assertions, then write the code, not the reverse.

## Sources

| Tag | Source |
|---|---|
| `[P10]` | Gerard J. Holzmann, *The Power of 10: Rules for Developing Safety-Critical Code*, NASA/JPL, https://spinroot.com/gerard/pdf/P10.pdf |
| `[TS]` | *TigerStyle*, TigerBeetle, https://github.com/tigerbeetle/tigerbeetle/blob/main/docs/TIGER_STYLE.md (Safety section). Talk: Joran Dirk Greef, *TigerStyle! Or How To Design Safer Systems in Less Time* |
| `[OSDI14]` | Yuan et al., *Simple Testing Can Prevent Most Critical Failures*, OSDI '14, https://www.usenix.org/system/files/conference/osdi14/osdi14-paper-yuan.pdf |
| | *Pair assertions*: https://tigerbeetle.com/blog/2023-12-27-it-takes-two-to-contract |
| | Framing article: https://double-trouble.dev/post/negativ-space-programming/ (secondary; its examples are Zig, and it notes it is AI-drafted) |

Where this file paraphrases TigerStyle, the wording is ours; TIGER_STYLE.md is Apache-2.0 licensed
if you want to quote it directly.
