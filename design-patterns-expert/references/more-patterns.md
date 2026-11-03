# More patterns: boundaries, values, and dispatch

Patterns the original catalog lacked. Each one has a runnable, tested example in `examples/` on the same toy domain (a small shop's orders), so you can show real output instead of describing it. Run an example before quoting its numbers.

Contents: [Functional Core, Imperative Shell](#functional-core-imperative-shell) · [Parse, don't validate](#parse-dont-validate) · [Value Object](#value-object) · [Sentinel and Null Object](#sentinel-object-and-null-object) · [Exhaustive dispatch with match](#exhaustive-dispatch-match--union--assert_never) · [Pipeline vs Chain of Responsibility](#pipeline-vs-chain-of-responsibility)

Hedged requests and retry budgets are resilience patterns and live in `resilience.md`.

---

## Functional Core, Imperative Shell

**What it is.** The core decides and returns *effects as plain data* (`PlacePurchaseOrder("MUG", 20)`); it never reads a clock, a database or the network. A thin shell gathers inputs, calls the core, and performs the effects.

**Buys:** business rules testable with plain values: no mocks, no fakes, no database, no frozen clock. `examples/functional_core.py` tests the Monday-summary rule by passing `date(2026, 10, 5)`.
**Costs:** inputs must be gathered up front, which is awkward when the decision needs data it only discovers mid-way (read, decide, read more). Effects-as-data adds one small type per effect.

**Relation to Ports & Adapters.** Same goal, smaller scale. Ports & Adapters puts an interface between the domain and each dependency; FC/IS removes the dependency from the core entirely. Reach for FC/IS first for a single use case; reach for ports when the domain must *call* outward during a decision.

**When not to:** a decision that interleaves many reads and writes (a saga step, a chatty protocol). Forcing it into one pure function produces a core that takes a giant snapshot of the world.

**Failure modes**
- The shell grows `if` statements. Any branch in the shell is a business rule that escaped the core; move it.
- Effects interpreted by `isinstance` chains in the shell. Use `match` over a closed Union of effect types (below), so a new effect type that the shell forgets is a type error.

> Source: Gary Bernhardt, [Functional Core, Imperative Shell](https://www.destroyallsoftware.com/screencasts/catalog/functional-core-imperative-shell) (screencast) and [Boundaries](https://www.destroyallsoftware.com/talks/boundaries) (talk)

---

## Parse, don't validate

**What it is.** Convert untrusted input into a type that *cannot* hold an invalid value, once, at the boundary. A validator checks and throws the knowledge away; a parser checks and keeps the knowledge in the type, so downstream code never re-checks.

```python
def parse_order(raw: Mapping[str, object]) -> OrderRequest: ...   # the only function that sees raw input
def line_total(req: OrderRequest) -> Money: ...                     # no checks: the type guarantees them
```

**Buys:** validation happens in one place; every downstream signature documents what it can rely on; a forgotten check becomes a type error instead of a production bug.
**Costs:** one small type per constrained value. Constructors that raise must be called inside the parser, not scattered.

**In Python:** `@dataclass(frozen=True, slots=True)` value types with `__post_init__` checks; collect *all* field errors before raising (see `examples/parse_dont_validate.py`, which reports `sku` and `quantity` errors together). Pydantic models do the same job at an HTTP boundary; keep them in the adapter and hand the domain plain dataclasses.

**Python-specific trap:** `bool` is a subclass of `int`, so `isinstance(True, int)` is true and `quantity=True` passes a naive integer check. The example rejects it explicitly and its test covers it.

**When not to:** internal data that never crosses a trust boundary. Wrapping every `int` in a class is ceremony.

> Source: Alexis King, [Parse, don't validate](https://lexi-lambda.github.io/blog/2019/11/05/parse-don-t-validate/) (2019)

---

## Value Object

**What it is.** A small object whose identity *is* its value: two `Money(1200, "USD")` are equal and interchangeable. Money, date ranges, SKUs, coordinates.

**In Python** this is one line: `@dataclass(frozen=True, slots=True)`. `frozen` generates `__eq__` and `__hash__` from the fields (so it works as a dict key), and `slots` makes instances smaller: 48 B versus 56 B for a `NamedTuple` (64 B on 3.14), with attribute reads about twice as fast as a `NamedTuple`'s on CPython 3.11 to 3.14 (check D4 in `scripts/verify_python_claims.py`).

**Rules that make it work**
- Immutable: operations return new values (`Money.__add__` returns a new `Money`).
- Integer minor units for money, never floats.
- Operations enforce their own invariants (`Money` refuses to add USD to EUR).

> Source: Martin Fowler, [ValueObject](https://martinfowler.com/bliki/ValueObject.html)

---

## Sentinel Object and Null Object

**Sentinel Object.** A unique object that means "not supplied" when `None` is a legitimate value. Code checks identity (`is`), never equality. `functools.lru_cache` uses one internally.

For typed code, use a one-member `Enum` instead of a bare `object()`: mypy and pyright can narrow `value is MISSING` on an enum member, but cannot narrow on an `object()` instance. `examples/null_object_sentinel.py` passes `mypy --strict` with no `type: ignore`.

**Null Object.** An object with the real interface and do-nothing behaviour (`NullAuditLog.record()` does nothing), so callers never write `if audit is not None`. It is a Strategy whose strategy is "nothing".

**Buys:** removes `None` checks at every call site; makes "off" a value you can inject at the composition root.
**Costs:** hides absence. If "no logger" is a configuration error, a Null Object turns a loud failure into silence.

**When not to:** when absence is an error the caller must handle. Raise, or return `None` and let the type checker force the check.

> Source: Brandon Rhodes, [The Sentinel Object Pattern](https://python-patterns.guide/python/sentinel-object/)

---

## Exhaustive dispatch: match + Union + assert_never

The modern replacement for Visitor and for `if isinstance(...) elif isinstance(...)` chains, **when the set of types is closed** (you own them all).

```python
OrderEvent = OrderPlaced | OrderRefunded | OrderShipped

def revenue_delta(event: OrderEvent) -> int:
    match event:
        case OrderPlaced(total_cents=cents):    return cents
        case OrderRefunded(amount_cents=cents): return -cents
        case OrderShipped():                    return 0
        case _:                                 assert_never(event)
```

Add `OrderCancelled` to the Union without a `case`, and both checkers fail the build (real output from `examples/match_dispatch.py` with one case removed):

```
mypy:    error: Argument 1 to "assert_never" has incompatible type "OrderCancelled"; expected "Never"
pyright: error: Argument of type "OrderCancelled" cannot be assigned to parameter "arg" of type "Never"
```

**Choosing between the three dispatch tools**

| Set of types | Who adds new types | Use |
|---|---|---|
| Closed | you, in this module | `match` + Union + `assert_never` (exhaustiveness checked) |
| Open | other modules or packages | `functools.singledispatch` (registration, no exhaustiveness) |
| Values, not types | anyone | a `dict` of callables, or a Registry |

`typing.assert_never` needs Python 3.11; `match` needs 3.10. The scanner reports class-pattern `match` statements as `isinstance_chain` leads with the note that a closed Union with `assert_never` is fine as it is.

> Sources: [PEP 634](https://peps.python.org/pep-0634/) (specification) · [PEP 636](https://peps.python.org/pep-0636/) (tutorial) · [typing.assert_never](https://docs.python.org/3/library/typing.html#typing.assert_never)

---

## Pipeline vs Chain of Responsibility

These are often confused, including in earlier versions of `behavioral.md`.

| | Chain of Responsibility | Pipeline |
|---|---|---|
| How many handlers run | until the first one accepts | all of them |
| What flows between them | the same request, unchanged | each stage's output is the next stage's input |
| Typical use | approval routing, fallback lookups, middleware that can short-circuit | enrichment, ETL, LLM agent steps |
| Defining test | a later handler can be skipped | removing a stage changes the result |

`examples/pipeline_vs_chain.py` shows both on orders: a refund of 1,500 cents is decided by `auto_approve` and the trace shows no other handler ran; the pipeline prices, taxes and filters orders.

**Pipelines from generators (the Python-native Iterator pattern).** Each stage takes an iterable and yields; composing them is function application. Stages are lazy: pulling the first result from a pipeline over a million orders pulls exactly one order from the source (the test asserts this). Costs: a generator can be consumed once, and an exception surfaces where the result is consumed, not where the bad record entered.

**Chain failure modes**
- No terminal handler, so an unmatched request falls off the end. End the chain with a handler that always answers, or raise a named error (the example does both).
- Handlers that mutate the request, which turns the chain into a pipeline nobody designed.

> Source: Brandon Rhodes, [The Iterator Pattern](https://python-patterns.guide/gang-of-four/iterator/) for the generator form; the GoF definitions for the distinction.
