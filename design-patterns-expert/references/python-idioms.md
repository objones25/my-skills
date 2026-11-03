# Python mechanics for pattern implementation

The language features patterns are built from, and the documented behavior that contradicts common folklore. Every claim here is from the official docs or a primary source; where a fact was verified by execution it says so.

Contents: [Protocol vs ABC](#protocol-vs-abc) · [singledispatch](#functoolssingledispatch-the-pythonic-dispatch-by-type) · [cache traps](#functoolscache-and-cached_property-traps) · [dataclasses](#dataclass-gotchas) · [closures & partial](#closures-and-partial) · [module-level state](#module-level-state-and-lazy-attributes) · [context managers](#context-managers) · [the folklore table](#folklore-vs-the-docs)

---

## Protocol vs ABC

The single most common implementation decision in pattern work, and the folklore is wrong in specific, checkable ways.

### What Protocol actually gives you

`Protocol` (PEP 544, Python 3.8+) provides **structural typing**: a class satisfies it by having the right methods, without importing or inheriting anything. That is genuinely valuable: implementations in other packages don't take a dependency on your interface.

What it does **not** give you:

**No runtime enforcement at instantiation, by default.** A class that explicitly inherits from a Protocol and implements nothing instantiates, and the unimplemented method returns `None` (the `...` body).

```python
class P(Protocol):
    def m(self) -> int: ...

class Impl(P):
    pass

Impl()        # succeeds
Impl().m()    # returns None at runtime; mypy and pyright reject Impl() itself
```

That is only the runtime half. Two things close the gap without leaving Protocol. First, mypy and pyright both reject `Impl()` with `Cannot instantiate abstract class "Impl"`. Second, marking `m` with `@abstractmethod` makes `Impl()` raise `TypeError` at runtime, exactly like an ABC (`scripts/verify_python_claims.py`, check P2); the typing spec's own example does this. Use an ABC instead only when you also want nominal typing or `register()`.

**`@runtime_checkable` checks much less than it appears to.** The docs: it *"will check only the presence of the required methods or attributes, not their type signatures or types."* A method taking four extra required parameters still passes `isinstance`. The docs' own example: `ssl.SSLObject` passes an `issubclass` check against `Callable` despite an `__init__` that does nothing but raise `TypeError`.

It also sees whatever the object happens to have *at that instant*: an attribute assigned after construction makes an unrelated object pass a data-protocol `isinstance` check.

**`issubclass()` doesn't work with data protocols at all.** A protocol with any non-method member (`name: str`) raises `TypeError: Protocols with non-method members don't support issubclass()`. PEP 544's reason: instance attributes set in `__init__` aren't visible on the class.

**`isinstance` against a runtime-checkable protocol is slow.** The docs say it *"can be surprisingly slow compared to an isinstance() check against a non-protocol class"* and recommend `hasattr` for structural checks in performance-sensitive code. Measured by check P7 across several runs and machines: 65 to 86x a plain class check on CPython 3.11 and 5.6 to 9x on 3.12 to 3.14, roughly 0.15 to 0.4 µs per call depending on the machine. That matters inside a tight loop; next to a network call it is noise.

**Decorate every sub-protocol you check against.** Type checkers treat runtime-checkability as not inherited: mypy and pyright both reject `isinstance(x, B)` when `B` extends a runtime-checkable `A` without its own decorator. CPython 3.11 to 3.14 accept it at runtime and check all of `B`'s members (check P6), so the code runs but fails type checking.

### When ABC is the right call

- **You need enforcement.** Abstract methods must be overridden before instantiation succeeds.
- **You need shared state.** An ABC can be a dataclass whose fields and `__init__` every subclass inherits. Shared *methods* alone do not force an ABC: explicit Protocol subclasses inherit default method bodies (check P3). Shared *fields* do, because a Protocol's annotated attributes are interface members, not storage. This is why **Bridge** with a device list is ABC territory.
- **You need to retrofit types you don't own.** `MyABC.register(tuple)` makes `issubclass(tuple, MyABC)` true. Caveat from the docs: a registered class *"won't show up in their MRO... nor will method implementations defined by the registering ABC be callable (not even via `super()`)."* And `@abstractmethod` *"only affects subclasses derived using regular inheritance"*: registering asserts conformance without checking it, which makes `register()` exactly as unsafe as a runtime protocol check.
- **You want custom structural rules.** `__subclasshook__` lets you customize `issubclass` without registering every class.

Decorator ordering is documented, not folklore: `@abstractmethod` goes **innermost**, under `@classmethod` / `@staticmethod` / `@property`.

### The practical recommendation

Use Protocol as an *interface*: define the contract, type your parameters with it, and let implementations satisfy it structurally without inheriting. That is the case where it is unambiguously better than an ABC.

Explicitly subclassing a Protocol, with `@abstractmethod` on the members that have no sensible default, is the typing spec's documented pattern. Implementations you own get definition-site signature checks and runtime enforcement; outside implementations still satisfy the Protocol structurally, without importing it. Reach for an ABC only for `register()`, shared fields, or when structural implementations should be refused.

One real cost of not inheriting: type checkers can't tell you that your implementation's signature drifted from the interface until something calls it through the protocol-typed parameter. Explicit inheritance, from an ABC or from the Protocol itself, catches it at the definition site (mypy: `Signature of "export" incompatible with supertype "Exporter"`).

> Sources: [typing.Protocol](https://docs.python.org/3/library/typing.html#typing.Protocol) · [abc](https://docs.python.org/3/library/abc.html) · [PEP 544](https://peps.python.org/pep-0544/) · [typing spec, Protocols](https://typing.python.org/en/latest/spec/protocol.html)

---

## `functools.singledispatch`: the Pythonic dispatch-by-type

Replaces the Visitor pattern and type-based `if isinstance` chains.

```python
from functools import singledispatch

@singledispatch
def render(node) -> str:              # the base implementation is the fallback
    raise TypeError(f"cannot render {type(node).__name__}")

@render.register
def _(node: TextNode) -> str:
    return node.text

@render.register
def _(node: ListNode) -> str:
    return "\n".join(render(child) for child in node.children)
```

Documented semantics:
- **Dispatch is on the type of the first argument only.** Visitor's double dispatch is out of scope.
- The undecorated function is registered for `object`, so it is the fallback for unregistered types. Make it raise if silent fallthrough would be a bug.
- **ABC registration works**: registering an implementation for `collections.abc.Mapping` catches virtual subclasses.
- `typing.Union` and `int | float` annotations are supported from 3.11.
- `register()` returns the undecorated function, so each variant remains independently testable.
- Introspection: `render.dispatch(TextNode)` shows what would be chosen; `render.registry` lists everything.

Limitations, two of which bite in practice:
- **No generic-parameter dispatch.** The docs are explicit: `[1,2,3]` and `["a","b"]` dispatch identically; register with `list` itself: the docs note that `list[int]` *"is for static type checkers only and has no runtime impact,"* and using a bare `list[int]` annotation with `@register` raises `TypeError: Invalid annotation ... list[int] is not a class` (3.11 to 3.14).
- **The first argument must be positional.** Calling entirely with keywords raises `TypeError: ... requires at least 1 positional argument`. (Verified by execution.)
- **`bool` beats `int`** via normal MRO: `True` will not reach an `int` handler if a `bool` handler exists.
- **Ambiguous ABC registration is a runtime error**, not a silent pick: registering both `Sized` and `Container` and dispatching a `list` raises `RuntimeError: Ambiguous dispatch`. (Verified by execution.) Register against concrete classes when you can.
- For methods, use `singledispatchmethod`, which dispatches on the first non-`self` argument and **must be the outermost decorator**.

> Source: [functools](https://docs.python.org/3/library/functools.html)

---

## `functools.cache` and `cached_property` traps

`@cache` is `lru_cache(maxsize=None)`: a thin dict wrapper, and unbounded by design.

- **Unbounded means unbounded.** On a long-lived service with an open-ended key space, `@cache` is a memory leak by construction. `lru_cache(maxsize=N)` exists precisely because *"The cache's size limit assures that the cache does not grow without bound on long-running processes such as web servers."*
- **Arguments must be hashable.** A list argument raises `TypeError: unhashable type: 'list'`.
- **Keyword order creates separate entries.** `f(a=1, b=2)` and `f(b=2, a=1)` *"may have two separate cache entries."*
- **References are kept alive.** *"The cache keeps references to the arguments and return values until they age out of the cache or until the cache is cleared."* And: *"If a method is cached, the `self` instance argument is included in the cache."* Verified: a weakref to a cached argument survives `del` plus a `gc.collect()`. **`@cache` on a method pins every instance it has ever seen for the life of the process.**
- **Thread safety is narrower than people assume.** The docs: the cache *"is threadsafe"* in that *"the underlying data structure will remain coherent during concurrent updates"*, immediately followed by *"It is possible for the wrapped function to be called more than once if another thread makes an additional call before the initial call has been completed and cached."* The dict is safe; the function is **not** called-once. Do not use `@cache` as a once-only initialization guard under threads.
- Escape hatches: `cache_info()`, `cache_clear()`, and `__wrapped__` to bypass.

`cached_property`:
- Requires a mutable instance `__dict__`: breaks with `__slots__` (unless `__dict__` is included) and with some metaclasses.
- Interferes with PEP 412 key-sharing dicts, so instances take more space.
- **Version-dependent thread safety, and this one breaks real code:** before 3.12 an *undocumented* per-property lock guaranteed the getter ran once per instance. *"In Python 3.12+ this locking is removed."* Anything relying on run-once behavior under threads is broken on 3.12+.

The official trade-off, from the Programming FAQ: `cached_property` takes no arguments and does not outlive the instance; `lru_cache` handles arguments but keeps instances alive until eviction.

> Sources: [functools](https://docs.python.org/3/library/functools.html) · [Programming FAQ](https://docs.python.org/3/faq/programming.html)

---

## Dataclass gotchas

- **Mutable defaults:** use `field(default_factory=list)`. Since 3.11 the check is *unhashability*, not a `list`/`dict`/`set` blacklist (*"Unhashability is used to approximate mutability"*) so a mutable-but-hashable default still slips through silently.
- **`eq=True` with `frozen=False` sets `__hash__` to `None`**, making instances unhashable. `frozen=True` generates a hash. `eq=False` leaves `__hash__` inherited. If your dataclass needs to go in a set or be a dict key, this is the field to check.
- **`slots=True` returns a *new class*.** The docs are explicit that *"new class will be returned instead of the original one"*, so anything holding a reference to the original (a decorator applied earlier, a registry entry, zero-arg `super()` in a method) can break. The zero-arg `super()` case is fixed in 3.13.14 and 3.14.0; 3.12 and 3.13.0 to 3.13.13 still raise `TypeError` (check D3). Also, passing parameters to a base class `__init_subclass__` with `slots=True` raises `TypeError`.
- `frozen=True` for value objects and for anything a Builder produces: it makes accidental mutation after construction impossible, which is most of the point of having a build step.
- `field(init=False)` for computed or lazily-populated state, so it doesn't appear in the constructor signature.

> Source: [dataclasses](https://docs.python.org/3/library/dataclasses.html)

---

## Closures and `partial`

A closure is a class with `__init__` and `__call__`, written shorter:

```python
def rate_limited(max_per_second: float):
    interval = 1.0 / max_per_second        # captured
    def decorator(fn):
        last = 0.0
        @wraps(fn)
        def wrapper(*args, **kwargs):
            nonlocal last
            wait = interval - (time.monotonic() - last)
            if wait > 0:
                time.sleep(wait)
            last = time.monotonic()
            return fn(*args, **kwargs)
        return wrapper
    return decorator
```

Captured variables are more protected than instance attributes: reaching them requires going through `__closure__`, so anyone doing it knows they shouldn't. They are not private, and they are harder to inspect in a debugger, which is the real trade against a small dataclass.

`functools.partial` is the functional adapter: it binds arguments and returns something callable with a narrower signature. Its weakness is tooling: type checkers infer partial's signature imperfectly, so if the checker's view matters, write a small wrapper function with an explicit signature instead.

Always use `@functools.wraps` on decorators, or `__name__`, `__doc__`, and `__wrapped__` are lost and introspection breaks.

---

## Module-level state and lazy attributes

A module is imported once and cached in `sys.modules`. That makes module-level objects singletons with none of the Singleton pattern's costs and, importantly, without removing the ability to construct a separate instance in a test.

**Avoid I/O at import time.** *"Never assume that simply because your code has been imported, it will be used."* A module that opens a connection on import charges every test and every unrelated script for it.

**PEP 562 lazy module attributes**: a clean way to defer expensive construction:

```python
# models.py
def __getattr__(name: str):
    if name == "model":
        value = _load_model()
        globals()["model"] = value       # memoized; __getattr__ won't fire again
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    return sorted([*globals(), "model"])
```

Module `__getattr__` fires only when normal lookup fails, so the cost is one lookup miss, once. It is not run-once under threads: 8 threads reading `model` concurrently ran the loader 8 times (check C2). Use the locked loader from `creational.md` inside it when first access can be concurrent.

**The prebound method pattern** shares state across module-level functions without exposing the object: `random.randrange`, `randint` and `choice` are bound methods of a single module-level `Random` instance. Useful for a module-shaped API over a stateful object.

> Sources: [PEP 562](https://peps.python.org/pep-0562/) · [Rhodes, Global Object](https://python-patterns.guide/python/module-globals/) · [Rhodes, Prebound Methods](https://python-patterns.guide/python/prebound-methods/)

---

## Context managers

`@contextmanager` gives you acquire/release without a class: the right tool for pool checkout, transaction scope, and temporary state.

Two documented traps:
- **Single use.** *"Context managers created using `@contextmanager` are also single use context managers, and will complain about the underlying generator failing to yield if an attempt is made to use them a second time."* A second `with` on the same object fails (on 3.11 to 3.14 it raises `AttributeError: '_GeneratorContextManager' object has no attribute 'args'`, not the `RuntimeError` the docs' example shows). Using one object as a decorator is fine: a new generator is created on each call.
- **Swallowed exceptions.** If you catch an exception inside the generator and don't re-raise, the `with` statement treats it as handled. That is occasionally what you want and usually a bug.

`ExitStack` is the answer for a *variable* number of resources: *"The primary use case for ExitStack is the one given in the class documentation: supporting a variable number of context managers and other cleanup operations in a single with statement."* It replaces hand-rolled nesting and the try/finally chains people write instead.

> Source: [contextlib](https://docs.python.org/3/library/contextlib.html)

---

## Folklore vs the docs

| Folklore | What the sources say |
|---|---|
| Use a Singleton class for shared state | *"The canonical way to share information across modules within a single program is to create a special module (often called config or cfg)... Because there is only one instance of each module, any changes made to the module object get reflected everywhere."* |
| Static methods organize helpers | *"a far more straightforward way to get the effect of a static method is via a simple module-level function."* |
| `@runtime_checkable` gives real interface checking | Presence only, *"not their type signatures or types"* |
| `isinstance` against a Protocol is cheap | *"can be surprisingly slow"*; docs recommend `hasattr` in hot paths |
| Protocols replace ABCs outright | Protocols don't enforce at instantiation; no `issubclass` for data protocols; runtime-checkability isn't inherited |
| `register()` means the class conforms | It asserts conformance without checking it; abstract methods don't apply to virtual subclasses |
| `@cache` is thread-safe so the function runs once | The dict is coherent; the docs say it is possible for the wrapped function *"to be called more than once"* under concurrency |
| `cached_property` is thread-safe | True before 3.12 via an undocumented lock; *"In Python 3.12+ this locking is removed."* |
| `lru_cache` on a method is harmless | It keeps `self` alive until eviction or clear |
| Dataclasses ban `list`/`dict`/`set` defaults | Since 3.11 the check is unhashability, so a mutable-but-hashable default passes |
| Immutable class + Builder class is the safe pattern | Rhodes ([Builder](https://python-patterns.guide/gang-of-four/builder/)): *"never necessary in Python since the language itself provides built-in syntactic support for optional constructor arguments"* |
| You need a Singleton for canonical instances | `Enum` members already are; CPython flyweights small ints, interned strings, and booleans |
| Runtime Protocol `isinstance` is around 100x slower | 65 to 86x on 3.11; 5.6 to 9x on 3.12 to 3.14 (check P7) |
| Protocols cannot share code, so use an ABC | Explicit subclasses inherit default bodies; the typing spec's own example puts `@abstractmethod` in a Protocol |
| PEP 562 `__getattr__` makes lazy init safe | Same race as `@cache`: the loader can run once per thread (check C2) |
| A `NamedTuple` is faster than a dataclass | Not on 3.11 to 3.14: a dataclass, slotted or not, reads about twice as fast, and a slotted one is smaller (check D4) |
