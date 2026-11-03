# Creational patterns

Separating *creating* objects from *using* them. The payoff is that using code never learns which concrete class it got, so implementations can be swapped at one place instead of many.

Contents: [Dependency Injection](#dependency-injection) · [Factory](#factory) · [Registry](#registry) · [Builder](#builder) · [Object Pool](#object-pool) · [Singleton](#singleton-and-why-you-want-a-module)

---

## Dependency Injection

The foundation the others build on. Pass collaborators in; don't construct them inside.

```python
from typing import Protocol

class Emailer(Protocol):
    def send(self, to: str, subject: str, body: str) -> None: ...

class OrderService:
    def __init__(self, emailer: Emailer, repo: OrderRepository) -> None:
        self._emailer = emailer
        self._repo = repo

    def confirm(self, order_id: str) -> None:
        order = self._repo.find(order_id)
        self._emailer.send(order.email, "Confirmed", f"Order {order_id}")
```

Functions take dependencies as parameters; classes take them in `__init__`. Both are DI. Neither needs a framework.

### The composition root

One place (`main()`, an app factory, a CLI entry point) creates concrete objects and wires them. Mark Seemann calls it the [composition root](https://blog.ploeh.dk/2011/07/28/CompositionRoot/): *"a (preferably) unique location in an application where modules are composed together."* Everything else depends on protocols. Keep it small and explicit, and don't try to share it between applications: [in Seemann's words](https://blog.ploeh.dk/2015/01/06/composition-root-reuse/), *"it's what defines a single application."*

```python
def main() -> None:
    emailer = SESEmailer(region="us-west-2")
    repo = SQLOrderRepository(engine)
    service = OrderService(emailer=emailer, repo=repo)
    run(service)
```

Concentrating coupling here is what makes every other pattern pay off. A recommendation that produces an interface but never shows where the concrete choice happens is incomplete.

### Containers: mostly don't

In cosmicpython ch. 13, when Harry wrote an `inspect()`-based injector, Bob *"accused him of overengineering and writing his own DI framework"*; the authors suggest a real DI framework only once dependency chains get deep.

A container earns its place when dependency chains are deep enough that manual wiring becomes genuinely unwieldy: components that need components that need components. Below that, a bootstrap function and `functools.partial` do the job with no magic:

```python
from functools import partial

handlers = {
    "allocate": partial(allocate, uow=uow, publish=publish),
    "change_quantity": partial(change_quantity, uow=uow),
}
```

FastAPI's `Depends` is a reasonable middle ground within a web app: it handles construction, scoping and teardown, and it is visible in the function signature. Its cost is that it only works inside FastAPI, so domain functions should still take plain parameters and let the endpoint do the wiring.

**On testability:** DI makes tests easier by letting you pass a fake instead of patching a module. Fowler notes that a service locator is equally stubbable, so testability alone doesn't decide DI-vs-locator, but it does decide DI-vs-`mock.patch`, and that is the comparison most Python codebases actually face. Patch-based tests couple to import paths and break on refactors; injected fakes don't.

> Sources: [Fowler, Inversion of Control Containers and the Dependency Injection pattern](https://martinfowler.com/articles/injection.html) · [cosmicpython ch. 13](https://www.cosmicpython.com/book/chapter_13_dependency_injection.html)

---

## Factory

**Use a factory when related objects must be chosen as a consistent set.** "Master quality" means a specific video codec *and* a specific audio codec *and* specific settings; mixing them arbitrarily would be a bug. The factory names a valid combination. A web shop's per-country settings (currency, language, tax rate, shipping rules) is the same shape.

**Don't use a factory when arbitrary combinations are legitimate.** If users can pick any codec with any container, you would need a factory class per combination. Use composition and dependency injection instead: more flexible, still decoupled, but you lose the ability to name presets in advance. That trade is the whole decision.

### Pythonic forms, cheapest first

**A dict of classes** covers the simple case:

```python
EXPORTERS: dict[str, type[VideoExporter]] = {
    "low": H264BPExporter,
    "high": H264HQExporter,
    "master": LosslessExporter,
}
exporter = EXPORTERS[quality]()
```

**A dict of tuples** groups related classes:

```python
FACTORIES: dict[str, tuple[type[VideoExporter], type[AudioExporter]]] = {
    "low": (H264BPExporter, AACExporter),
    "master": (LosslessExporter, WavExporter),
}
video_cls, audio_cls = FACTORIES[quality]
```

Cheap, but three real problems: positional order must be remembered; attaching configuration alongside the classes gets ugly fast; and every valid combination is now written out explicitly in one dict, which costs cohesion once there are many. A `NamedTuple` fixes the first problem only.

**A dataclass with `__call__`** is the version that scales:

```python
@dataclass(frozen=True)
class MediaExporter:
    video: VideoExporter
    audio: AudioExporter

@dataclass(frozen=True)
class MediaExporterFactory:
    video_class: type[VideoExporter]
    audio_class: type[AudioExporter]

    def __call__(self) -> MediaExporter:
        return MediaExporter(video=self.video_class(), audio=self.audio_class())

FACTORIES = {
    "low": MediaExporterFactory(H264BPExporter, AACExporter),
    "master": MediaExporterFactory(LosslessExporter, WavExporter),
}
exporter = FACTORIES[quality]()
```

Named fields instead of positional ones; still callable; and the factory itself can be injected, which means `main()` receives an abstract factory and never learns the concrete classes.

Note the honest limitation: this form calls the classes' initializers with no arguments, so it gives you little control over per-exporter constructor parameters. Solving that by subclassing `MediaExporterFactory` walks you back to the classic factory-class hierarchy, which is fine, but do it knowingly.

If footprint or attribute speed matters in a hot path, use `@dataclass(frozen=True, slots=True)`, not a `NamedTuple`. Measured on CPython 3.11 to 3.14: 48 B per instance against 56 B for a `NamedTuple` (64 B on 3.14), and roughly 11 to 13 ns per attribute read against 20 to 26 ns for a `NamedTuple`. A plain dataclass reads about as fast as the slotted one; `slots` mainly buys size. Check D4 prints the numbers on your machine. A plain tuple indexed by position is right only when names would not help.

**Ownership.** A factory does not own what it creates and keeps no reference to it. Python's refcounting means an unreferenced object is collected, but when lifetime matters (file handles, connections) say in the docstring that the caller owns the result.

### Brandon Rhodes' objection, and when it applies

Rhodes calls the GoF *Factory Method* *"a poor fit for Python"*: *"It was designed for underpowered programming languages where classes and functions can't be passed as parameters or stored as attributes."* Same for Abstract Factory: *"an awkward workaround for the lack of first-class functions and classes in less powerful programming languages."*

He is right about the *mechanism*: in Python you pass the class or the callable, you don't build a parallel hierarchy of creator classes. He is not arguing against the *intent*: grouping related choices and keeping construction out of using code. A dict of dataclass factories honors the intent using Python's actual powers, and that is the version to write.

> Sources: [Rhodes, Factory Method](https://python-patterns.guide/gang-of-four/factory-method/) · [Rhodes, Abstract Factory](https://python-patterns.guide/gang-of-four/abstract-factory/)

---

## Registry

A dict from key to callable, populated by decorators, so new behaviors are added by adding a file rather than editing a dispatch table.

```python
from pathlib import Path
from typing import Callable

ExportFn = Callable[[str, Path], None]
_registry: dict[str, ExportFn] = {}

def register_exporter(name: str) -> Callable[[ExportFn], ExportFn]:
    def decorator(fn: ExportFn) -> ExportFn:
        if name in _registry:
            raise ValueError(f"exporter {name!r} already registered")
        _registry[name] = fn
        return fn
    return decorator

@register_exporter("pdf")
def export_pdf(data: str, path: Path) -> None:
    ...

def export(name: str, data: str, path: Path) -> None:
    try:
        fn = _registry[name]
    except KeyError:
        raise ValueError(f"no exporter {name!r}; have {sorted(_registry)}") from None
    fn(data, path)
```

The duplicate-registration check and the "have: ..." error message matter more than they look: they convert the two most common registry failures into loud ones.

### Discovery

**Directory scan**. Simple, works within your own codebase:

```python
import importlib

def load_plugins(package: str, plugin_dir: Path) -> None:
    for path in sorted(plugin_dir.glob("*.py")):
        if not path.stem.startswith("_"):
            importlib.import_module(f"{package}.{path.stem}")
```

**Entry points**: the right mechanism when third parties ship plugins as installed packages:

```python
from importlib.metadata import entry_points

for ep in entry_points(group="myapp.exporters"):
    register_exporter(ep.name)(ep.load())
```

### Failure modes
- **Import order is the whole pattern's weakness.** A decorator only fires when its module is imported. Nothing registered means silent absence, not an error. Load plugins explicitly at startup and log what got registered.
- **Invisible inventory.** With enough implicit registration, no one can tell what's available by reading. Provide a `list_exporters()` and a CLI command that prints the registry.
- **Registry for two entries.** An `if/else` is fine. The pattern pays off when entries live in files the core module doesn't import.
- Silent overwrite when two plugins claim the same name: hence the check above.

---

## Builder

**Buys:** readable stepwise construction of an object with many optional or nested parts.
**Costs:** two types where one might do, and a `.build()` call that is easy to forget.

```python
@dataclass(frozen=True)
class HTMLPage:
    title: str
    body: str
    metadata: dict[str, str] = field(default_factory=dict)

class HTMLBuilder:
    def __init__(self) -> None:
        self._title = ""
        self._parts: list[str] = []
        self._metadata: dict[str, str] = {}

    def set_title(self, title: str) -> "HTMLBuilder":
        self._title = title
        return self

    def add_heading(self, text: str, level: int = 1) -> "HTMLBuilder":
        self._parts.append(f"<h{level}>{escape(text)}</h{level}>")
        return self

    def build(self) -> HTMLPage:
        return HTMLPage(self._title, "".join(self._parts), dict(self._metadata))

page = (HTMLBuilder()
        .set_title("My Page")
        .add_heading("Welcome")
        .build())
```

Returning `self` gives chaining; `build()` produces an immutable product.

**Check first whether you need it.** Rhodes' verdict is worth internalizing: the newer "Builder" that *"pairs each immutable class in a program with a more convenient builder class"* is *"never necessary in Python since the language itself provides built-in syntactic support for optional constructor arguments."* Keyword arguments with defaults handle most of what a builder is used for. A builder earns its place when construction is genuinely *incremental*: parts appended in a loop, order-dependent assembly, or a fluent DSL that reads well in the domain (query builders, animation sequences, `DataFrame.style`).

**When not to:** two or three fields (use a dataclass); or configuration complex enough that it should be a YAML/JSON file loaded into a dataclass instead of expressed as chained calls.

**The `.build()` trap** is real but type-checkable: annotate the variable as `HTMLPage` and mypy/pyright will flag a missing `.build()` because `HTMLBuilder` isn't assignable to it.

> Source: [Rhodes, Builder](https://python-patterns.guide/gang-of-four/builder/)

---

## Object Pool

**Buys:** reuse of instances that are expensive to create and needed only briefly, such as connections, sockets, large mesh or model objects.
**Costs:** correctness burden that the type system will not enforce.

Both conditions must hold: expensive to create *and* short-lived in use. If either is false, don't pool.

```python
@dataclass
class ReusablePool:
    size: int
    _free: list[Reusable] = field(default_factory=list, init=False)
    _in_use: list[Reusable] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self._free = [Reusable() for _ in range(self.size)]

    def acquire(self) -> Reusable:
        if not self._free:
            raise RuntimeError("pool exhausted")
        obj = self._free.pop()
        self._in_use.append(obj)
        return obj

    def release(self, obj: Reusable) -> None:
        self._in_use.remove(obj)
        obj.reset()                 # not optional, see below
        self._free.append(obj)

@contextmanager
def borrowed(pool: ReusablePool) -> Iterator[Reusable]:
    obj = pool.acquire()
    try:
        yield obj
    finally:
        pool.release(obj)
```

Always hand out via a context manager. Manual `acquire`/`release` pairs leak under exceptions.

### Failure modes
- **Not resetting on release.** The pooled object carries the previous user's state (auth tokens, buffers, partial results) into the next request. This is a data-leak class of bug, not a tidiness issue.
- **Use after release.** Nothing stops it. The context manager makes it unlikely, not impossible.
- **Instances created outside the pool.** The pool's size guarantee is advisory unless you take steps to prevent direct construction.
- **Threads.** Two threads can be handed the same object without locking. Guard the pool.
- Prefer the pool your library already ships (SQLAlchemy's connection pool, `httpx` connection limits) over a hand-rolled one.

If you don't want a class, a module with `acquire()` and `release()` functions is a perfectly good pool and gives you the single-instance property for free.

---

## Singleton, and why you want a module

Don't implement Singleton in Python. **A module is already a singleton**: imported once, cached in `sys.modules`, shared everywhere.

```python
# config.py
database_url: str = "postgresql://..."
debug: bool = False

# anywhere
import config
print(config.database_url)
```

### Why the class version is worse

1. **Global shared mutable state.** Code that looks like it creates a new instance mutates the shared one.
2. **It destroys test isolation.** This is Rhodes' third objection: a module global *"still leaves a programmer free to create other instances of the class — which can be particularly helpful for tests, letting them each test a completely separate object without needing to reset a shared object back to a known good state."* Singleton removes that freedom by design.
3. **`Logger()` lies to the reader.** It looks like construction and isn't.
4. **Hidden coupling.** The dependency doesn't appear in any signature.
5. **Inheritance breaks the guarantee**: a subclass gives you a second instance.
6. **Threading.** Naive implementations race on first construction.

### What the pattern was actually protecting

The legitimate concern is *controlling when* an expensive resource is created. Solve that directly:

```python
from functools import cache

@cache
def get_model() -> Model:
    return load_model()          # runs once per process, on first call
```

Two caveats on `@cache` here: it is not called-once under concurrency (two threads can both enter before either caches), and it keeps its return value alive for the process lifetime. For genuine once-only initialization under threads, use a lock:

```python
_lock = threading.Lock()
_model: Model | None = None

def get_model() -> Model:
    global _model
    if _model is None:              # fast path: no lock once loaded
        with _lock:
            if _model is None:      # re-check: another thread may have loaded it
                _model = load_model()
    return _model
```

With 8 threads hitting it at once, this loads once (check C4); `@cache` loaded 8 times (check C1).

For a lazily-created *module attribute*, PEP 562's module-level `__getattr__` reads well, but it has the same race as `@cache`: 8 threads ran the loader 8 times (check C2). Call the locked loader above from inside it when first access can be concurrent:

```python
# models.py
def __getattr__(name: str):
    if name == "model":
        value = load_model()
        globals()["model"] = value      # memoized: __getattr__ won't fire again
        return value
    raise AttributeError(name)
```

**Avoid I/O at import time** regardless of which form you choose. Rhodes: *"Never assume that simply because your code has been imported, it will be used."* A module that opens a database connection on import imposes that cost on every test and every unrelated script.

> Sources: [Rhodes, Singleton](https://python-patterns.guide/gang-of-four/singleton/) · [Rhodes, Global Object](https://python-patterns.guide/python/module-globals/) · [PEP 562](https://peps.python.org/pep-0562/)
