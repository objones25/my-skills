#!/usr/bin/env python3
# type: ignore
# Deliberately ill-typed: it executes code that mypy and pyright reject (an abstract
# Protocol subclass, a data protocol in issubclass, ...) to show what happens at runtime.
"""Execute every Python-behaviour claim the skill's references make.

Each check prints PASS when the interpreter behaves the way the skill says,
FAIL when it does not. Run it under each Python version you support; several
claims are version-dependent and this is the only way to keep them honest.

Usage:
    python scripts/verify_python_claims.py          # human-readable
    python scripts/verify_python_claims.py --json   # machine-readable
    python scripts/verify_python_claims.py --skip-timing   # CI: skip the two timing checks

Stdlib only. Exit code 1 if any check FAILs.
"""
from __future__ import annotations

import abc
import dataclasses
import functools
import json
import sys
import threading
import time
import timeit
import types
import weakref
from collections.abc import Container, Sized
from typing import NamedTuple, Protocol, runtime_checkable

RESULTS: list[dict[str, object]] = []


SKIP_TIMING = "--skip-timing" in sys.argv


def check(claim_id: str, where: str, claim: str, timing: bool = False):
    def deco(fn):
        if timing and SKIP_TIMING:
            return fn
        try:
            ok, observed = fn()
        except Exception as exc:  # a crash is a failed claim, not a crashed script
            ok, observed = False, f"raised {type(exc).__name__}: {exc}"
        RESULTS.append({"id": claim_id, "where": where, "claim": claim,
                        "ok": bool(ok), "observed": observed})
        return fn
    return deco


# --- Protocol vs ABC ------------------------------------------------------- #

@check("P1", "python-idioms.md#protocol-vs-abc",
       "Explicit Protocol subclass that implements nothing instantiates; method returns None")
def _():
    class P(Protocol):
        def m(self) -> int: ...
    class Impl(P):
        pass
    return Impl().m() is None, f"Impl().m() -> {Impl().m()!r}"


@check("P2", "python-idioms.md#the-practical-recommendation",
       "@abstractmethod inside a Protocol DOES enforce at instantiation for explicit subclasses")
def _():
    class P(Protocol):
        @abc.abstractmethod
        def m(self) -> int: ...
    class Impl(P):
        pass
    try:
        Impl()
    except TypeError as exc:
        return True, f"TypeError: {exc}"
    return False, "instantiated without error"


@check("P3", "structural.md#bridge + python-idioms.md#when-abc-is-the-right-call",
       "Explicit Protocol subclasses DO inherit default method bodies (code reuse is possible)")
def _():
    class Device(Protocol):
        def name(self) -> str: ...
        def describe(self) -> str:
            return f"device:{self.name()}"
    class Cam(Device):
        def name(self) -> str:
            return "cam"
    return Cam().describe() == "device:cam", f"Cam().describe() -> {Cam().describe()!r}"


@check("P4", "python-idioms.md#protocol-vs-abc",
       "@runtime_checkable isinstance ignores signatures")
def _():
    @runtime_checkable
    class Exporter(Protocol):
        def export(self, data: str) -> bytes: ...
    class Wrong:
        def export(self, a, b, c, d): ...
    return isinstance(Wrong(), Exporter), f"isinstance(Wrong(), Exporter) -> {isinstance(Wrong(), Exporter)}"


@check("P5", "python-idioms.md#protocol-vs-abc",
       "issubclass() raises TypeError for data protocols")
def _():
    @runtime_checkable
    class Named(Protocol):
        name: str
    try:
        issubclass(int, Named)
    except TypeError as exc:
        return True, f"TypeError: {exc}"
    return False, "no error"


@check("P6", "python-idioms.md#protocol-vs-abc",
       "CPython lets an undecorated sub-protocol of a runtime protocol pass isinstance "
       "(type checkers reject it, so decorate it anyway)")
def _():
    @runtime_checkable
    class A(Protocol):
        def a(self) -> None: ...
    class B(A, Protocol):
        def b(self) -> None: ...
    class OnlyA:
        def a(self) -> None: ...
    try:
        result = isinstance(OnlyA(), B)
    except TypeError as exc:
        return False, f"TypeError: {exc}"
    return result is False, f"isinstance(OnlyA(), B) -> {result} (checks both a and b)"


@check("P7", "python-idioms.md#protocol-vs-abc",
       "Runtime-Protocol isinstance is tens of x slower on 3.11, under 25x on 3.12+ (timing)",
       timing=True)
def _():
    @runtime_checkable
    class HasExport(Protocol):
        def export(self, data: str) -> bytes: ...
    class Plain:
        def export(self, data: str) -> bytes:
            return b""
    obj, n = Plain(), 50_000
    t_cls = min(timeit.repeat(lambda: isinstance(obj, Plain), number=n, repeat=5))
    t_pro = min(timeit.repeat(lambda: isinstance(obj, HasExport), number=n, repeat=5))
    ratio = t_pro / t_cls
    ok = ratio > 20 if sys.version_info < (3, 12) else ratio < 25
    return ok, f"ratio {ratio:.1f}x ({t_pro / n * 1e9:.0f} ns vs {t_cls / n * 1e9:.0f} ns per call)"


@check("A1", "python-idioms.md#when-abc-is-the-right-call",
       "ABC.register() asserts conformance without checking abstract methods")
def _():
    class Exporter(abc.ABC):
        @abc.abstractmethod
        def export(self) -> bytes: ...
    class Unrelated:
        pass
    Exporter.register(Unrelated)
    return issubclass(Unrelated, Exporter) and Unrelated() is not None, \
        f"issubclass(Unrelated, Exporter) -> {issubclass(Unrelated, Exporter)}"


# --- singledispatch -------------------------------------------------------- #

@check("S1", "python-idioms.md#functoolssingledispatch-the-pythonic-dispatch-by-type",
       "singledispatch requires a positional first argument")
def _():
    @functools.singledispatch
    def render(node) -> str:
        return "base"
    try:
        render(node=1)
    except TypeError as exc:
        return True, f"TypeError: {exc}"
    return False, "keyword call worked"


@check("S2", "python-idioms.md#functoolssingledispatch-the-pythonic-dispatch-by-type",
       "Ambiguous ABC registration raises RuntimeError")
def _():
    @functools.singledispatch
    def f(x) -> str:
        return "base"
    @f.register(Sized)
    def _(x) -> str:
        return "sized"
    @f.register(Container)
    def _(x) -> str:
        return "container"
    class Both:
        def __len__(self) -> int: return 0
        def __contains__(self, item) -> bool: return False
    try:
        f(Both())
    except RuntimeError as exc:
        return True, f"RuntimeError: {exc}"
    return False, f"dispatched to {f(Both())!r}"


@check("S3", "python-idioms.md#functoolssingledispatch-the-pythonic-dispatch-by-type",
       "Union annotations work with register() (3.11+)")
def _():
    @functools.singledispatch
    def f(x) -> str:
        return "base"
    @f.register
    def _(x: int | float) -> str:
        return "number"
    return f(1.5) == "number", f"f(1.5) -> {f(1.5)!r}"


# --- caching and lazy init ------------------------------------------------- #

def _hammer(fn, n: int = 8) -> None:
    ts = [threading.Thread(target=fn) for _ in range(n)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()


@check("C1", "creational.md#singleton-and-why-you-want-a-module"
       " + python-idioms.md#functoolscache-and-cached_property-traps",
       "@cache does NOT guarantee the wrapped function runs once under threads")
def _():
    calls = []
    @functools.cache
    def load():
        calls.append(1)
        time.sleep(0.02)
        return object()
    _hammer(load)
    return len(calls) > 1, f"loader ran {len(calls)} times for 8 threads"


@check("C2", "creational.md#singleton-and-why-you-want-a-module (PEP 562 '__getattr__ is cleaner')",
       "PEP 562 module __getattr__ is NOT run-once under threads either")
def _():
    mod = types.ModuleType("lazy_demo")
    calls: list[int] = []
    def _getattr(name):
        if name == "model":
            calls.append(1)
            time.sleep(0.02)
            value = object()
            mod.__dict__["model"] = value
            return value
        raise AttributeError(name)
    mod.__getattr__ = _getattr
    _hammer(lambda: mod.model)
    return len(calls) > 1, f"loader ran {len(calls)} times for 8 threads"


@check("C4", "creational.md#singleton-and-why-you-want-a-module (the locked loader)",
       "Double-checked locking loads exactly once under threads")
def _():
    calls: list[int] = []
    lock = threading.Lock()
    box: dict[str, object] = {}
    def get_model() -> object:
        if "model" not in box:
            with lock:
                if "model" not in box:
                    calls.append(1)
                    time.sleep(0.02)
                    box["model"] = object()
        return box["model"]
    _hammer(get_model)
    return len(calls) == 1, f"loader ran {len(calls)} time(s) for 8 threads"


@check("C3", "python-idioms.md#functoolscache-and-cached_property-traps",
       "@cache keeps arguments alive after del + gc")
def _():
    import gc
    class Arg:
        pass
    @functools.cache
    def f(a):
        return 1
    a = Arg()
    ref = weakref.ref(a)
    f(a)
    del a
    gc.collect()
    return ref() is not None, f"weakref alive after del+gc: {ref() is not None}"


# --- dataclasses and tuples ------------------------------------------------ #

@check("D1", "python-idioms.md#dataclass-gotchas",
       "A mutable-but-hashable default slips past the dataclass default check")
def _():
    class Box:  # mutable, hashable by identity
        def __init__(self) -> None:
            self.items: list[int] = []
    @dataclasses.dataclass
    class Holder:
        box: Box = Box()  # noqa: RUF009 -- the mistake this check demonstrates
    a, b = Holder(), Holder()
    a.box.items.append(1)
    return b.box.items == [1], f"second instance sees first's mutation: {b.box.items}"


@check("D2", "python-idioms.md#dataclass-gotchas",
       "eq=True, frozen=False makes instances unhashable")
def _():
    @dataclasses.dataclass
    class P:
        x: int
    try:
        hash(P(1))
    except TypeError as exc:
        return True, f"TypeError: {exc}"
    return False, "hashable"


@check("D3", "python-idioms.md#dataclass-gotchas",
       "slots=True breaks zero-arg super() before 3.13.14; fixed in 3.13.14 and 3.14.0")
def _():
    @dataclasses.dataclass(slots=True)
    class Base:
        x: int = 0
        def describe(self) -> str:
            return "base"
    @dataclasses.dataclass(slots=True)
    class Child(Base):
        def describe(self) -> str:
            return "child+" + super().describe()
    # Backported to 3.13 in 3.13.14 (3.13.13 still raises); 3.12 is security-only and keeps the bug.
    broken_expected = sys.version_info < (3, 13, 14)
    try:
        out = Child().describe()
    except TypeError as exc:
        return broken_expected, f"TypeError: {exc}"
    return not broken_expected, f"worked: {out!r}"


@check("D4", "creational.md#factory + more-patterns.md#value-object",
       "NamedTuple attribute reads are not faster than a dataclass; a slotted frozen dataclass "
       "is smaller than a NamedTuple and reads faster (timing)",
       timing=True)
def _():
    class NT(NamedTuple):
        a: int
        b: int
    @dataclasses.dataclass
    class DC:
        a: int
        b: int
    @dataclasses.dataclass(frozen=True, slots=True)
    class DS:
        a: int
        b: int
    nt, dc, ds = NT(1, 2), DC(1, 2), DS(1, 2)
    def per_read(x: object) -> float:
        return min(timeit.repeat("x.a", globals={"x": x}, number=1_000_000, repeat=5)) * 1e3
    t_nt, t_dc, t_ds = per_read(nt), per_read(dc), per_read(ds)
    size_nt, size_ds = sys.getsizeof(nt), sys.getsizeof(ds)
    ok = t_nt >= t_dc * 0.9 and t_ds < t_nt and size_ds < size_nt
    return ok, (f"read ns: namedtuple {t_nt:.1f}, dataclass {t_dc:.1f}, slotted {t_ds:.1f}; "
                f"bytes: namedtuple {size_nt}, slotted {size_ds}")


@check("K1", "python-idioms.md#functoolscache-and-cached_property-traps",
       "cached_property runs once under threads before 3.12, can run repeatedly on 3.12+")
def _():
    calls: list[int] = []
    class Model:
        @functools.cached_property
        def weights(self):
            calls.append(1)
            time.sleep(0.02)
            return object()
    m = Model()
    _hammer(lambda: m.weights)
    expect_many = sys.version_info >= (3, 12)
    return (len(calls) > 1) == expect_many, f"getter ran {len(calls)} times for 8 threads"


def main() -> int:
    as_json = "--json" in sys.argv
    version = sys.version.split()[0]
    if as_json:
        print(json.dumps({"python": version, "results": RESULTS}, indent=2))
    else:
        print(f"Python {version}")
        for r in RESULTS:
            mark = "PASS" if r["ok"] else "FAIL"
            print(f"  {mark}  {r['id']:3s} {r['claim']}")
            print(f"              observed: {r['observed']}")
        fails = [r["id"] for r in RESULTS if not r["ok"]]
        print(f"  {len(RESULTS) - len(fails)}/{len(RESULTS)} claims hold"
              + (f"; failing: {', '.join(fails)}" if fails else ""))
    return 1 if any(not r["ok"] for r in RESULTS) else 0


if __name__ == "__main__":
    sys.exit(main())
