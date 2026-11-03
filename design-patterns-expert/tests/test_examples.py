"""The examples are part of the skill's claims, so they are tested like code."""
from __future__ import annotations

import asyncio
import random
import shutil
import sqlite3
import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.mark.parametrize("script", sorted(p.name for p in EXAMPLES.glob("*.py")))
def test_every_example_runs(script: str) -> None:
    subprocess.run([sys.executable, str(EXAMPLES / script)], check=True, capture_output=True, timeout=60)


# ---------------------------------------------------------------- strategy

def test_strategy_rungs_share_one_contract() -> None:
    import strategy_ladder as sl
    exporters = sl.build_exporters()
    assert all(callable(e) for e in exporters.values())
    assert ",0\n" in sl.run_export(sl.SAMPLE, exporters["csv-public"])
    batched = sl.BatchedExporter(sl.export_json, batch_size=2)
    sl.run_export(sl.SAMPLE, batched)
    assert batched.batches_written == 2


# ----------------------------------------------------------- state machine

def test_state_machine_legal_illegal_and_duplicate() -> None:
    import state_machine as smod
    ctx = smod.OrderCtx("A1")
    state = smod.machine.fire(ctx, smod.Status.PENDING, smod.Event.PAY)
    assert state is smod.Status.PAID and ctx.log == ["charged card"]
    with pytest.raises(smod.IllegalTransition):
        smod.machine.fire(ctx, smod.Status.SHIPPED, smod.Event.CANCEL)
    with pytest.raises(ValueError, match="duplicate"):
        smod.machine.transition(smod.Status.PENDING, smod.Event.PAY, smod.Status.PAID)(lambda c: None)


# ------------------------------------------------- ports & adapters (contract)

def _sqlite_inventory(stock: dict[str, int]):
    from shop.adapters.sqlite import SqliteInventory
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE stock (sku TEXT PRIMARY KEY, qty INTEGER NOT NULL)")
    conn.executemany("INSERT INTO stock VALUES (?, ?)", stock.items())
    return SqliteInventory(conn)


def _memory_inventory(stock: dict[str, int]):
    from shop.adapters.memory import InMemoryInventory
    return InMemoryInventory(stock)


@pytest.mark.parametrize("make", [_memory_inventory, _sqlite_inventory], ids=["memory", "sqlite"])
def test_inventory_adapters_honour_the_same_contract(make) -> None:
    from shop.domain import OrderRequest, OutOfStock, UnknownSKU, place_order
    inv = make({"MUG": 5, "TEE": 1})
    assert place_order(OrderRequest("MUG", 2), inv).remaining == 3
    with pytest.raises(OutOfStock):
        place_order(OrderRequest("TEE", 3), inv)
    with pytest.raises(UnknownSKU):
        place_order(OrderRequest("CAP", 1), inv)


@pytest.mark.skipif(shutil.which("lint-imports") is None, reason="import-linter not installed")
def test_import_linter_contract_is_kept() -> None:
    root = EXAMPLES / "ports_adapters"
    out = subprocess.run(["lint-imports", "--no-cache"], cwd=root, capture_output=True, text=True,
                         env={"PYTHONPATH": str(root), "PATH": "/usr/bin:/usr/local/bin:/bin"})
    assert "1 kept, 0 broken" in out.stdout, out.stdout + out.stderr


# ------------------------------------------------------------------- retry

def test_budget_bounds_attempts_during_an_outage() -> None:
    import retry_budget as rb
    assert rb.simulate(rb.outage, 1000, None)["attempts"] == 3000
    s = rb.simulate(rb.outage, 1000, rb.RetryBudget())
    assert s["attempts"] == 1100 and s["refused_by_budget"] == 950


def test_permanent_errors_are_not_retried_and_retry_after_is_honoured() -> None:
    import retry_budget as rb
    stats: dict[str, int] = {}
    def bad() -> str:
        raise rb.Permanent("422")
    with pytest.raises(rb.Permanent):
        rb.call_with_retry(bad, budget=None, clock=rb.Clock(), rng=random.Random(0), stats=stats)
    assert stats["attempts"] == 1

    calls = iter([rb.Throttled(retry_after=2.0), None])
    def throttled_once() -> str:
        exc = next(calls)
        if exc:
            raise exc
        return "ok"
    clock = rb.Clock()
    assert rb.call_with_retry(throttled_once, budget=rb.RetryBudget(), clock=clock, rng=random.Random(0)) == "ok"
    assert clock.now() >= 2.0


def test_retry_stops_at_the_deadline() -> None:
    import retry_budget as rb
    stats: dict[str, int] = {}
    def slow_throttle() -> str:
        raise rb.Throttled(retry_after=30.0)
    with pytest.raises(rb.Throttled):
        rb.call_with_retry(slow_throttle, budget=None, clock=rb.Clock(), rng=random.Random(0),
                           deadline_s=10.0, stats=stats)
    assert stats["attempts"] == 1


def test_success_after_retry_refunds_the_budget() -> None:
    import retry_budget as rb
    budget = rb.RetryBudget()
    outcomes = iter([rb.Transient("timeout"), None])
    def once() -> str:
        exc = next(outcomes)
        if exc:
            raise exc
        return "ok"
    rb.call_with_retry(once, budget=budget, clock=rb.Clock(), rng=random.Random(0))
    assert budget.tokens == budget.capacity


# ------------------------------------------------------------- idempotency

def test_idempotency_semantics() -> None:
    import idempotency as idem
    store, gw = idem.IdempotencyStore(), idem.Gateway()
    body: dict[str, object] = {"order_id": "A1", "amount_cents": 2400}
    first = store.handle("t1", "k", body, gw.charge)
    assert [store.handle("t1", "k", body, gw.charge) for _ in range(3)] == [first] * 3
    assert len(gw.charges) == 1
    assert store.handle("t1", "k", {**body, "amount_cents": 1}, gw.charge)[0] == 422
    assert store.handle("t1", None, body, gw.charge)[0] == 400
    store.handle("t2", "k", body, gw.charge)
    assert len(gw.charges) == 2                      # keys are scoped per tenant


def test_concurrent_retry_gets_409_and_crash_stays_retryable() -> None:
    import idempotency as idem
    store = idem.IdempotencyStore()
    body: dict[str, object] = {"amount_cents": 1}
    seen: list[int] = []
    def charge_with_concurrent_retry(b: dict[str, object]) -> idem.Response:
        seen.append(store.handle("t", "k", b, charge_with_concurrent_retry)[0])
        return 201, {}
    store.handle("t", "k", body, charge_with_concurrent_retry)
    assert seen == [409]

    def crash(b: dict[str, object]) -> idem.Response:
        raise RuntimeError("process died")
    with pytest.raises(RuntimeError):
        store.handle("t", "k2", body, crash)
    assert store.handle("t", "k2", body, lambda b: (201, {}))[0] == 201


# ------------------------------------------------- parse, don't validate

def test_parsing_collects_errors_and_rejects_bools() -> None:
    import parse_dont_validate as pdv
    req = pdv.parse_order({"sku": "MUG", "quantity": 2})
    assert pdv.line_total(req) == pdv.Money(2400)
    with pytest.raises(pdv.ParseError) as exc:
        pdv.parse_order({"sku": "mug", "quantity": True})
    assert set(exc.value.errors) == {"sku", "quantity"}
    with pytest.raises(ValueError, match="currency"):
        pdv.Money(1, "USD") + pdv.Money(1, "EUR")


# ------------------------------------------------------- functional core

def test_core_is_pure_and_shell_executes() -> None:
    import functional_core as fc
    levels = [fc.StockLevel("MUG", 3, 5, 20), fc.StockLevel("CAP", 0, 2, 10)]
    tuesday = fc.plan_restock(levels, date(2026, 10, 6), frozenset({"CAP"}))
    assert tuesday == [fc.PlacePurchaseOrder("MUG", 20)]
    monday = fc.plan_restock(levels, date(2026, 10, 5), frozenset())
    assert monday[-1] == fc.NotifyBuyer("2 purchase order(s) placed")
    outbox: list[str] = []
    fc.Shell({lv.sku: lv for lv in levels}, set(), outbox).run(date(2026, 10, 6))
    assert outbox == ["PO MUG x20", "PO CAP x10"]


# --------------------------------------------------------- match dispatch

def test_match_dispatch_and_singledispatch() -> None:
    import match_dispatch as md
    log = [md.OrderPlaced("A1", 2400), md.OrderRefunded("A1", 600), md.OrderShipped("A1", "UPS")]
    assert sum(md.revenue_delta(e) for e in log) == 1800
    with pytest.raises(TypeError):
        md.describe("not an event")


@pytest.mark.skipif(shutil.which("mypy") is None, reason="mypy not installed")
def test_type_checker_catches_a_missing_case(tmp_path: Path) -> None:
    src = (EXAMPLES / "match_dispatch.py").read_text().replace(
        "OrderEvent = OrderPlaced | OrderRefunded | OrderShipped",
        "@dataclass(frozen=True, slots=True)\nclass OrderCancelled:\n    order_id: str\n\n\n"
        "OrderEvent = OrderPlaced | OrderRefunded | OrderShipped | OrderCancelled")
    target = tmp_path / "missing_case.py"
    target.write_text(src)
    out = subprocess.run(["mypy", "--strict", "--cache-dir", str(tmp_path / ".mypy_cache"), str(target)],
                         capture_output=True, text=True, cwd=tmp_path)
    assert out.returncode == 1 and "assert_never" in out.stdout


# ------------------------------------------------------ chain vs pipeline

def test_chain_exits_early_and_pipeline_is_lazy() -> None:
    import pipeline_vs_chain as pc
    trace: list[str] = []
    assert pc.decide(pc.Refund("A1", 100), [pc.auto_approve, pc.team_lead, pc.finance], trace) == "auto-approved"
    assert trace == ["auto_approve"]
    pulled = 0
    def source():
        nonlocal pulled
        for i in range(10_000):
            pulled += 1
            yield pc.Order(f"X{i}", "MUG", 3)
    next(pc.pipeline(source()))
    assert pulled == 1


# --------------------------------------------------------------- hedging

def test_hedged_call_sends_one_copy_when_fast_two_when_slow() -> None:
    import hedged_request as hr
    cancelled: list[int] = []
    def make(delays: dict[int, float]):
        async def call(replica: int) -> str:
            try:
                await asyncio.sleep(delays[replica])
            except asyncio.CancelledError:
                cancelled.append(replica)
                raise
            return f"r{replica}"
        return call
    assert asyncio.run(hr.hedged_call(make({0: 0.001, 1: 0.001}), 0.05)) == ("r0", 1)
    assert asyncio.run(hr.hedged_call(make({0: 0.5, 1: 0.005}), 0.01)) == ("r1", 2)
    assert cancelled == [0]


# ------------------------------------------------- sentinel / null object

def test_sentinel_distinguishes_none_from_missing() -> None:
    import null_object_sentinel as no
    assert no.discount_percent("A2") is None
    assert no.discount_percent("ZZ", default=0) == 0
    with pytest.raises(KeyError):
        no.discount_percent("ZZ")
    assert no.apply_discount("A1", 2400, no.NullAuditLog()) == 2160
