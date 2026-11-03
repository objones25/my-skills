"""State machine as a transition table, with the decorator defined as a real method.

Fixes the reference snippet in behavioral.md, where `transition` was shown as a
module-level function taking `self`. Toy domain: an order's lifecycle.
Run: python examples/state_machine.py
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum, auto


class Status(Enum):
    PENDING = auto()
    PAID = auto()
    SHIPPED = auto()
    CANCELLED = auto()


class Event(Enum):
    PAY = auto()
    SHIP = auto()
    CANCEL = auto()


@dataclass
class OrderCtx:
    order_id: str
    log: list[str] = field(default_factory=list)


Action = Callable[[OrderCtx], None]


class IllegalTransition(Exception):
    pass


@dataclass
class StateMachine:
    table: dict[tuple[Status, Event], tuple[Status, Action]] = field(default_factory=dict)

    def transition(self, sources: Status | Iterable[Status], event: Event,
                   target: Status) -> Callable[[Action], Action]:
        srcs = (sources,) if isinstance(sources, Status) else tuple(sources)
        def register(fn: Action) -> Action:
            for s in srcs:
                if (s, event) in self.table:
                    raise ValueError(f"duplicate transition {s.name} --{event.name}-->")
                self.table[(s, event)] = (target, fn)
            return fn
        return register

    def fire(self, ctx: OrderCtx, state: Status, event: Event) -> Status:
        try:
            target, action = self.table[(state, event)]
        except KeyError:
            raise IllegalTransition(f"{state.name} --{event.name}--> (no such transition)") from None
        action(ctx)              # action first: if it raises, the state does not advance
        return target


machine = StateMachine()


@machine.transition(Status.PENDING, Event.PAY, Status.PAID)
def on_pay(ctx: OrderCtx) -> None:
    ctx.log.append("charged card")


@machine.transition(Status.PAID, Event.SHIP, Status.SHIPPED)
def on_ship(ctx: OrderCtx) -> None:
    ctx.log.append("label printed")


@machine.transition([Status.PENDING, Status.PAID], Event.CANCEL, Status.CANCELLED)
def on_cancel(ctx: OrderCtx) -> None:
    ctx.log.append("cancelled")


if __name__ == "__main__":
    ctx, state = OrderCtx("A1"), Status.PENDING
    for ev in (Event.PAY, Event.SHIP):
        state = machine.fire(ctx, state, ev)
        print(f"{ev.name:6s} -> {state.name:9s} log={ctx.log}")
    try:
        machine.fire(ctx, state, Event.CANCEL)
    except IllegalTransition as exc:
        print("rejected:", exc)
    print("table has", len(machine.table), "transitions")
