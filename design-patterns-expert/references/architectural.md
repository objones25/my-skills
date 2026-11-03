# Architectural patterns

These are the expensive ones. Every entry leads with the published warning, because the failure mode here is adopting them on a system that doesn't need them, and unlike a misplaced Strategy, these are hard to back out of.

Contents: [Repository](#repository) · [Unit of Work](#unit-of-work) · [Service Layer](#service-layer) · [Domain Events & message bus](#domain-events--message-bus) · [Event Sourcing](#event-sourcing) · [CQRS](#cqrs) · [Transactional Outbox](#transactional-outbox) · [Saga](#saga) · [Strangler Fig](#strangler-fig)

---

## Repository

An abstraction over persistent storage that *"hides the boring details of data access by pretending that all of our data is in memory."* Minimum surface: `add()` and `get()`.

```python
class OrderRepository(Protocol):
    def add(self, order: Order) -> None: ...
    def get(self, reference: str) -> Order: ...

class FakeOrderRepository:              # the test double, ~6 lines
    def __init__(self, orders: list[Order] | None = None) -> None:
        self._orders = {o.reference: o for o in orders or []}
    def add(self, order: Order) -> None:
        self._orders[order.reference] = order
    def get(self, reference: str) -> Order:
        return self._orders[reference]
```

**Buys:** the domain model has no persistence concerns; tests run without a database; you control the object-to-table mapping.

**Costs, stated by the authors themselves:** *"An ORM already buys you some decoupling."* · *"Maintaining ORM mappings by hand requires extra work and extra code."* · *"Any extra layer of indirection always increases maintenance costs and adds a 'WTF factor' for Python programmers who've never seen the Repository pattern before."*

**When not to:** *"If your app is just a simple CRUD (create-read-update-delete) wrapper around a database, then you don't need a domain model or a repository."*

**Scope discipline from Evans:** *"Provide repositories only for aggregate roots that actually need direct access."* A repository per table is the standard over-application; it recreates the ORM with more code.

> Sources: [cosmicpython ch. 2](https://www.cosmicpython.com/book/chapter_02_repository.html) · [Evans, DDD Reference](https://www.domainlanguage.com/wp-content/uploads/2016/05/DDD_Reference_2015-03.pdf)

---

## Unit of Work

An abstraction over *atomic operations*: one entry point to storage that groups repository work into a transaction, usually as a context manager.

```python
with uow:
    order = uow.orders.get(reference)
    order.allocate(line)
    uow.commit()          # explicit; no commit means rollback
```

**Buys:** the atomic boundary is visible in the code; failure defaults to safe (no commit → rollback); a natural home for repositories; a place to collect domain events for publishing after a successful commit.

**Costs, again from the authors:** *"Your ORM probably already has some perfectly good abstractions around atomicity. SQLAlchemy even has context managers."* · *"We've made it look easy, but you have to think quite carefully about things like rollbacks, multithreading, and nested transactions."*

SQLAlchemy's `Session` *is* a Unit of Work. Wrapping it is justified only when you want a narrower, intention-revealing surface, not by default.

> Source: [cosmicpython ch. 6](https://www.cosmicpython.com/book/chapter_06_uow.html)

---

## Service Layer

The layer that *drives* the application: fetch from storage, invoke the domain model, persist changes. Orchestration, not rules.

Two things get called "service" and confusing them causes real damage:
- **Application service** (the service layer): handles a request from outside and orchestrates an operation.
- **Domain service**: a piece of business logic that belongs in the domain but doesn't sit naturally on any single entity or value object. This lives *inside* the domain.

**When to introduce it:** *"It's better to introduce this layer after you spot orchestration logic creeping into your controllers."* Not before. If you only ever have one web app, controllers already give you a single place for use cases.

**The specific risk:** *"Putting too much logic into the service layer can lead to the Anemic Domain antipattern."* Fowler's version is blunter: *"If all your logic is in services, you've robbed yourself blind"*, and *"In essence the problem with anemic domain models is that they incur all of the costs of a domain model, without yielding any of the benefits."*

Note that Fowler's escape hatch is legitimate: if all your logic really is procedural orchestration, you have chosen **Transaction Script**, which is a fine pattern. The error is paying domain-model costs while getting Transaction Script benefits: anemic entities plus fat services plus repositories plus a UoW is the worst of both.

**For legacy code, this is the recommended first move.** Asked whether the patterns must be adopted all at once, the cosmicpython authors say no: *"If you have an existing system, we recommend building a service layer to try to keep orchestration in one place."*

> Sources: [cosmicpython ch. 4](https://www.cosmicpython.com/book/chapter_04_service_layer.html) · [cosmicpython epilogue](https://www.cosmicpython.com/book/epilogue_1_how_to_get_there_from_here.html) · [Fowler, Anemic Domain Model](https://martinfowler.com/bliki/AnemicDomainModel.html)

---

## Domain Events & message bus

Facts about things that have happened, dispatched to handlers through a dict from event type to handlers. The discovery heuristic is linguistic: when a domain expert says *"when X, then Y"*, X is usually an event.

The trade-offs here are unusually well documented and worth reading in full before adopting:

- *"The message bus is an additional thing to wrap your head around; the implementation in which the unit of work raises events for us is neat but also magic."*
- *"It's not obvious when we call `commit` that we're also going to go and send email to people."*
- *"[T]hat hidden event-handling code executes synchronously, meaning your service-layer function doesn't finish until all the handlers for any events are finished."*
- *"[E]vent-driven workflows can be confusing because after things are split across a chain of multiple handlers, there is no single place in the system where you can understand how a request will be fulfilled."*
- *"You also open yourself up to the possibility of circular dependencies between your event handlers, and infinite loops."*
- Duplication: *"There will be duplication of fields and structure between model objects and events, which will have a maintenance cost."*

The cosmicpython authors take this from Fowler, whose original reads: *"it's very easy to make nicely decoupled systems with event notification, without realizing that you're losing sight of that larger-scale flow, and thus set yourself up for trouble in future years."*

**Mitigations that actually work:** a documented event catalog; correlation IDs threaded through handlers and logs; a test that asserts the handler graph is acyclic; and an explicit decision about whether handlers run inline or on a queue.

> Sources: [cosmicpython ch. 8](https://www.cosmicpython.com/book/chapter_08_events_and_message_bus.html), [ch. 9](https://www.cosmicpython.com/book/chapter_09_all_messagebus.html) · [Fowler, What do you mean by "Event-Driven"?](https://martinfowler.com/articles/201701-event-driven.html)

---

## Event Sourcing

Store every state change as an immutable event; compute current state by replaying the log.

```python
@dataclass(frozen=True)
class Event(Generic[T]):
    type: EventType
    data: T
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class EventStore(Generic[T]):
    events: list[Event[T]] = field(default_factory=list)
    def append(self, event: Event[T]) -> None:
        self.events.append(event)
    def get_all(self) -> list[Event[T]]:
        return list(self.events)
```

Derived state is a cache over the log, invalidated on append. The same log supports arbitrary projections ("items added on date D", "balance as of last quarter") without changing how data is stored, which is the real payoff.

### Read Fowler's caveats before recommending it

- **Replay hits external systems.** *"[I]f these events cause update messages to be sent to external systems, then things will go wrong because those external systems don't know the difference between real processing and replays."* You need gateways that know they're in replay context. *"Both queries and updates to external systems cause a lot of complication with Event Sourcing."*
- **External queries aren't reproducible.** Replaying a December 5th event on December 20th needs December 5th's exchange rate. You must log external *responses*, not just your own events.
- **Code changes break replay, and bug fixes at external gateways are the hard case.** New features usually survive reprocessing. An internal bug fix is easy: fix the code and reprocess the events. The hard case is external systems, where the gateways *"need to track the difference between what happened with the bug, and what happens without it."* Temporal business rules *"will get messy if you have much temporal logic"*, and bi-temporal cases get *"very messy"*.
- **Reversal is easy only for difference-style events.** If an event records a new value rather than a change, it must also store the prior value so it can be reversed. Fowler's fallback: revert to a snapshot and replay forward.
- **Growth.** Rebuilding from empty *"is a slow process, particularly if there are many events"*: hence snapshots/checkpoints, exactly as in the Command ledger.
- **Adopt early if you will need its payoffs.** Fowler: *"it's very hard to retrofit these patterns onto a system that wasn't built with Event Sourcing"*, where "these patterns" are Parallel Models and Retroactive Events. If you may need them later, build Event Sourcing now.
- **If you only want an audit log, don't.** *"you could also do this with more regular logging mechanisms, and that way not have to deal with the odd interface."* Fowler calls Event Sourcing a good way to give a support desk an audit trail, but notes ordinary logging does the job without the odd interface. That covers most requests that are really for an audit log.

One correction to common folklore: Fowler's page does **not** discuss event *schema* versioning. His versioning discussion is about code evolving over time. Schema evolution is a real problem you will still have to solve yourself (upcasters, versioned event types), just don't attribute the guidance to him.

> Source: [Fowler, Event Sourcing](https://martinfowler.com/eaaDev/EventSourcing.html)

---

## CQRS

Use a different model to write than to read.

**Lead with the warning, every time.** Fowler: *"[F]or most systems CQRS adds risky complexity."* And: *"While I have come across successful uses of CQRS, so far the majority of cases I've run into have not been so good, with CQRS seen as a significant force for getting a software system into serious difficulties."* And the scoping rule: *"CQRS should only be used on specific portions of a system (a BoundedContext in DDD lingo) and not the system as a whole."*

**Escalate only as far as the pain requires.** The cosmicpython graded options, cheapest first:

| Option | Pro | Con |
|---|---|---|
| Just use repositories | Simple, consistent | Performance issues with complex query patterns |
| Custom ORM queries | Reuses existing config and models | Another query language with its own quirks |
| Hand-rolled SQL | Fine control over performance | Schema changes must land in two places |
| Denormalized read tables | Much faster to query | Slows writes slightly |
| Separate read store fed by events | Read replicas scale out easily | *"Complex technique. Harry will be forever suspicious of your tastes and motives."* |

Most "we need CQRS" conversations are solved by the first two rows. Their default: *"Often, your read operations will be acting on the same conceptual objects as your write model, so using the ORM… is just fine."*

**The consistency point people miss:** *"all distributed systems are inconsistent. As soon as you have a web server and two customers, you have the potential for stale data."* CQRS makes an existing inconsistency explicit; it doesn't introduce one. But if a user must read their own write immediately and exactly, a lagging projection will produce support tickets: design for it (read-your-writes routing, or return the computed result from the write).

> Sources: [Fowler, CQRS](https://martinfowler.com/bliki/CQRS.html) · [cosmicpython ch. 12](https://www.cosmicpython.com/book/chapter_12_cqrs.html)

---

## Transactional Outbox

Solves the **dual-write problem**: an operation that must both change the database and publish an event, where doing them separately can leave the system inconsistent in either direction, DB committed but event lost (downstream never learns), or event published but DB rolled back (downstream acts on a change that didn't happen).

The fix: write the event into an `outbox` table **in the same transaction** as the business change. A relay process then publishes from that table: either by polling and deleting on success, or via change-data-capture on the outbox table.

```python
with session.begin():
    session.add(order)
    session.add(OutboxRow(topic="orders", payload=json.dumps(event), created_at=now()))
# a separate relay reads unpublished rows, publishes, marks them sent
```

**Non-negotiable consequences:**
- **Delivery is at-least-once, not exactly-once.** *"The events processing service might send out duplicate messages or events, so we recommend that you make the consuming service idempotent by tracking the processed messages."* An outbox without an idempotent consumer is only half the pattern.
- **Preserve order**: publish in the order the writes happened, particularly if consumers are event-sourced.
- Never publish for a rolled-back transaction (the same-transaction write gives you this for free).
- Spanning services means you also need **Saga**.

**When not to** (this skill's judgment; the AWS page lists only when to use it): no event dependency; no need for guaranteed delivery; or the system requires immediate synchronous consistency and cannot tolerate duplicates.

> Source: [AWS, Transactional outbox](https://docs.aws.amazon.com/prescriptive-guidance/latest/cloud-design-patterns/transactional-outbox.html)

---

## Saga

A sequence of local transactions across services, each with a compensating transaction, replacing a distributed ACID transaction that isn't available.

**Classify every step**. This drives the whole design:
- **Compensable**: can be semantically undone.
- **Pivot**: the point of no return. After it succeeds, only retryable steps may follow.
- **Retryable**: idempotent and guaranteed to eventually succeed.

**Choreography** (services react to each other's events) suits short workflows: no coordinator, no single point of failure, but Microsoft notes the workflow *"can be confusing when you add new steps"*, cyclic dependencies become a risk, and integration testing needs everything running. **Orchestration** (a central coordinator) suits complex workflows: explicit flow, no cycles, at the cost of a component to build and a single point of failure.

**The failure everyone underestimates: sagas have no isolation.** No "I" in ACID. Documented anomalies: lost updates (one saga overwrites another's change), dirty reads (reading a saga's uncommitted state), and non-repeatable reads between steps. Countermeasures exist: semantic locks, commutative updates, reordering so risky updates land in the retryable phase, re-read before write, version fields: but they are design work you must do, not properties you get.

Other documented hazards: irreversible steps can't be rolled back; **compensating transactions can themselves fail**, leaving the system inconsistent; debugging complexity scales with participant count.

**When not to:** tightly coupled transactions, compensations that would have to reach back into earlier participants, or cyclic dependencies between services.

In Python, `temporalio` is the strongest option for orchestrated sagas: durable workflows, activities with retry policies and timeouts, and state that survives process restarts. Compensation is implemented by catching activity failures and running compensating activities; there is no formally named "saga" API in the Python SDK. Temporal's own saga page shows the Python shape: register each compensation in a list before its forward activity, and on failure run them in `reversed()` order, then re-raise (Java, by contrast, has a `Saga` helper) ([Temporal, Saga Pattern](https://docs.temporal.io/design-patterns/saga-pattern)).

> Source: [Azure, Saga](https://learn.microsoft.com/en-us/azure/architecture/patterns/saga)

---

## Strangler Fig

Replace a legacy system incrementally: put a façade in front, move one capability at a time to the new system, retire the old one when nothing routes to it.

**Why not a rewrite:** long timelines mean users wait for features the whole time; existing behavior is hard to specify precisely; and much of the old behavior is unwanted, so rebuilding it faithfully is waste.

**The hard part is seams.** Legacy systems rarely have clean component boundaries: well-factored legacy systems are, in Fowler's phrase, unicorns. Finding a boundary where old and new can coexist is most of the work.

**Transitional architecture is a real cost, and it is worth paying.** People balk at building code that exists only to let the two systems coexist and will be deleted afterwards. Fowler's position is that the reduced risk and earlier value justify it.

**Operational hazards:** the façade must not become a single point of failure or a bottleneck; it must keep pace with the migration; shared data stores must be accessible from both systems concurrently; and cross-system dependencies need an **Anti-Corruption Layer**, or the new system ends up adopting legacy conventions, which defeats the migration.

**Design the new system to be stranglable in turn:** *"strive to have clear demarcations between parts of your solution so that you can migrate each part individually."*

**The organizational point that is usually skipped:** *"Legacy systems become rigid and brittle because the design thinking and organizational processes that produced them built them that way."* Without changing how the team works, the replacement arrives at the same place.

**When not to:** you can't intercept requests to the back end; you can't access the legacy source (you need it to disable migrated features and redirect internal calls); the system is small enough to replace wholesale; or you must decommission quickly.

> Sources: [Fowler, StranglerFigApplication](https://martinfowler.com/bliki/StranglerFigApplication.html) · [Azure, Strangler Fig](https://learn.microsoft.com/en-us/azure/architecture/patterns/strangler-fig)

---

## Getting there from here

For a user staring at an existing system, the sequencing advice from the cosmicpython epilogue is the most practical guidance in this file:

- **Start from pain, not from patterns.** *"[W]hat problem are you trying to solve? Is the software too hard to change? Is the performance unacceptable? Have you got weird, inexplicable bugs?"*
- **Fund it with feature work.** Structural change is an easier sell attached to something the business wants.
- **Start from use cases**, and tolerate mess while extracting them: *"It's fine if you have duplication in the use-case functions. We're not trying to write perfect code; we're just trying to extract some meaningful layers."*
- **Adopt incrementally.** Service layer first on legacy code; the rest later, if at all.
- **Don't boil the ocean.** *"Just look for somewhere small to start."*
- And the line to quote at anyone treating this as doctrine: *"The techniques we're presenting in this book are intended to make your life easier. They're not some kind of ascetic discipline with which to punish yourself."*

> Source: [cosmicpython epilogue](https://www.cosmicpython.com/book/epilogue_1_how_to_get_there_from_here.html)
