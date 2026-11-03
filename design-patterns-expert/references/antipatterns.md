# How patterns fail

The catalog of ways pattern application goes wrong. Read this when a design "looks right" but feels heavy, or before recommending anything from `architectural.md`.

Contents: [The over-engineering checklist](#the-over-engineering-checklist) · [Per-pattern failure modes](#per-pattern-failure-modes) · [Smells that look structural but aren't](#smells-that-look-structural-but-arent) · [Reviewing a design](#reviewing-a-design)

---

## The over-engineering checklist

Run this against any design that has more than one pattern in it.

**1. Can you name the change each abstraction buys?** Not "flexibility", a specific change: "adding a payment provider", "swapping Postgres for DynamoDB", "running the domain logic in a batch job". An abstraction whose purchase you cannot name is paying carrying cost for nothing.

**2. Has that change ever happened, or is it scheduled?** One implementation of an interface, with no second one planned, is a wrapper. Fowler's four costs of a presumptive feature apply: build, delay, carry, repair. The carry cost is the sneaky one (*"adds some complexity to the software, this complexity makes it harder to modify and debug"*) because it is paid on every future edit by every future reader.

**3. How many files must be opened to trace one behavior end to end?** Three is usually fine. Seven is a signal. If you can't trace a request through the system in a reasonable time, the decoupling has cost more than it bought.

**4. Does the test suite need infrastructure to test logic?** If yes, the boundary is in the wrong place: that is a genuine reason to add structure.

**5. Could a dict or a function do it?** A dict of functions replaces Strategy, Registry, Factory, State and Command in their simplest forms. Reach past it only for a stated reason.

**6. Is this pattern being applied to the whole system when it belongs to one part?** The most common architectural error. Fowler on CQRS: it *"should only be used on specific portions of a system... and not the system as a whole."* The same scoping applies to Event Sourcing, hexagonal layering, and the message bus.

**7. Is the pain the pattern addresses the pain the user actually described?** Restructuring rarely fixes a correctness bug or a missing metric. Say so instead of restructuring.

The counter-check, so this doesn't become an excuse for mess: *"Yagni is not a justification for neglecting the health of your code base. Yagni requires (and enables) malleable code."* Declining a pattern never justifies a 200-line function, duplicated business rules, or logic tangled with I/O.

---

## Per-pattern failure modes

### Strategy
- Passing settings through the caller as `**kwargs` or a shared parameters object. This puts implementation details of specific strategies back into the calling code and undoes the decoupling entirely: the single most common way Strategy is broken.
- A "strategy" that reaches into the caller's internals to do its job.
- Only one implementation, ever.

### Policy
- Silent order dependence. Rules that must run in sequence, ordered only by however the list was written.
- Policies that both mutate and veto without the caller being able to tell which happened.

### Registry
- **Import order becomes semantics.** A plugin module that is never imported is never registered, and the failure is silence, not an error. Load explicitly at startup; log what registered.
- Silent overwrite when two plugins claim the same key. Raise instead.
- No way to enumerate what's registered, so no one can tell what the system can do by reading it.
- Used for two entries that live in the same file.

### Observer / event bus
- **The flow disappears.** Fowler warns of *"losing sight of that larger-scale flow"*, and cosmicpython (which quotes the same Fowler article) puts it as *"there is no single place in the system where you can understand how a request will be fulfilled."*
- Unclear failure semantics: one handler raising either silently breaks a side effect or breaks the primary operation. Decide which, and log the other.
- Handlers running synchronously inside what looks like a fast function.
- Circular handler dependencies and infinite loops.
- Order-dependent listeners.

### Command
- Undo that can itself fail, with no plan for it. You cannot generally undo an undo; for money, escalate to a human rather than retrying.
- Commands holding live connections or sessions, so they can't be queued, serialized, or replayed.
- Keeping both a mutable state field and a command history without deciding which is the truth. Two sources of truth that can disagree is worse than either alone.

### State machine
- Guard conditions creeping into actions until the transition table no longer describes the machine.
- Side effects that run before the state change on some transitions and after on others.

### Factory
- Used where arbitrary combinations are wanted. If you'd need a factory class per combination, you wanted dependency injection.
- Factory classes that do work beyond construction.
- Undocumented lifetime: the caller owns what the factory returns; say so when it matters.

### Builder
- Two types where a dataclass with keyword defaults would do. Check that first: Python's optional arguments cover most builder use cases.
- A forgotten `.build()`. Type-checkable if the product and builder types are distinct and the variable is annotated.
- Builders that validate nothing, producing an invalid product that fails far from the mistake.

### Singleton
- Any implementation of it. It destroys test isolation by design, which a module-level object does not; `Logger()` reads as construction and isn't; inheritance breaks the single-instance guarantee; and the dependency appears in no signature.

### Object Pool
- **Not resetting on release**: the next borrower inherits the previous one's tokens, buffers, and partial state. A data-leak bug, not an untidiness bug.
- Use after release.
- No locking under threads.
- Hand-rolling one when SQLAlchemy or httpx already ships a pool.

### Adapter
- Class-based adapters (inheriting from the adaptee). Overriding a method the adaptee already has changes its meaning for every existing user of that type.
- Adapters that grow business rules.
- Adapters that flatten distinct errors into one, so callers can't tell "not found" from "service down".

### Bridge
- Applied when only one axis actually varies: that's Strategy with more classes.
- Coupling the two hierarchies at the concrete level, which is exactly the thing the pattern exists to avoid.

### Ports & Adapters
- **The layering relapse**, in Cockburn's own words: teams add the layer and find *"a few years later that the new layer is cluttered with business logic and the old problem has reappeared."* A layer without an enforced import direction buys nothing. Add a lint rule or a test.
- Ports defined in infrastructure vocabulary (`execute_sql`, `get_json`) rather than domain vocabulary (`reserve`, `get_stock`): the abstraction leaks and swapping the implementation is impossible anyway.
- Applied to a CRUD app with no domain logic.

### Repository / UoW / Service Layer
- A repository per table: that is the ORM again, with more code. Evans: *"Provide repositories only for aggregate roots that actually need direct access."*
- Wrapping SQLAlchemy's `Session` in a UoW that adds nothing; `Session` already is one.
- **Anemic domain**: entities reduced to data holders while all rules live in services. Fowler: *"they incur all of the costs of a domain model, without yielding any of the benefits."* Note the legitimate escape: if the logic really is procedural, choose Transaction Script deliberately and skip the entities.
- A service layer introduced before orchestration logic ever appeared in the controllers.

### Event Sourcing
- Adopted for an audit trail alone. Fowler: *"you could also do this with more regular logging mechanisms, and that way not have to deal with the odd interface."*
- Replay that re-triggers external side effects: emails re-sent, payments re-charged.
- Replay that re-queries external systems and gets *today's* answer for a past event.
- Assuming code changes are safe under replay. New features usually are; **bug fixes are not**, because the system must distinguish what happened with the bug from what happens without it.
- No snapshot or checkpoint strategy, so replay gets slower forever.
- Needing Parallel Models or Retroactive Events later: Fowler says *"it's very hard to retrofit these patterns onto a system that wasn't built with Event Sourcing."*

### CQRS
- Applied system-wide instead of to one bounded context.
- Adopted for a read that would have been fixed by an index or a denormalized column. Escalate through the cheap options first.
- Ignoring read-your-writes: a user who submits a form and immediately doesn't see their change files a bug report.

### Saga
- **Assuming isolation.** There is no I in ACID here. Lost updates, dirty reads, and non-repeatable reads between steps are design work you must do, not properties you get.
- Compensations that can themselves fail, with no plan.
- Irreversible steps placed before the pivot.
- Choreography for a workflow with many steps: Microsoft's saga guidance says choreography *"can be confusing when you add new steps"* and carries a risk of cyclic dependencies between participants ([source](https://learn.microsoft.com/en-us/azure/architecture/patterns/saga)).

### Retry
- **Retrying at every layer.** 3 retries × 3 layers = 64 attempts on the database. Find where retries already happen before adding more.
- **No jitter.** Synchronized retries after a blip re-create the overload.
- Retrying non-idempotent operations; retrying throttling responses (429/503) immediately or without a budget. Retry them slower, honour the server's delay hint, and stop when the per-client budget is empty.
- Backoff that outlives the caller's deadline.

### Circuit Breaker
- One breaker in front of a sharded resource, merging healthy and unhealthy shards into one signal.
- No manual override for operators.
- Added where bounded retries were already enough. AWS: breakers *"introduce modal behavior into systems that can be difficult to test, and can introduce significant addition time to recovery."*

### Cache-Aside
- Invalidating before writing, which lets a concurrent read repopulate the stale value for a full TTL.
- Presenting a cache as a consistency mechanism.
- No dogpile protection on hot keys.

### Feature flags
- Toggle debt: flags that outlive their purpose, each doubling the untested state space.
- A flag used as the rollback plan for a destructive migration. It is not one.

### Context Object
- Every function in the codebase taking it: that is a global variable wearing a dataclass.
- Fields added over time to avoid changing signatures, with no relationship to each other.
- Low-level helpers taking a context to reach two values.

### Functional Core, Imperative Shell
- `if` statements in the shell: each one is a business rule that escaped the core.
- A core that needs a snapshot of half the database to decide anything. The decision interleaves reads and writes; use ports instead.

### Parse, don't validate
- Parsing at the boundary and then passing the raw dict onward anyway, so downstream code re-validates.
- Accepting `bool` where an `int` is required: `isinstance(True, int)` is true.

### Null Object
- Using one where absence is a configuration error, which turns a loud failure into silence.

### Hedged requests
- Hedging writes or any non-idempotent call.
- Hedging every request instead of only those slower than about p95, which doubles load.
- Hedging while the dependency is overloaded.

---

## Smells that look structural but aren't

Recognizing these prevents the most wasteful failure mode of this skill: a well-argued refactor that doesn't touch the user's actual problem.

| What they describe | The real cause | Don't reach for |
|---|---|---|
| "It's slow" | An N+1 query, a missing index, a sync call in an async path | CQRS, caching layers |
| "It breaks when we deploy" | No migration strategy, no health checks | Hexagonal architecture |
| "We can't understand it" | No names, no types, no tests: often not a structure problem at all | More layers |
| "Bugs keep coming back" | No regression tests | Any pattern |
| "Two teams keep conflicting in this file" | Ownership and process; a registry may help but won't fix the org | Microservices |
| "We can't test it" | Genuinely structural: this one **is** a design problem | none |
| "It's hard to add a feature" | Ask which feature. If the answer is specific, so is the fix | Rewriting the architecture |

The middle column is where the fix lives. Say so plainly, then offer the structural improvement separately if one is warranted.

---

## Reviewing a design

Fast checks, in order of information per unit of effort:

1. **Read the imports.** Do domain modules import frameworks? Do arrows run inward? This finds layering violations in seconds.
2. **Read a test.** How much has to be constructed before the line under test runs? That is a direct measure of coupling.
3. **Count implementations per interface.** One, forever, means the interface is a wrapper.
4. **Trace one request end to end.** Count the files. Note where the flow becomes unfollowable.
5. **Find the composition root.** If concrete classes are constructed in many places, there isn't one, and no pattern in the codebase is fully paying off.
6. **Ask what the last three changes were.** The axes that actually vary are in the git history, not in anyone's roadmap.

That last one is worth asking the user directly. It converts a design debate into a factual question, and the answer is usually surprising to everyone involved.
