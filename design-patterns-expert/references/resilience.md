# Resilience patterns at service boundaries

Extensibility is not only about swapping implementations. A system that falls over when one dependency slows down is not extensible in any useful sense: every new integration is a new way to take the whole thing down. These patterns belong in the adapter layer, where the failure originates, never inside domain logic.

Contents: [Why these are one family](#0-why-these-are-one-family-load-amplification) · [Timeout](#1-timeout-and-deadline-propagation) · [Retry](#2-retry-with-backoff-and-jitter) · [Circuit Breaker](#3-circuit-breaker) · [Bulkhead](#4-bulkhead) · [Throttling & rate limiting](#5-throttling-server-side-and-rate-limiting-client-side) · [Idempotency keys](#6-idempotency-keys) · [Idempotent Consumer](#7-idempotent-consumer) · [Cache-Aside](#8-cache-aside) · [Feature flags](#9-feature-flags) · [Hedged requests](#10-hedged-requests) · [Library table](#python-libraries)

---

## 0. Why these are one family: load amplification

Read this before recommending any of them individually. It is the mechanism that connects them.

**Retries multiply across layers.** Google SRE: *"a single request at the highest layer may produce a number of attempts as large as the product of the number of attempts at each layer to the lowest layer."* Their example: with the backend, frontend, and JavaScript layers each retrying three times, *"a single user action may create 64 attempts (4^3) on the database."* AWS's five-layer version puts it at *"the load on the database will increase 243x."* (AWS counts 3 tries per layer, so 3^5; Google counts 3 retries as 4 attempts, which over five layers would be 4^5 = 1,024.)

**The runaway loop.** Failed requests get retried, which adds load; *"100 QPS of retries in the first second leads to 200 QPS, then to 300 QPS"*; fewer requests succeed on the first attempt, so more are retried; the backend *"can melt down and crash under the sheer load of requests and retries"*; and crashed servers push their load onto the survivors.

**Retries are structurally hostile under overload.** AWS: *"Retries are 'selfish.' In other words, when a client retries, it spends more of the server's time to get a higher chance of success."* And: *"when failures are caused by overload, retries that increase load can make matters significantly worse."*

Three enforceable limits, from Google SRE:
1. **Per-request budget**: *"up to three attempts. If a request has already failed three times, we let the failure bubble up to the caller."*
2. **Per-client budget**: track the ratio of retries to requests; *"A request will only be retried as long as this ratio is below 10%."*
3. **Retry at exactly one layer**: *"If multiple layers retried, we'd have a combinatorial explosion."* AWS says the same: *"retry at a single point in the stack."*

Plus a process-wide cap: *"only allow 60 retries per minute in a process, and if the retry budget is exceeded, don't retry; just fail the request."*

That last rule is the one to state whenever you add retry logic to a system that already has some: **find out where retries already happen before adding more.**

> Sources: [Google SRE, Addressing Cascading Failures](https://sre.google/sre-book/addressing-cascading-failures/) · [Handling Overload](https://sre.google/sre-book/handling-overload/) · [Amazon Builders' Library, Timeouts, retries, and backoff with jitter](https://d1.awsstatic.com/builderslibrary/pdfs/timeouts-retries-and-backoff-with-jitter.pdf)

---

## 1. Timeout and deadline propagation

**Always the first thing you add.** Without it, everything else is decoration.

Nygard (*Release It!*, "Fail Fast"): *"If slow responses are worse than no response, the worst must surely be a slow failure response."* A slow dependency ties up caller threads and connections until the pool is exhausted, at which point *unrelated* requests start failing. That is how one degraded dependency becomes a total outage.

**Choosing the value**: the only concrete method I found in a primary source, from AWS: pick an acceptable rate of false timeouts (say 0.1%), then use the corresponding latency percentile of the downstream service (p99.9). For clients calling over the internet, factor in worst-case network latency, and add padding when the dependency's p99.9 is close to its p50.

**Deadline propagation** is the part most systems get wrong. Set one absolute deadline high in the stack and pass it down: *"if server A selects a 30-second deadline, and processes the request for 7 seconds before sending an RPC to server B, the RPC from A to B will have a 23-second deadline."* Servers should check the remaining deadline before starting more work. Without propagation, a downstream service pulls work off its queue for a caller that gave up long ago (burning capacity on results nobody will read) and total worst-case latency becomes the *sum* of per-hop timeouts.

Python: `asyncio.timeout()` / `asyncio.wait_for`; `httpx.Timeout` for per-phase (connect/read/write/pool) bounds. Pass a deadline explicitly through your call chain: there is no maintained Python library for automatic deadline propagation.

---

## 2. Retry with backoff and jitter

Only for **transient** faults. Classify before retrying:

| Fault | Action |
|---|---|
| Not transient / won't succeed on repeat | **Cancel**: report immediately |
| Unusual or rare (corrupted packet) | **Retry immediately** |
| Commonplace connectivity / busy | **Retry after delay**, exponential with jitter |

**Jitter is mandatory, not an optimization.** Without it, all clients that failed together retry together. SRE: *"Always use randomized exponential backoff when scheduling retries... If retries aren't randomly distributed over the retry window, a small perturbation (e.g., a network blip) can cause retry ripples to schedule at the same time, which can then amplify themselves."* AWS quantifies the contention: *"With N clients contending, the total amount of work done by the system increases with N²."*

The four formulas, from AWS's canonical article:

```python
# No jitter: do not ship this
sleep = min(cap, base * 2 ** attempt)

# Full Jitter: the default recommendation
sleep = random.uniform(0, min(cap, base * 2 ** attempt))

# Equal Jitter
temp  = min(cap, base * 2 ** attempt)
sleep = temp / 2 + random.uniform(0, temp / 2)

# Decorrelated Jitter, stateful; `sleep` is the previous sleep
sleep = min(cap, random.uniform(base, sleep * 3))
```

At 100 contending clients, jitter *"reduced our call count by more than half"* versus plain exponential backoff. Full Jitter does the least work (about the same call count as Equal Jitter, fewer than Decorrelated) but finishes slightly later than Decorrelated; Equal Jitter is the slowest. The author calls Equal the loser and the choice between Full and Decorrelated "less clear". Full Jitter is a sound default.

### Throttling (429/503) and retry budgets

Throttling responses are retried, but slower and from a budget. AWS's experimental SDK for Python v4 (not Boto3) uses a 1 s base delay for throttling errors against 50 ms for other transient errors, and a per-client token bucket: 500 tokens; a retry costs 5 after a throttle and 14 otherwise; a success after retries refunds what it spent; a first-try success adds 1; an empty bucket means fail without retrying. Honour the server's delay hint when the API documents one (`Retry-After`; AWS uses its own `x-amz-retry-after`). Never retry a throttle immediately.

`examples/retry_budget.py` runs 1,000 calls against a fully throttled dependency: 3,000 attempts with plain 3-attempt retries, 1,100 with the budget (950 retries refused). With a healthy dependency (5% transient failures) both make 1,060 attempts.

> Source: [AWS SDK for Python v4, Retries](https://docs.aws.amazon.com/sdk-for-python/v4/guide/config-retries.html) (an experimental, non-GA SDK; the page says to use Boto3 for production)

### Documented anti-patterns
- **Nested retry policies.** A task with retry logic calling another task with retry logic *"this extra layer of retries can add long delays to the processing"*: the layered version of amplification.
- **Retry as a scalability workaround.** *"If an application experiences frequent busy faults, it's often a sign that the service or resource being accessed should be scaled up."*
- **Retrying non-idempotent operations.** *"Operations should be idempotent when you use the retry with backoff pattern. Otherwise, partial updates might corrupt the system state."* See §6.
- **Retrying business-logic errors.** A 400 will be a 400 next time. For non-transient errors, *"it is more efficient to fail fast by using the circuit breaker pattern."*
- **Backoff blowing the caller's deadline.** Exponential backoff *"can have an impact on the service timeout, resulting in longer wait times for the end user."* Bound total retry time by the remaining deadline, not just by attempt count.
- **Log flooding.** Log intermediate failures at info level and only the final failure as an error, *"in order to avoid flooding operators with alerts on operations where subsequently retried attempts were successful."*

> Sources: [Azure, Retry](https://learn.microsoft.com/en-us/azure/architecture/patterns/retry) · [AWS, Exponential Backoff And Jitter](https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/) · [AWS, Retry with backoff](https://docs.aws.amazon.com/prescriptive-guidance/latest/cloud-design-patterns/retry-backoff.html)

---

## 3. Circuit Breaker

A proxy that counts failures and, past a threshold, fails calls immediately without invoking the dependency, probing periodically for recovery.

States: **Closed** (pass through, count failures) → **Open** on threshold breach (fail fast, start timer) → **Half-Open** on expiry (limited probes) → Closed on success, or back to Open on any probe failure. Azure: *"The Half-Open state helps prevent a recovering service from suddenly being flooded with requests."*

**What it prevents, precisely:** resource exhaustion in the *caller*. Azure, on requests blocked waiting for timeouts: *"These blocked requests might hold critical system resources, such as memory, threads, and database connections. This problem can exhaust resources, which might fail other unrelated parts of the system that need to use the same resources."*

### When NOT to use it: Azure's own list
- Local in-memory resources: *"a circuit breaker adds overhead to your system."*
- As a substitute for exception handling in business logic.
- When *"well-known retry algorithms are sufficient and your dependencies are designed to handle retry mechanisms"*: the breaker *"might add unnecessary complexity to your system."*
- When *"waiting for a circuit breaker to reset might introduce unacceptable delays."*
- Message-driven architectures where DLQs plus built-in isolation already cover it.
- When infrastructure already handles it (global load balancer health checks, service mesh).

### Additional hazards
- **Don't put one breaker in front of a sharded resource.** *"one shard might be fully accessible while another experiences a temporary problem. If the error responses in these scenarios are merged, an application might try to access some shards even when failure is likely. And access to other shards might be blocked even though it's likely to succeed."*
- **Tuning is a real risk in both directions.** *"if the circuit breaker remains in the Open state for a long period, it can raise exceptions even if the reason for the failure is resolved. Similarly, a circuit breaker can fluctuate and reduce the response times of applications if it switches from the Open state to the Half-Open state too quickly."*
- **Half-open timer stampede** under concurrency: the first failed call defines the expiry; make sure subsequent calls don't push it forward indefinitely.
- **AWS's counter-argument, worth quoting to anyone reaching for a breaker reflexively:** circuit breakers *"introduce modal behavior into systems that can be difficult to test, and can introduce significant addition time to recovery."* AWS often prefers token-bucket retry limiting instead.
- **Build a manual override.** Azure advises a manual reset, especially when recovery time varies widely, so operators can force a breaker open or closed.

> Sources: [Azure, Circuit Breaker](https://learn.microsoft.com/en-us/azure/architecture/patterns/circuit-breaker) · [AWS Builders' Library, Timeouts, retries, and backoff with jitter](https://d1.awsstatic.com/builderslibrary/pdfs/timeouts-retries-and-backoff-with-jitter.pdf)

---

## 4. Bulkhead

Partition resources so exhaustion in one pool can't starve the others.

The failure it prevents, stated precisely: *"When the consumer sends a request to a misconfigured or unresponsive service, the resources that the client's request uses might remain unavailable for an extended period."* ... *"Eventually, the consumer can't send requests to any other services, not only the original unresponsive service."*

Implementations, coarse to fine: separate processes/containers; **a connection pool per dependency**; a thread pool per workload class; separate queues with dedicated consumers.

In Python the primitives are stdlib and the discipline is what matters:

```python
# one semaphore per dependency, never one global
_limits = {
    "payments": asyncio.Semaphore(10),
    "search":   asyncio.Semaphore(50),
}

# separate clients so connection pools don't share
payments = httpx.AsyncClient(limits=httpx.Limits(max_connections=10))
search   = httpx.AsyncClient(limits=httpx.Limits(max_connections=50))
```

**When not to:** bulkheads intentionally strand headroom, so *"less efficient use of resources might not be acceptable"*; and don't re-create what your platform already provides (API gateway rate limits, container resource limits).

Partition along meaningful boundaries: by dependency, tenant, or bounded context. Arbitrary partitioning gives you isolation without meaning.

> Source: [Azure, Bulkhead](https://learn.microsoft.com/en-us/azure/architecture/patterns/bulkhead)

---

## 5. Throttling (server side) and rate limiting (client side)

These are two patterns, and the distinction is load-bearing: **throttling is your service enforcing its own limit; rate limiting is your client staying under someone else's.**

### Throttling

Cap consumption and reject or degrade past a threshold, so the system stays within its SLOs instead of collapsing. It buys time for autoscaling: at T1 the soft limit trips and scale-out begins; between T1 and T2 throttling rejects excess; at T2 capacity arrives and throttling lifts.

Algorithms: **token bucket** (bursts up to bucket size; suits gateways), **leaky bucket** (constant emission; suits backends needing steady ingress), **fixed window** (simple, but *"admits back-to-back bursts at window boundaries"*), **sliding window** (smoother, more state).

Documented hazards, several of which are counterintuitive:
- **Limiting on the wrong dimension.** *"Request rate is the most familiar dimension to limit, but the real bottleneck is often concurrent in-flight requests, queue depth, CPU or memory utilization, or a downstream dependency's own limits."*
- **Swallowing downstream 429/503.** If you hide a downstream 429 or 503 by retrying silently or returning a generic 500, *"callers can't slow down, retries amplify, and the overload cascades back upstream."* Azure: *"Surface back-pressure to upstream callers so that the entire call chain sheds load together."*
- **Expensive rejections.** If rejecting requires full auth and parsing, a flood of rejections still saturates you. Reject early and load-test the rejection path.
- **Caching is not throttling.** *"A cache lowers average load on the origin but doesn't bound peak load."*
- **Averages hide breaches.** Trigger on p99 latency against SLO, not average utilization.
- **Retrofitting is expensive**: throttling affects the whole system.

Return 429 for caller-exceeded-rate, 503 for unexpected spikes, and include `Retry-After`.

### Adaptive client-side throttling (Google SRE)

The highest-leverage variant, because each client decides locally with no coordination. Track `requests` and `accepts` over a two-minute window and reject locally with probability:

```
P(reject) = max(0, (requests - K * accepts) / (requests + 1))     # K = 2
```

Lower K is more aggressive. The property that makes it work: *"the decision is made by the client task based entirely on local information and using a relatively simple implementation."*

### Related SRE guidance worth carrying over
- **QPS is a bad capacity metric**: *"Different queries can have vastly different resource requirements."* Measure CPU or another direct resource instead.
- **Criticality levels** propagated with the request (`CRITICAL_PLUS` → `SHEDDABLE`), so a loaded backend sheds batch work before user-facing work.
- **Load shedding is not turning down.** *"It's a common mistake to assume that an overloaded backend should turn down and stop accepting all traffic... We actually want the backend to continue accepting as much traffic as possible, but to only accept that load as capacity frees up."*
- **Rarely-used degradation paths are themselves a risk.** Graceful degradation *"shouldn't trigger very often... Keep the system simple and understandable, particularly if it isn't used often."*

> Sources: [Azure, Throttling](https://learn.microsoft.com/en-us/azure/architecture/patterns/throttling) · [Azure, Rate Limiting](https://learn.microsoft.com/en-us/azure/architecture/patterns/rate-limiting-pattern) · [Google SRE, Handling Overload](https://sre.google/sre-book/handling-overload/) · [Google SRE, Addressing Cascading Failures](https://sre.google/sre-book/addressing-cascading-failures/)

---

## 6. Idempotency keys

A client-generated token on a mutating request that lets the server recognize and deduplicate retries. This is the precondition every retry source demands: *"APIs with side effects aren't safe to retry unless they provide idempotency."*

The failure it prevents: the server commits, the response is lost in flight, the client retries, and the customer is charged twice.

**Server behavior**, from the IETF draft and Stripe's production implementation:
- Fingerprint the request body so key reuse with different content is detectable.
- First occurrence → process. Retry after completion → **return the stored response**, including if it was a 500. Concurrent retry while the original is in flight → **409 Conflict**.
- `400` if a required key is missing; `422` if the key is reused with a different payload.
- **Save results only once execution begins.** Validation failures and concurrent conflicts stay retryable: Stripe is explicit about this.
- **Scope the cache key to the caller/tenant**, not the client-supplied key alone. Otherwise one tenant's key can collide with another's.
- Keys: V4 UUIDs or equivalent entropy; up to 255 chars; never sensitive data. Stripe (API v1) prunes keys once they are at least 24 hours old.
- **TTL must exceed the maximum client retry window**, or a late retry re-executes.
- POST and PATCH only; keys on GET/DELETE do nothing.

Note the IETF `Idempotency-Key` spec is still an Internet-Draft, not an RFC (revision -08, published 2026-10-22, expires 2027-04-25; check the datatracker for the current state): it is the best available spec text and widely followed, but it is not a standard. Say so if you cite it.

> Sources: [IETF draft-ietf-httpapi-idempotency-key-header](https://datatracker.ietf.org/doc/html/draft-ietf-httpapi-idempotency-key-header) · [Stripe, Idempotent requests](https://docs.stripe.com/api/idempotent_requests)

---

## 7. Idempotent Consumer

The message-side counterpart. At-least-once delivery is the default for Service Bus, Event Hubs, Kafka and RabbitMQ, so duplicates are guaranteed, not hypothetical: producer retries after a lost ack, broker redelivery after an expired lock, consumer crash between the DB write and the ack.

**Design for natural idempotency first.** An upsert keyed on a business identifier, a write that sets an absolute value rather than incrementing, or an HTTP `PUT` needs no bookkeeping at all. *"use deduplication techniques only for operations that can't be made naturally idempotent."* Shipping absolute state rather than deltas is the lever that makes this possible.

When bookkeeping is needed:
- **Pick a stable key**: a producer-assigned message id or business idempotency key. **Not** `CorrelationId` (it groups related messages), not transport identifiers (regenerated on redelivery), not timestamps.
- **Commit the dedup marker and the side effect in one transaction** (the inbox variant). Otherwise there is a window where the work happened but the marker didn't.
- **Concurrency is the classic bug.** Under competing consumers, two instances can both pass an existence check before either commits. Fix it at the data store: a **unique constraint** on the dedup key so only one insert wins, or an atomic conditional write. Never check-then-set.
- **External side effects that can't join the transaction:** record the key *in-progress*, make the call, then mark *completed* with the outcome.
- **TTL must exceed the redelivery window**: including DLQ resubmission, which *"can occur long after the normal redelivery window closes."*
- **Broker dedup is not a substitute**: it operates send-side within a bounded window and doesn't stop a consumer processing a redelivery twice.
- **Dedup is not ordering.** Add sessions or a version field to reject stale messages.
- **Propagate the key downstream**: making one consumer idempotent doesn't protect the services it calls.

> Source: [Azure, Idempotent Consumer](https://learn.microsoft.com/en-gb/azure/architecture/patterns/idempotent-consumer)

---

## 8. Cache-Aside

The application owns loading: read cache, on miss read the store and populate, on write update the store and then invalidate.

**Order matters and the reason is specific:** *"Update the data store before removing the item from the cache. If you remove the cached item first, there's a small window of time when a client might fetch the item before the data store is updated."* Invalidate first and a concurrent read can repopulate from the pre-write value, and that stale entry then survives its full TTL.

**It does not give you consistency.** *"an external process can change an item in the data store at any time. This change doesn't appear in the cache until the item loads again."* Don't sell it as consistency.

Other documented limits: local in-memory caches diverge across instances; don't cache sensitive data; prime the cache at startup for static datasets rather than caching lazily; skip it when most requests miss. TTL tuning is a genuine trade (too short and the app keeps refetching from the data store; too long and cached data goes stale) and under memory pressure most caches evict LRU anyway, so a long TTL is not a retention guarantee.

**The dogpile.** When a hot key expires under load, every concurrent request misses and hits the origin at once. Azure warns against substituting caching for throttling for exactly this reason. Mitigate with a per-key lock on refill, or serve-stale-while-revalidate.

> Sources: [Azure, Cache-Aside](https://learn.microsoft.com/en-us/azure/architecture/patterns/cache-aside) · [Azure, Throttling](https://learn.microsoft.com/en-us/azure/architecture/patterns/throttling) (caching is not a substitute for throttling)

---

## 9. Feature flags

Decouple deploy from release. Four kinds, with different lifetimes and different dynamism; mixing them up is the root of most flag messes:

| Toggle | Purpose | Lifetime | Dynamism |
|---|---|---|---|
| **Release** | Ship dormant incomplete features; enables trunk-based development | days–weeks | static |
| **Experiment** | A/B testing, consistent per-user cohorting | hours–weeks | per request |
| **Ops** | Degrade or kill a feature during an incident | short (kill switches may be permanent) | very fast reconfiguration |
| **Permissioning** | Gate by user group (beta, premium, internal) | months–years | per request |

An Ops toggle is a manually-operated circuit breaker, and it is the one worth adding proactively to any risky new integration.

Prefer the least dynamic configuration mechanism that works: hardcoded → env vars → config in source control → database + admin UI → distributed store. Hodgson: *"Managing toggle configuration via source control and re-deployments is preferable, if the nature of the feature flag allows it."*

**Toggle debt is the documented anti-pattern.** *"Savvy teams view the Feature Toggles in their codebase as inventory which comes with a carrying cost and seek to keep that inventory as low as possible."* The cost is code complexity plus a combinatorial explosion of untested toggle-state combinations. Hodgson cites Knight Capital's *"$460 million dollar mistake"* as partly a toggle-management failure. Three countermeasures: add the removal task to the backlog when you add the flag; set expiry dates, with some teams shipping "time bombs" that fail the test suite once a flag outlives its date; or cap total inventory so a new flag requires deleting one.

A toggle is not a rollback for a destructive migration. Don't let it be treated as one.

> Source: [Pete Hodgson, Feature Toggles (aka Feature Flags)](https://martinfowler.com/articles/feature-toggles.html), martinfowler.com (2017)

---

## 10. Hedged requests

Send a read to one replica; if it has not answered by about the p95 latency, send a copy to a second replica, take the first answer and cancel the other. Only requests slower than p95 get a copy, so extra load stays near 5%.

`examples/hedged_request.py` simulates 200,000 calls with a 2% stall rate. Hedging at p95 (15.9 ms) sent 5.0% extra requests and cut p99 from 212.7 ms to 26.0 ms, and p99.9 from 254.9 ms to 32.0 ms. Its asyncio `hedged_call` cancels the losing request, which the tests assert.

- **Idempotent reads only.** Hedging a write is a duplicate write.
- **Not under overload.** Hedges add load when everything is slow; turn hedging off when the retry budget is empty.
- **Not for uniform slowness.** If every replica is slow, the hedge waits and then duplicates.

> Source: Dean & Barroso, [The Tail at Scale](https://research.google/pubs/the-tail-at-scale/), Communications of the ACM 56, pp. 74-80 (2013)

---

## Python libraries

Verify the current API before writing any of these into example code.

| Pattern | Library | Notes |
|---|---|---|
| Retry | **`tenacity`** | `wait_random_exponential` is Full Jitter (its source returns `random.uniform(min, high)`, checked on 9.1.4); `stop_after_attempt`, `stop_after_delay`, `retry_if_exception_type`, `AsyncRetrying` |
| Retry (opinionated) | **`stamina`** | Wraps tenacity; jitter on by default; async + Trio; Prometheus/structlog instrumentation; test-mode disable. Prefer for production: the defaults match the guidance above and it emits the metrics you need to detect amplification |
| Circuit breaker | **`pybreaker`** | `CircuitRedisStorage` for shared state across processes. Its documented async support is Tornado, not asyncio |
| Circuit breaker (asyncio) | **`circuitbreaker`** | `@circuit` decorator works on async functions; fallbacks; `CircuitBreakerMonitor` |
| Rate limiting (in-process) | **`aiolimiter`** | asyncio leaky bucket as an async context manager |
| Rate limiting (distributed) | **`limits`** | Fixed / moving / sliding-window strategies; Redis, Memcached, MongoDB backends; matching sync and async APIs |
| Saga orchestration | **`temporalio`** | Durable workflows, activity retries and timeouts, time-skipping test harness. No formally named saga API |
| Feature flags | **`openfeature-sdk`** | Vendor-neutral; **pre-1.0**, so the API may move |
| Timeout, bulkhead | *stdlib* | `asyncio.timeout()`, `asyncio.Semaphore` per dependency, `httpx.Limits`, separate executors |
| Transactional outbox | *hand-rolled* | An outbox table in the same transaction plus a relay, or Debezium for CDC. No canonical Python library |
