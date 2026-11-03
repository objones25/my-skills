# Structured concurrency: anyio, cancel scopes, task groups

Verified against anyio 4.15.1 on Python 3.14.

## The argument

Nathaniel J. Smith's *Notes on structured concurrency, or: Go statement considered harmful* is the
origin of everything in this file. The claim: a bare "spawn a task and move on" primitive
(`go f()`, `Thread(...).start()`, `asyncio.create_task()`) breaks three guarantees that
structured programming otherwise gives you for free.

1. **You can no longer read a function and know when it is done.** If `process()` spawned a
   background task, `await process()` returning does not mean the work finished.
2. **`with` blocks stop guaranteeing cleanup.** A spawned task may still be using the file, the
   connection, or the lock after the block that opened it has exited.
3. **Errors vanish.** An exception in a spawned task has nowhere to propagate to, so it is logged
   at best and silently dropped at worst.

The fix is the **nursery** (anyio and asyncio call it a task group): a scoped object that tasks
are spawned into, where the enclosing block cannot exit until every child has finished. That one
rule restores all three guarantees. A function that uses a task group internally is a black box
again (it cannot return before its children are done) and a child's exception propagates to the
parent like any other.

`asyncio.TaskGroup` (3.11, October 2022) is this idea landing in the standard library, eight
years after asyncio shipped without it (3.4, March 2014).

## anyio task groups

```python
import anyio

async def main():
    async with anyio.create_task_group() as tg:
        tg.start_soon(worker, 1)          # fire and forget INTO the group; not awaited
        tg.start_soon(worker, 2)
    # control does not reach here until both workers have finished
```

`start_soon(func, *args)` schedules and returns immediately. Note it takes the function and its
arguments separately, not a called coroutine.

`await tg.start(func, *args)` is the other half, and has no asyncio equivalent: it blocks the
caller until the new task signals readiness. The target must accept a `task_status` argument and
call `task_status.started()`:

```python
async def serve(port, *, task_status=anyio.TASK_STATUS_IGNORED):
    listener = await anyio.create_tcp_listener(local_port=port)
    task_status.started(listener.extra(anyio.abc.SocketAttribute.local_port))
    await listener.serve(handle)

async with anyio.create_task_group() as tg:
    bound_port = await tg.start(serve, 0)   # returns only once the socket is listening
```

That is the pattern for "start a server, then connect to it" without a sleep. If the target never
calls `started()`, the caller hangs forever. That is the one sharp edge.

On a child failure, all children are cancelled and the exceptions surface as a
`BaseExceptionGroup` from the task group's `__aexit__`, same as `asyncio.TaskGroup`.

## Cancel scopes

This is what anyio has that asyncio does not. A cancel scope is a region that can be cancelled as
a unit and carries a deadline.

```python
with anyio.move_on_after(5):        # timeout expires -> block just exits, no exception
    await slow_thing()

with anyio.fail_after(5):           # timeout expires -> raises TimeoutError
    await slow_thing()

with anyio.CancelScope() as scope:  # manual
    ...
    scope.cancel()                  # cancel everything in this scope from anywhere
```

Note these are plain `with`, not `async with`. `CancelScope(shield=True)` exempts the block from
outer cancellation. It is the same use as `asyncio.shield`, with the same warning: use it for cleanup,
not for work.

An anyio task group *contains* a cancel scope, so `tg.cancel_scope.cancel()` stops every child at
once regardless of where it was spawned. `asyncio.TaskGroup` has no equivalent: per anyio's own
docs, it *"does not offer any way to cancel, or even list all of the contained tasks."*

## Level vs edge cancellation

The deepest behavioral difference, in anyio's words:

> Asyncio employs a type of cancellation called *edge cancellation*. This means that when a task
> is cancelled, a `CancelledError` is raised in the task and the task then gets to handle it
> however it likes, even opting to ignore it entirely. In contrast, tasks that either explicitly
> use a cancel scope, or are spawned from an AnyIO task group, use *level cancellation* … as long
> as a task remains within an *effectively cancelled* cancel scope, it will get hit with a
> cancellation exception any time it hits a *yield point*.

Concretely: under asyncio, a task can swallow `CancelledError` and keep going. That is the
`SWALLOWED_CANCEL` bug, and it is possible because cancellation is delivered once. Under anyio's
level model, swallowing it does not help; the next `await` raises again. Cancellation becomes a
state rather than an event.

The cost is that asyncio-native code not written to expect repeated cancellation can misbehave
under anyio. The docs cite `asyncio.Condition`, which suppresses `CancelledError` until it can
reacquire its lock and can end up busy-retrying.

## Threads

```python
import anyio.to_thread

value = await anyio.to_thread.run_sync(blocking_fn, arg)
```

Signature (4.15.1): `run_sync(func, *args, abandon_on_cancel=False, cancellable=None, limiter=None)`.
`cancellable` is the deprecated old name for `abandon_on_cancel`, renamed in 4.1.0.

The default limiter allows **40 concurrent threads**: *"any calls to `to_thread.run_sync()`
without an explicit limiter argument will cause a maximum of 40 threads to be spawned."* That
same limiter is what Starlette uses for `def` handlers, which is why a FastAPI app with sync
database calls stalls at 40 concurrent requests. `asyncio.to_thread` does not draw from it (it uses
the event loop's own default executor), so raising this number does nothing for code that calls
`asyncio.to_thread`. Change it from inside the running loop (outside one, the call raises
`NoEventLoopError`):

```python
anyio.to_thread.current_default_thread_limiter().total_tokens = 100
```

Or pass a scoped one so a single noisy subsystem cannot starve the rest:

```python
db_limiter = anyio.CapacityLimiter(10)
await anyio.to_thread.run_sync(query, sql, limiter=db_limiter)
```

## Memory object streams

anyio's typed channel: the structured-concurrency version of `asyncio.Queue`.

```python
send, receive = anyio.create_memory_object_stream[str](max_buffer_size=10)

async def consume(receive):
    async with receive:
        async for item in receive:
            print("received", item)

async with anyio.create_task_group() as tg:
    tg.start_soon(consume, receive)
    async with send:
        for n in range(10):
            await send.send(f"number {n}")
```

The generic-subscript syntax `create_memory_object_stream[T](...)` arrived in anyio 4.0.0 and
replaced the old `item_type=` argument.

`max_buffer_size` defaults to **0**, which means a rendezvous: `send()` blocks until a receiver is
actually waiting. That is stronger backpressure than `asyncio.Queue`'s default of unbounded, and a
better default, because the producer cannot run ahead of the consumer at all unless you opt in.

Closing the send side ends the consumer's `async for` cleanly, which is why both sides are used as
context managers above. Forgetting to close `send` hangs the consumer forever.

## When to reach for anyio over plain asyncio

Use anyio when you want cancel scopes (cancelling a group of tasks from outside, deadlines that
compose), when you need `tg.start()` readiness signaling, when you want level cancellation's
stronger guarantee, or when you are writing a library that should run under either asyncio or
trio (httpx is exactly that case).

Stay on plain asyncio when the codebase is asyncio-native, the team knows it, and `TaskGroup` plus
`asyncio.timeout` already cover what you need. They cover more than people assume.

anyio's own docs name its remaining gaps honestly: task groups cannot return child results
directly and cannot cancel one child without cancelling the group.

## Sources

- Nathaniel J. Smith, *Notes on structured concurrency*: https://vorpus.org/blog/notes-on-structured-concurrency-or-go-statement-considered-harmful/
- anyio, *Creating and managing tasks*: https://anyio.readthedocs.io/en/stable/tasks.html
- anyio, *Cancellation and timeouts*: https://anyio.readthedocs.io/en/stable/cancellation.html
- anyio, *Working with threads*: https://anyio.readthedocs.io/en/stable/threads.html
- anyio, *Streams*: https://anyio.readthedocs.io/en/stable/streams.html
- anyio, *Why AnyIO?* (TaskGroup comparison): https://anyio.readthedocs.io/en/stable/why.html
- anyio, *Version history*: https://anyio.readthedocs.io/en/stable/versionhistory.html
- Python docs, *Task Groups*: https://docs.python.org/3/library/asyncio-task.html#task-groups
