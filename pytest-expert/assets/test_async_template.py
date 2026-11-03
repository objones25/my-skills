"""Async test patterns, runnable as-is. Each one passes the audit.

Needs pytest-asyncio 1.x with these keys in the pytest config (see
assets/pytest-config.toml):

    asyncio_mode = "strict"
    asyncio_default_fixture_loop_scope = "function"
    asyncio_default_test_loop_scope = "function"

Strict mode means only tests marked @pytest.mark.asyncio (or pytestmark) run on the
asyncio plugin, so anyio-marked tests in the same suite are left to anyio's plugin.

Verified against pytest 9.2.0, pytest-asyncio 1.4.1, anyio 4.16.0, Python 3.13
(November 2026). references/async-testing.md explains each pattern.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from unittest import mock

import pytest
import pytest_asyncio

# --- code under test (inline so the template runs on its own) -----------------------

Sleep = Callable[[float], Awaitable[None]]


class TransientError(Exception):
    pass


async def fetch_with_retry(fetch: Callable[[], Awaitable[str]], *, attempts: int = 3,
                           base_delay: float = 0.5, sleep: Sleep = asyncio.sleep) -> str:
    """Retry with exponential backoff. `sleep` is injected so tests never wait."""
    for attempt in range(attempts):
        try:
            return await fetch()
        except TransientError:
            if attempt == attempts - 1:
                raise
            await sleep(base_delay * 2**attempt)
    raise AssertionError("unreachable")


class Worker:
    """Consumes a queue until cancelled, then records that cleanup ran."""

    def __init__(self) -> None:
        self.queue: asyncio.Queue[int] = asyncio.Queue()
        self.processed: list[int] = []
        self.closed = False

    async def run(self) -> None:
        try:
            while True:
                self.processed.append(await self.queue.get())
        finally:
            self.closed = True


async def notify(client, user: str) -> None:
    await client.send(user, "hello")


async def gather_profiles(load: Callable[[str], Awaitable[dict]], users: list[str]) -> list[dict]:
    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(load(u)) for u in users]
    return [t.result() for t in tasks]


# --- 1. An injected sleep: backoff is tested in microseconds --------------------------


@pytest.fixture
def fake_sleep():
    """Records requested delays instead of waiting."""
    delays: list[float] = []

    async def _sleep(seconds: float) -> None:
        delays.append(seconds)

    _sleep.delays = delays
    return _sleep


@pytest.mark.asyncio
async def test_retry_backs_off_exponentially_then_succeeds(fake_sleep):
    outcomes = iter([TransientError(), TransientError(), "ok"])

    async def flaky_fetch() -> str:
        result = next(outcomes)
        if isinstance(result, Exception):  # audit: ignore[BRANCHING] scripted fake, not a case loop
            raise result
        return result

    assert await fetch_with_retry(flaky_fetch, sleep=fake_sleep) == "ok"
    assert fake_sleep.delays == [0.5, 1.0]


@pytest.mark.asyncio
async def test_retry_reraises_after_last_attempt(fake_sleep):
    async def always_fails() -> str:
        raise TransientError("down")

    with pytest.raises(TransientError, match=r"^down$"):
        await fetch_with_retry(always_fails, attempts=2, sleep=fake_sleep)
    assert fake_sleep.delays == [0.5]


# --- 2. A hard timeout inside the test: a hang becomes a failure ----------------------


@pytest.mark.asyncio
async def test_worker_processes_queued_items_within_a_deadline():
    worker = Worker()
    task = asyncio.create_task(worker.run())
    for item in (1, 2, 3):  # audit: ignore[BRANCHING] arranging input, not driving cases
        worker.queue.put_nowait(item)
    async with asyncio.timeout(1.0):
        while len(worker.processed) < 3:  # audit: ignore[BRANCHING] wait-until with a deadline
            await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert worker.processed == [1, 2, 3]


# --- 3. Cancellation runs cleanup ----------------------------------------------------


@pytest.mark.asyncio
async def test_cancelling_the_worker_runs_its_finally_block():
    worker = Worker()
    task = asyncio.create_task(worker.run())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert worker.closed is True


# --- 4. Async fixtures with teardown --------------------------------------------------


@pytest_asyncio.fixture
async def running_worker():
    worker = Worker()
    task = asyncio.create_task(worker.run())
    yield worker
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_fixture_worker_is_live(running_worker):
    running_worker.queue.put_nowait(7)
    async with asyncio.timeout(1.0):
        while not running_worker.processed:  # audit: ignore[BRANCHING] wait-until with a deadline
            await asyncio.sleep(0)
    assert running_worker.processed == [7]


# --- 5. AsyncMock through autospec: awaited, with the real signature -------------------


class Client:
    async def send(self, user: str, message: str) -> None:
        raise NotImplementedError


@pytest.mark.asyncio
async def test_notify_awaits_send_with_user_and_message():
    client = mock.create_autospec(Client, instance=True)
    await notify(client, "ada")
    client.send.assert_awaited_once_with("ada", "hello")


def test_autospec_rejects_a_call_the_real_method_cannot_take():
    client = mock.create_autospec(Client, instance=True)
    with pytest.raises(TypeError, match="missing a required argument"):
        client.send("ada")


# --- 6. Task groups raise ExceptionGroup: assert with RaisesGroup ----------------------


@pytest.mark.asyncio
async def test_gather_profiles_surfaces_every_failure():
    async def load(user: str) -> dict:
        raise LookupError(user)

    with pytest.RaisesGroup(pytest.RaisesExc(LookupError, match="^ada$"), pytest.RaisesExc(LookupError, match="^bob$")):
        await gather_profiles(load, ["ada", "bob"])


# --- 7. anyio's plugin: one test, run on each backend that is installed ---------------


@pytest.fixture(params=["asyncio"])
def anyio_backend(request):
    """Add "trio" to params when trio is installed and the code supports it."""
    return request.param


@pytest.mark.anyio
async def test_anyio_fail_after_turns_a_hang_into_timeout_error():
    import anyio

    never = anyio.Event()
    with pytest.raises(TimeoutError):
        with anyio.fail_after(0.05):
            await never.wait()
