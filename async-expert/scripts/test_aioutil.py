"""Behavior tests for assets/aioutil. Needs httpx (or httpx2 plus alias_httpx); no network.

    uv run --with httpx python scripts/test_aioutil.py
    AIOUTIL_PATH=path/to/assets uv run --with httpx python scripts/test_aioutil.py   (test another copy)
"""

from __future__ import annotations

import asyncio
import itertools
import os
import sys
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ASSETS = Path(os.environ.get("AIOUTIL_PATH", Path(__file__).resolve().parents[1] / "assets"))
sys.path.insert(0, str(ASSETS))

import httpx  # noqa: E402

from aioutil import (  # noqa: E402
    RetriesExhausted,
    RetryPolicy,
    bounded_map,
    bounded_map_settled,
    build_client,
    cpu_map,
    partition,
    request_with_retry,
)

FAST = RetryPolicy(attempts=3, base_delay=0.001, max_delay=0.01)


def slow_square(x: int) -> int:
    """Module-level so the process pool can pickle it."""
    if x == 0:
        raise ValueError("fails immediately")
    time.sleep(1.5)
    return x * x


def mock_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=1.0)


class Retry(unittest.IsolatedAsyncioTestCase):
    async def test_get_retries_503_then_succeeds(self):
        calls = itertools.count()

        def handler(request):
            return httpx.Response(503 if next(calls) == 0 else 200)

        async with mock_client(handler) as client:
            response = await request_with_retry(client, "GET", "http://t/", policy=FAST)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(next(calls), 2)

    async def test_post_5xx_is_not_replayed(self):
        for status in (500, 502, 503, 504):
            calls = itertools.count()

            def handler(request, status=status, calls=calls):
                next(calls)
                return httpx.Response(status)

            async with mock_client(handler) as client:
                response = await request_with_retry(client, "POST", "http://t/", policy=FAST)
            self.assertEqual(response.status_code, status)
            self.assertEqual(next(calls), 1, f"POST got {status} and was sent again")

    async def test_post_429_is_retried(self):
        calls = itertools.count()

        def handler(request):
            return httpx.Response(429 if next(calls) == 0 else 201, headers={"retry-after": "0"})

        async with mock_client(handler) as client:
            response = await request_with_retry(client, "POST", "http://t/", policy=FAST)
        self.assertEqual(response.status_code, 201)

    async def test_post_read_timeout_not_retried(self):
        calls = itertools.count()

        def handler(request):
            next(calls)
            raise httpx.ReadTimeout("slow", request=request)

        async with mock_client(handler) as client:
            with self.assertRaises(httpx.ReadTimeout):
                await request_with_retry(client, "POST", "http://t/", policy=FAST)
        self.assertEqual(next(calls), 1)

    async def test_retry_after_beyond_max_delay_gives_up(self):
        async with mock_client(lambda r: httpx.Response(429, headers={"retry-after": "3600"})) as client:
            started = time.perf_counter()
            with self.assertRaises(RetriesExhausted):
                async with asyncio.timeout(2.0):
                    await request_with_retry(client, "GET", "http://t/", policy=RetryPolicy(max_delay=1.0))
        self.assertLess(time.perf_counter() - started, 0.5)

    async def test_cancellation_is_not_retried(self):
        async def hang(request):
            await asyncio.sleep(10)

        async with mock_client(hang) as client:
            started = time.perf_counter()
            with self.assertRaises(TimeoutError):
                async with asyncio.timeout(0.1):
                    await request_with_retry(client, "GET", "http://t/")
        self.assertLess(time.perf_counter() - started, 0.5)


class Bounded(unittest.IsolatedAsyncioTestCase):
    async def test_limit_is_respected_and_order_kept(self):
        live = peak = 0

        async def work(i):
            nonlocal live, peak
            live += 1
            peak = max(peak, live)
            await asyncio.sleep(0.01)
            live -= 1
            return i * 2

        self.assertEqual(await bounded_map(work, list(range(30)), limit=4), [i * 2 for i in range(30)])
        self.assertEqual(peak, 4)

    async def test_settled_keeps_going(self):
        async def maybe(i):
            if i % 5 == 0:
                raise ValueError(i)
            return i

        values, errors = partition(await bounded_map_settled(maybe, list(range(10)), limit=3))
        self.assertEqual([e.index for e in errors], [0, 5])
        self.assertEqual(len(values), 8)

    async def test_cpu_map_failure_does_not_block_the_loop(self):
        ticks: list[float] = []

        async def ticker():
            while True:
                ticks.append(time.perf_counter())
                await asyncio.sleep(0.05)

        tick_task = asyncio.create_task(ticker())
        try:
            with self.assertRaises(ValueError):
                await cpu_map(slow_square, [0, 1, 2], max_workers=3)
            await asyncio.sleep(0.2)
        finally:
            tick_task.cancel()
        worst_gap = max(b - a for a, b in itertools.pairwise(ticks))
        self.assertLess(worst_gap, 0.5, f"event loop stalled for {worst_gap:.2f}s")


class RealServer(unittest.IsolatedAsyncioTestCase):
    """build_client against a local http.server, including a redirect."""

    @classmethod
    def setUpClass(cls):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/old":
                    self.send_response(301)
                    self.send_header("location", "/new")
                    self.send_header("content-length", "0")
                    self.end_headers()
                    return
                body = self.path.encode()
                self.send_response(200)
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    async def test_build_client_follows_redirects_and_fans_out(self):
        async with build_client(base_url=self.base, max_connections=4) as client:
            response = await client.get("/old")
            self.assertEqual(response.text, "/new")

            async def get(i):
                r = await client.get(f"/{i}")
                r.raise_for_status()
                return r.text

            self.assertEqual(await bounded_map(get, list(range(12)), limit=4), [f"/{i}" for i in range(12)])


if __name__ == "__main__":
    unittest.main()
