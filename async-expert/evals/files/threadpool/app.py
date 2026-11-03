import asyncio
import time
from contextlib import asynccontextmanager

import anyio.to_thread
from fastapi import FastAPI


def legacy_query(n: int) -> int:
    """Stands in for a sync DB driver call that takes ~200 ms."""
    time.sleep(0.2)
    return n


@asynccontextmanager
async def lifespan(app: FastAPI):
    anyio.to_thread.current_default_thread_limiter().total_tokens = 200
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/q/{n}")
async def q(n: int) -> dict:
    return {"value": await asyncio.to_thread(legacy_query, n)}
