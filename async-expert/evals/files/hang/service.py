import asyncio
import signal

signal.signal(signal.SIGUSR1, lambda *_: asyncio.print_call_graph())


async def poll_upstream(name: str) -> None:
    while True:
        await asyncio.sleep(30)


async def main() -> None:
    async with asyncio.TaskGroup() as tg:
        for name in ("orders", "billing", "inventory"):
            tg.create_task(poll_upstream(name), name=f"poll-{name}")


asyncio.run(main())
