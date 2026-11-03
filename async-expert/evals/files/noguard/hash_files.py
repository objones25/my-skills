import asyncio
import hashlib
import sys
from concurrent.futures import ProcessPoolExecutor


def digest(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


async def main(paths: list[str]) -> None:
    loop = asyncio.get_running_loop()
    with ProcessPoolExecutor() as pool:
        results = await asyncio.gather(
            *[loop.run_in_executor(pool, digest, p) for p in paths], return_exceptions=False
        )
    for p, h in zip(paths, results):
        print(h[:16], p)


asyncio.run(main(sys.argv[1:]))
