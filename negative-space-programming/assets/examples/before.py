"""The same module, before negative space. Every function below is 'correct',
and every function type-checks: types and checks are different tools.

Run the audit on this file to see what it flags:
    python scripts/audit_negative_space.py assets/examples/before.py
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Protocol, TypedDict, cast


class Response(Protocol):
    status: int
    body: bytes | None


class HttpClient(Protocol):
    def get(self, url: str) -> Response: ...


class Model(Protocol):
    def load_state_dict(self, state: dict[str, object], *, strict: bool) -> object: ...


class Checkpoint(TypedDict):
    weights: dict[str, object]
    step: int


def split_indices(n: int, train_frac: float) -> tuple[list[int], list[int]]:
    k = int(n * train_frac)
    idx = list(range(n))
    random.shuffle(idx)
    return idx[:k], idx[k:]


def normalize(rows: list[list[float]], stats: dict[str, list[float]]) -> list[list[float]]:
    out: list[list[float]] = []
    for r in rows:
        out.append([(v - m) / s for v, m, s in zip(r, stats["mean"], stats["std"])])
    return out


def fetch_with_retry(client: HttpClient, url: str) -> bytes | None:
    while True:
        try:
            response = client.get(url)
            if response.status == 200:
                return response.body
        except Exception:
            pass
        time.sleep(1)


def load_checkpoint(path: str, model: Model) -> int:
    state = _read(path)
    model.load_state_dict(state["weights"], strict=False)
    return state["step"]


def _read(path: str) -> Checkpoint:
    # cast() is an unchecked claim: the checker believes it, nothing verifies it.
    return cast("Checkpoint", _deserialize(Path(path).read_bytes()))


def _deserialize(blob: bytes) -> object:
    raise NotImplementedError  # stand-in for the project's real loader


def running_mean(values: list[float], window: int) -> list[float]:
    out: list[float] = []
    total = 0.0
    for i, v in enumerate(values):
        total += v
        if i >= window:
            total -= values[i - window]
        out.append(total / min(i + 1, window))
    return out
