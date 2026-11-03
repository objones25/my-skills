"""The same module, after negative space. Audits clean, and passes pyright strict:

    python scripts/audit_negative_space.py assets/examples/after.py

Note what changed and what did not: no function grew an abstraction, no control
flow moved. The checks encode facts that were already true in the author's head.
The one structural change is at the checkpoint boundary, where `cast()` (a claim
the checker trusts) became `_as_checkpoint()` (a claim the program verifies).
"""

from __future__ import annotations

import random
import time
from http import HTTPStatus
from pathlib import Path
from typing import Protocol, TypedDict, cast

from negative_space import CheckFailed, bounded, check_finite, expect, require

MAX_RETRIES = 5
RETRY_SLEEP_S = 1.0


class Response(Protocol):
    status: int
    body: bytes | None


class HttpClient(Protocol):
    def get(self, url: str) -> Response: ...


class LoadResult(Protocol):
    """What torch's load_state_dict returns: a NamedTuple of two key lists."""

    missing_keys: list[str]
    unexpected_keys: list[str]


class Model(Protocol):
    def load_state_dict(self, state: dict[str, object], *, strict: bool) -> LoadResult: ...


class Checkpoint(TypedDict):
    weights: dict[str, object]
    step: int


def split_indices(n: int, train_frac: float) -> tuple[list[int], list[int]]:
    require(n >= 2, f"need at least 2 samples to split, got {n}")
    require(0.0 < train_frac < 1.0, f"train_frac must be interior, got {train_frac}")

    k = int(n * train_frac)
    require(1 <= k <= n - 1, f"split point {k} leaves an empty side for n={n}")

    idx = list(range(n))
    random.shuffle(idx)
    train, val = idx[:k], idx[k:]

    require(len(train) + len(val) == n, "split lost or duplicated samples")
    require(not (set(train) & set(val)), "train and val overlap")
    require(min(train + val) == 0, "index below range")
    require(max(train + val) == n - 1, "index above range")
    return train, val


def normalize(rows: list[list[float]], stats: dict[str, list[float]]) -> list[list[float]]:
    require(rows, "normalize() on empty rows")
    n_features = len(rows[0])
    mean = expect(stats.get("mean"), "stats has no 'mean'")
    std = expect(stats.get("std"), "stats has no 'std'")
    require(len(mean) == n_features, "mean has wrong feature count")
    require(len(std) == n_features, "std has wrong feature count")
    require(all(s > 0.0 for s in std), "zero or negative std would divide by zero")

    out: list[list[float]] = []
    for r in rows:
        require(len(r) == n_features, f"ragged row: {len(r)} != {n_features}")
        out.append([(v - m) / s for v, m, s in zip(r, mean, std, strict=True)])

    require(len(out) == len(rows), "row count changed during normalization")
    return out


def fetch_with_retry(client: HttpClient, url: str) -> bytes:
    """Bounded retry. Programmer errors crash; the network failing is an
    operating error and gets handled and reported."""
    require(url.startswith(("http://", "https://")), f"not a fetchable url: {url}")

    attempt = 0  # bound before the loop, so "ran zero times" is checkable below
    last_error: Exception | None = None
    for attempt in bounded(range(1, MAX_RETRIES + 1), MAX_RETRIES, name="fetch"):
        try:
            response = client.get(url)
        except (TimeoutError, ConnectionError) as exc:
            last_error = exc  # operating error: retry
        else:
            if response.status == HTTPStatus.OK:
                return expect(response.body, "200 response with no body")
            if 400 <= response.status < 500:  # client error: will not become 200 by waiting
                msg = f"{url} returned {response.status} on attempt {attempt}"
                raise OSError(msg)
            last_error = OSError(f"{url} returned {response.status} on attempt {attempt}")
        time.sleep(RETRY_SLEEP_S)

    require(attempt == MAX_RETRIES, f"retry loop exited early at attempt {attempt}")
    msg = f"{url} failed after {MAX_RETRIES} attempts"
    raise TimeoutError(msg) from last_error


def load_checkpoint(path: str, model: Model, previous_step: int = -1) -> int:
    state = _read(path)
    require(state["step"] > previous_step,
            f"checkpoint step {state['step']} is not after {previous_step}")

    # strict=False so the check is ours: with strict=True torch raises
    # RuntimeError on a mismatch and these two lines could never run.
    result = model.load_state_dict(state["weights"], strict=False)
    require(not result.missing_keys, f"missing weight tensors: {result.missing_keys}")
    require(not result.unexpected_keys,
            f"unexpected weight tensors: {result.unexpected_keys}")
    return state["step"]


def _read(path: str) -> Checkpoint:
    return _as_checkpoint(_deserialize(Path(path).read_bytes()))


def _as_checkpoint(raw: object) -> Checkpoint:
    """Turn an untyped blob into a Checkpoint, checking what cast() only claimed.

    The file was written by our own save path, so a missing key is our bug
    (the read half of a pair assertion) and crashes. A checkpoint from outside
    would get `raise ValueError` instead: same predicate, different origin.
    """
    fields = _as_dict(raw, "checkpoint")
    require("weights" in fields, f"checkpoint missing 'weights': {sorted(map(str, fields))}")
    require("step" in fields, f"checkpoint missing 'step': {sorted(map(str, fields))}")
    step = fields["step"]
    if not isinstance(step, int):
        raise CheckFailed(f"step is {type(step).__name__}, not int")
    weights = {str(k): v for k, v in _as_dict(fields["weights"], "weights").items()}
    return {"weights": weights, "step": step}


def _as_dict(value: object, what: str) -> dict[object, object]:  # nsp: ignore NSP001 one check
    if not isinstance(value, dict):
        raise CheckFailed(f"{what} is {type(value).__name__}, not dict")
    # A cast AFTER a runtime check states a verified fact; a cast INSTEAD of
    # one (before.py) is an unverified claim.
    return cast("dict[object, object]", value)


def running_mean(values: list[float], window: int) -> list[float]:
    require(window >= 1, f"window must be positive, got {window}")
    require(values, "running_mean() on empty values")

    out: list[float] = []
    total = 0.0
    for i, v in enumerate(values):
        check_finite(v, name=f"values[{i}]")
        total += v
        if i >= window:
            total -= values[i - window]
        out.append(total / min(i + 1, window))

    require(len(out) == len(values), "output length diverged from input")
    require(min(out) >= min(values), "mean below the minimum input")
    require(max(out) <= max(values), "mean above the maximum input")
    return out


def _deserialize(blob: bytes) -> object:
    raise NotImplementedError  # stand-in for the project's real loader


__all__ = [
    "CheckFailed",
    "fetch_with_retry",
    "load_checkpoint",
    "normalize",
    "running_mean",
    "split_indices",
]
