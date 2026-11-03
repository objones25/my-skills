"""Small, dependency-light helpers for bounded async I/O.

    from aioutil import build_client, bounded_map_settled, request_with_retry

Requires: Python >= 3.11 (TaskGroup), httpx.
"""

from .bounded import Err, Ok, bounded_map, bounded_map_settled, cpu_map, partition
from .client import FAST_TIMEOUT, SLOW_TIMEOUT, build_client, client_session
from .retry import (
    ALWAYS_RETRYABLE,
    ALWAYS_RETRYABLE_STATUS,
    IDEMPOTENT_METHODS,
    IDEMPOTENT_ONLY_RETRYABLE,
    RETRYABLE_EXCEPTIONS,
    RETRYABLE_STATUS,
    RetriesExhausted,
    RetryPolicy,
    request_with_retry,
    with_retry,
)

__all__ = [
    "ALWAYS_RETRYABLE",
    "ALWAYS_RETRYABLE_STATUS",
    "FAST_TIMEOUT",
    "IDEMPOTENT_METHODS",
    "IDEMPOTENT_ONLY_RETRYABLE",
    "RETRYABLE_EXCEPTIONS",
    "RETRYABLE_STATUS",
    "SLOW_TIMEOUT",
    "Err",
    "Ok",
    "RetriesExhausted",
    "RetryPolicy",
    "bounded_map",
    "bounded_map_settled",
    "build_client",
    "client_session",
    "cpu_map",
    "partition",
    "request_with_retry",
    "with_retry",
]
