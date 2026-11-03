"""Service track: request-scoped logging with no web framework required.

Simulates three concurrent requests through a thread pool to show the part that
actually matters: every log record emitted while handling a request carries that
request's id, without a single function taking a `request_id` argument.

    LOG_FORMAT=json python examples/service_example.py
    LOG_FORMAT=console python examples/service_example.py

The same three moves work in FastAPI/Flask/Django middleware -- the only change
is where `new_request_id()` gets called and where the timing is measured.
"""

from __future__ import annotations

import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor

from obslog.context import log_context, new_request_id
from obslog.setup import configure

logger = logging.getLogger(__name__)
access_logger = logging.getLogger("access")


def charge_card(amount_cents: int, token: str) -> str:
    """A deep call that logs without being told which request it belongs to."""
    logger.debug("charging %d cents", amount_cents)
    time.sleep(random.uniform(0.01, 0.05))
    if amount_cents > 50_000:
        raise ValueError("amount exceeds per-transaction limit")
    # `token` is deliberately never logged. The RedactingFilter would catch it
    # if it were passed via extra=, but not logging it is the actual rule.
    return f"ch_{random.randrange(16**8):08x}"


def handle_request(path: str, amount_cents: int) -> int:
    """One request. Returns an HTTP-ish status code."""
    request_id = new_request_id()
    started = time.perf_counter()

    # Everything logged inside this block -- including inside charge_card, and
    # including third-party libraries -- gets request_id attached.
    with log_context(request_id=request_id, path=path):
        logger.info("request received")
        try:
            charge_id = charge_card(amount_cents, token="sk-live-do-not-log-me")
        except ValueError:
            # .exception() attaches the traceback. Never `logger.error(str(e))`:
            # the string is the one part of an exception you can reconstruct.
            logger.exception("payment rejected")
            status = 400
        else:
            logger.info("payment captured", extra={"charge_id": charge_id})
            status = 200

        duration_ms = (time.perf_counter() - started) * 1000
        # One access record per request, with the fields you would group by.
        # Level tracks the outcome so a 5xx is findable without parsing status.
        access_logger.log(
            logging.WARNING if status >= 400 else logging.INFO,
            "%s %s -> %d in %.1fms",
            "POST",
            path,
            status,
            duration_ms,
            extra={"status": status, "duration_ms": round(duration_ms, 1)},
        )
        return status


def main() -> None:
    configure(service="checkout-api", version="1.4.2", env="dev", level="DEBUG")
    requests = [("/charge", 1_200), ("/charge", 99_000), ("/charge", 4_500)]
    with ThreadPoolExecutor(max_workers=3) as pool:
        statuses = list(pool.map(lambda r: handle_request(*r), requests))
    logger.info("processed %d requests: %s", len(statuses), statuses)


if __name__ == "__main__":
    main()
