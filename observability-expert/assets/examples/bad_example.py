"""Every gate the audit enforces, violated once, on purpose.

`python scripts/audit_observability.py assets/examples/bad_example.py` should
report one finding per code. Keep this file broken -- it is the audit's own test
fixture. Do not import it.
"""

from __future__ import annotations

import logging
import logging.config

import wandb

# LOG005: configuring logging at import time in a non-entry-point module.
# LOG010: no disable_existing_loggers key, so every logger built before this
# call is silenced.
logging.config.dictConfig(
    {
        "version": 1,
        "handlers": {"h": {"class": "logging.StreamHandler"}},
        "root": {"handlers": ["h"]},
    }
)

# LOG003: no name means the root logger.
logger = logging.getLogger()

# WB005: credential in source control.
WANDB_API_KEY = "abcd1234abcd1234abcd1234abcd1234abcd1234"


def train(order_id: int, api_key: str) -> None:
    # WB001 + WB002 + WB003: no project, no config, no finish.
    run = wandb.init(name="run-1")

    # LOG001: f-string builds the message even when DEBUG is off.
    logger.debug(f"starting order {order_id}")

    # LOG009: warn() is the deprecated spelling.
    logger.warn("low disk")

    # LOG002: the module-level function logs to the root logger.
    logging.info("training started")

    # LOG004: print has no level, no timestamp and no routing.
    print("epoch 0 done")

    # LOG008: the key is written into a place other people can read.
    logger.info("authenticating with %s", api_key)

    # WB006: an explicit step that goes backwards is dropped without an error.
    run.log({"loss": 0.5}, step=order_id)

    # WB004: the implicit global run raises once any run has finished.
    wandb.log({"loss": 0.4})

    try:
        risky()
    except ValueError as exc:
        # LOG006 + LOG011: no traceback, and only the exception's str().
        logger.error(str(exc))

    try:
        risky()
    except KeyError:
        # LOG007: the failure leaves no trace at all.
        pass


def risky() -> None:
    raise ValueError("boom")
