"""obslog -- drop-in observability plumbing for Python applications and ML runs.

    from obslog.setup import configure
    from obslog.context import log_context, new_request_id
    from obslog.wandb_run import start_run, log_checkpoint

Nothing here imports wandb at module load, so the logging half works in a
project that has never heard of W&B.
"""

__all__ = ["context", "json_formatter", "redact", "setup", "wandb_run"]
