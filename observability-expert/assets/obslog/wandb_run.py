"""W&B run lifecycle helper: start correctly, degrade gracefully, close always.

Vocabulary, because W&B reuses ordinary words for specific things:

- **run**: one execution of your script. Everything you log belongs to exactly
  one run. `wandb.init()` creates it; `run.finish()` closes it.
- **config**: the run's immutable inputs -- hyperparameters, seed, data version.
  Written once at `init`. Attempting to change a key later raises unless you
  pass `allow_val_change=True`.
- **history**: the time series. Every `run.log({...})` appends one step.
- **summary**: one value per key, the "final answer" for the run. Auto-set to
  the last logged value; override it for things like best-so-far.
- **artifact**: a versioned file or directory (checkpoint, dataset, eval set)
  tied to the run that produced it, so the run and the file cannot drift apart.
- **entity**: the user or team the project belongs to.

The wrapper exists because W&B is a network service in the middle of a training
loop. If the auth token is missing or the network is down, `wandb.init()` should
not be the reason a 6-hour run does not start. On its own, `wandb.init()` raises
`UsageError` with no key configured and `CommError` (after `init_timeout`,
90 s by default) when the server is unreachable. `start_run` falls back to
offline mode in both cases (data on disk, syncable later with `wandb sync`) and
says so in the logs.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from obslog.context import bind_log_context

logger = logging.getLogger(__name__)

def _has_credentials() -> bool:
    """Ask W&B itself whether a key is configured, without prompting.

    Grepping ~/.netrc for a hostname breaks whenever W&B changes its default
    server: wandb 0.31.0 moved it from api.wandb.ai to api.forge.coreweave.com,
    and a fresh `wandb login` now writes only the new host. `wandb.login(
    prompt=False)` (0.30.0+) reads env vars, netrc and settings the way
    `wandb.init` will, and returns False instead of asking for a key.
    """
    import wandb

    try:
        return bool(wandb.login(prompt=False, verify=False))
    except TypeError:  # wandb < 0.30: no `prompt` argument
        return bool(os.getenv("WANDB_API_KEY")) or wandb.api.api_key is not None


def resolve_mode(requested: str | None = None) -> str:
    """Pick a W&B mode that will actually work here.

    online   -> live sync (needs credentials)
    offline  -> writes to ./wandb/offline-run-*, sync later with `wandb sync`
    disabled -> every run method becomes a no-op; use in unit tests
    """
    if requested:
        return requested
    env_mode = os.getenv("WANDB_MODE")
    if env_mode:
        return env_mode
    if os.getenv("WANDB_API_KEY") or _has_credentials():
        return "online"
    return "offline"


@contextmanager
def start_run(
    *,
    project: str,
    config: Mapping[str, Any],
    name: str | None = None,
    group: str | None = None,
    job_type: str | None = None,
    tags: list[str] | None = None,
    run_id: str | None = None,
    resume: str | None = None,
    mode: str | None = None,
    entity: str | None = None,
    dir: str | None = None,
    step_metrics: Mapping[str, str] | None = None,
    summary_goals: Mapping[str, str] | None = None,
) -> Iterator[Any]:
    """Start a run, yield it, and close it exactly once.

    The context manager form is the one to use: on a normal exit it finishes the
    run, and on an exception it calls `finish(exit_code=1)` so the run is marked
    **failed** at that moment. Without it, an exception that is caught further
    up (a sweep agent, a retry loop, a notebook cell) leaves the run open, and
    it is later recorded as *finished* with exit code 0.

    Args:
        project: W&B project name. Required -- without it runs land in
            "uncategorized" and are effectively lost.
        config: hyperparameters and inputs. Pass the whole config object
            (`vars(args)`, `dataclasses.asdict(cfg)`, an OmegaConf container),
            not a hand-picked subset: the field you leave out is the one that
            explains the anomaly three weeks later.
        run_id / resume: pass a stable id (and resume="allow") to make a
            preempted job continue the same run instead of starting a new one.
            Online only: in offline mode `run.resumed` is False and `run.step`
            restarts at 0, so take the resume point from your checkpoint.
        step_metrics: metric name -> x-axis metric name, e.g.
            {"val/loss": "epoch"}. Lets each metric carry its own x-axis so you
            never have to pass `step=` to `log()`.
        summary_goals: metric name -> "min" | "max" | "last" | "mean", so the
            run's summary holds the best value rather than the last one.
    """
    import wandb  # imported lazily so `import obslog` never needs wandb

    resolved_mode = resolve_mode(mode)
    if resolved_mode == "offline" and mode is None:
        logger.warning(
            "no W&B credentials found; running offline. sync later with "
            "`wandb sync wandb/offline-run-*`"
        )

    init_kwargs: dict[str, Any] = dict(
        entity=entity,
        name=name,
        group=group,
        job_type=job_type,
        tags=tags,
        id=run_id,
        resume=resume,
        dir=dir,
    )
    try:
        run = wandb.init(
            project=project, config=dict(config), mode=resolved_mode, **init_kwargs
        )
    except wandb.errors.AuthenticationError:
        # A revoked or wrong key. AuthenticationError subclasses CommError, so
        # without this clause a bad key would silently degrade to offline.
        raise
    except wandb.errors.CommError:
        # Credentials exist but the server is unreachable. wandb.init raises
        # CommError after `init_timeout` (90 s by default) rather than falling
        # back on its own, so do the fallback here -- but only when the caller
        # did not ask for a specific mode.
        if mode is not None or resolved_mode != "online":
            raise
        logger.warning(
            "W&B server unreachable; running offline. sync later with "
            "`wandb sync wandb/offline-run-*`",
            exc_info=True,
        )
        resolved_mode = "offline"
        run = wandb.init(
            project=project, config=dict(config), mode=resolved_mode, **init_kwargs
        )

    for metric, step_metric in (step_metrics or {}).items():
        run.define_metric(metric, step_metric=step_metric)
    for metric, goal in (summary_goals or {}).items():
        run.define_metric(metric, summary=goal)

    # run.url is None offline -- guard rather than formatting a None into logs.
    bind_log_context(wandb_run_id=run.id)
    logger.info(
        "wandb run started: id=%s name=%s mode=%s url=%s",
        run.id,
        run.name,
        resolved_mode,
        run.url or "(offline)",
    )

    try:
        yield run
    except BaseException:
        # finish(exit_code=1) is what marks the run failed. Logging it here
        # means the training log and the dashboard agree on what happened.
        logger.exception("run %s failed", run.id)
        run.finish(exit_code=1)
        raise
    else:
        run.finish()
        logger.info("wandb run %s finished", run.id)


def log_checkpoint(
    run: Any,
    path: str | Path,
    *,
    name: str = "model-weights",
    metadata: Mapping[str, Any] | None = None,
    aliases: list[str] | None = None,
    artifact_type: str = "model",
) -> Any:
    """Log a checkpoint file or directory as a versioned artifact.

    Prefer this over `run.save(path)`: an artifact is versioned (v0, v1, ...),
    carries metadata, records which run produced it and which runs consumed it,
    and can be fetched later by alias (`model-weights:best`). `run.save` just
    copies bytes into the run's file list with no lineage.

    Note: reading `artifact.version` or `.aliases` off the returned object
    raises `ArtifactNotLoggedError` until the upload completes -- call
    `artifact.wait()` first, and expect it to stay unavailable in offline mode.
    """
    import wandb

    artifact = wandb.Artifact(
        name=name, type=artifact_type, metadata=dict(metadata or {})
    )
    target = Path(path)
    if target.is_dir():
        artifact.add_dir(str(target))
    else:
        artifact.add_file(str(target))
    logged = run.log_artifact(artifact, aliases=aliases)
    logger.info(
        "logged artifact name=%s type=%s aliases=%s from=%s",
        name,
        artifact_type,
        aliases or ["latest"],
        target,
    )
    return logged


def redact_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Drop secret-looking keys before they are written to a run's config.

    A run config is visible to everyone with project access and is exported by
    the public API. An API key that lands there is disclosed, not stored.
    """
    from obslog.redact import REDACTED, is_secret_name

    return {
        key: (REDACTED if is_secret_name(str(key)) else value)
        for key, value in config.items()
    }
