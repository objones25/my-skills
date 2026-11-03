"""ML track: a training loop instrumented the way the skill prescribes.

Runs with no GPU, no PyTorch, and no W&B account -- the "model" is a scalar
gradient-descent toy so the instrumentation is the only thing on display.

    WANDB_MODE=offline python examples/train_example.py

What to copy out of this file:

1. `configure()` once in the entry point, then `logging.getLogger(__name__)`.
2. The whole config dict goes into `wandb.init(config=...)`.
3. `define_metric` sets each metric's x-axis, so `run.log` is never passed
   `step=` -- out-of-order steps are dropped without an exception.
4. Per-step metrics go to W&B; per-phase events go to the log. The log tells you
   the run started, resumed, hit NaN, and finished; W&B tells you the loss curve.
5. The checkpoint is an artifact, so the weights and the run that made them stay
   attached to each other.
"""

from __future__ import annotations

import argparse
import logging
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from obslog.context import log_context
from obslog.setup import configure, install_excepthook
from obslog.wandb_run import log_checkpoint, start_run

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainConfig:
    lr: float = 0.1
    epochs: int = 5
    steps_per_epoch: int = 20
    seed: int = 0
    log_every: int = 5
    project: str = "obslog-demo"


def make_batch(rng: random.Random) -> tuple[float, float]:
    """Toy data: y = 3x + noise."""
    x = rng.uniform(-1.0, 1.0)
    return x, 3.0 * x + rng.gauss(0.0, 0.05)


def train(cfg: TrainConfig, out_dir: Path) -> float:
    rng = random.Random(cfg.seed)
    weight = 0.0

    with start_run(
        project=cfg.project,
        name=f"lr{cfg.lr}-seed{cfg.seed}",
        config=asdict(cfg),
        job_type="train",
        group=f"seed-sweep-lr{cfg.lr}",
        tags=["demo"],
        # Each metric declares its own x-axis. train/* moves per step,
        # val/* per epoch, and W&B plots them on the right axis without
        # anyone passing step= to log().
        step_metrics={
            "train/loss": "train/step",
            "train/grad_norm": "train/step",
            "val/loss": "epoch",
        },
        summary_goals={"val/loss": "min", "train/loss": "min"},
    ) as run:
        logger.info(
            "training started: %d epochs x %d steps, lr=%s",
            cfg.epochs,
            cfg.steps_per_epoch,
            cfg.lr,
        )
        global_step = 0
        best_val = math.inf

        for epoch in range(cfg.epochs):
            # Bind epoch onto every log record emitted below, so the log lines
            # carry it without every f-string repeating it.
            with log_context(epoch=epoch):
                running = 0.0
                for _ in range(cfg.steps_per_epoch):
                    x, y = make_batch(rng)
                    pred = weight * x
                    loss = (pred - y) ** 2
                    grad = 2 * (pred - y) * x
                    weight -= cfg.lr * grad
                    running += loss
                    global_step += 1

                    if not math.isfinite(loss):
                        # A real failure gets a log record with a traceback and
                        # a marker on the run, not a silent NaN curve.
                        logger.error(
                            "non-finite loss at step %d, aborting", global_step
                        )
                        run.summary["diverged_at_step"] = global_step
                        raise FloatingPointError(f"loss={loss}")

                    run.log(
                        {
                            "train/loss": loss,
                            "train/grad_norm": abs(grad),
                            "train/step": global_step,
                        }
                    )

                train_loss = running / cfg.steps_per_epoch
                val_loss = sum(
                    (weight * x - y) ** 2
                    for x, y in (make_batch(rng) for _ in range(50))
                ) / 50

                run.log({"val/loss": val_loss, "epoch": epoch, "lr": cfg.lr})
                # %-args, not f-strings: the string is built only if INFO passes.
                logger.info(
                    "epoch %d done: train_loss=%.5f val_loss=%.5f weight=%.4f",
                    epoch,
                    train_loss,
                    val_loss,
                    weight,
                )

                if val_loss < best_val:
                    best_val = val_loss
                    ckpt = out_dir / "model.txt"
                    ckpt.write_text(f"{weight}\n")
                    log_checkpoint(
                        run,
                        ckpt,
                        name="demo-model",
                        metadata={"epoch": epoch, "val_loss": val_loss},
                        aliases=["best"],
                    )

        # Summary holds the answer you compare runs by. Left alone it is the
        # last logged value, which is not the same as the best one.
        run.summary["best_val_loss"] = best_val
        run.summary["final_weight"] = weight
        logger.info("training finished: best_val_loss=%.5f", best_val)
        return best_val


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("./_demo_out"))
    args = parser.parse_args()

    # Entry point configures logging exactly once, before anything else logs.
    configure(level="INFO", fmt="console", service="train-demo")
    install_excepthook()

    args.out.mkdir(parents=True, exist_ok=True)
    cfg = TrainConfig(lr=args.lr, epochs=args.epochs, seed=args.seed)
    train(cfg, args.out)


if __name__ == "__main__":
    main()
