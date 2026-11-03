"""Batch arithmetic and compute-cost estimates for massively parallel on-policy RL.

State the batch arithmetic whenever num_envs, steps per env or minibatches change (the G1
project's rule). numpy-free, standard library only.

    python batch_math.py                                   # G1 shipped config at 64..16384 envs
    python batch_math.py --envs 4096 --steps 24 --minibatches 4 --epochs 5
    python batch_math.py --envs 4096 --iters-per-min 57 --minutes 45 --price-per-hour 1.79

Definitions
    batch B            num_envs * steps_per_env   (samples per PPO iteration)
    minibatch          B / num_mini_batches
    grad steps / iter  epochs * num_mini_batches
    sim seconds / iter steps_per_env * policy_dt  (per env)
    sample reuse       epochs (each sample is seen once per epoch)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class PPOBatch:
    num_envs: int
    steps_per_env: int = 24
    num_mini_batches: int = 4
    num_epochs: int = 5
    policy_dt: float = 0.02

    def __post_init__(self) -> None:
        for name in ("num_envs", "steps_per_env", "num_mini_batches", "num_epochs"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.batch % self.num_mini_batches:
            raise ValueError(f"batch {self.batch} not divisible by num_mini_batches {self.num_mini_batches}")
        if self.policy_dt <= 0:
            raise ValueError("policy_dt must be positive")

    @property
    def batch(self) -> int:
        return self.num_envs * self.steps_per_env

    @property
    def minibatch(self) -> int:
        return self.batch // self.num_mini_batches

    @property
    def grad_steps_per_iter(self) -> int:
        return self.num_epochs * self.num_mini_batches

    @property
    def sim_seconds_per_iter(self) -> float:
        return self.steps_per_env * self.policy_dt

    def total_env_steps(self, iterations: int) -> int:
        return self.batch * iterations

    def row(self) -> dict[str, int | float]:
        return {"num_envs": self.num_envs, "steps/env": self.steps_per_env, "batch": self.batch,
                "minibatches": self.num_mini_batches, "minibatch": self.minibatch,
                "grad steps/iter": self.grad_steps_per_iter}


def cost(iters_per_min: float, minutes: float, price_per_hour: float, batch: int) -> dict[str, float]:
    """Iterations, env steps and dollars for a timed run (training clock only)."""
    if min(iters_per_min, minutes, price_per_hour) < 0:
        raise ValueError("inputs must be non-negative")
    iters = iters_per_min * minutes
    return {"iterations": iters, "env_steps": iters * batch, "dollars": minutes / 60 * price_per_hour}


def table(rows: list[dict]) -> str:
    keys = list(rows[0])
    widths = [max(len(k), *(len(f"{r[k]:,}" if isinstance(r[k], int) else str(r[k])) for r in rows)) for k in keys]
    fmt = lambda v: f"{v:,}" if isinstance(v, int) else str(v)  # noqa: E731
    lines = [" | ".join(k.rjust(w) for k, w in zip(keys, widths))]
    lines.append("-+-".join("-" * w for w in widths))
    lines += [" | ".join(fmt(r[k]).rjust(w) for k, w in zip(keys, widths)) for r in rows]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--envs", type=int, nargs="+", default=[64, 256, 1024, 4096, 16384])
    p.add_argument("--steps", type=int, default=24)
    p.add_argument("--minibatches", type=int, default=4)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--iters-per-min", type=float)
    p.add_argument("--minutes", type=float, default=45.0)
    p.add_argument("--price-per-hour", type=float, default=1.79)
    a = p.parse_args()
    batches = [PPOBatch(n, a.steps, a.minibatches, a.epochs, a.dt) for n in a.envs]
    print(table([b.row() for b in batches]))
    print(f"sim seconds per env per iteration: {batches[0].sim_seconds_per_iter:g}")
    if a.iters_per_min is not None:
        for b in batches:
            c = cost(a.iters_per_min, a.minutes, a.price_per_hour, b.batch)
            print(f"N={b.num_envs}: {c['iterations']:,.0f} iters, {c['env_steps']:,.0f} env steps, ${c['dollars']:.2f}")


if __name__ == "__main__":
    main()
