#!/usr/bin/env python3
"""Measure environment steps per second as the number of parallel envs N grows.

Steps N gymnax (or Craftax) environments in lockstep with random actions inside one
jitted lax.scan, the same structure a PureJaxRL training loop uses, and reports
throughput for each N. The point where doubling N stops raising steps/s by much is
the hardware's saturation point: past it, more envs buy bigger batches but no extra
speed, so wall-clock per update grows linearly.

Only environment cost is measured (no network). Add the policy's forward cost with
--hidden to see how much the network eats into throughput.

Usage
  python throughput_bench.py --env CartPole-v1 --n 64 256 1024 4096 --steps 128
  python throughput_bench.py --env Craftax-Symbolic-v1 --n 256 1024 4096 --steps 64
  python throughput_bench.py --env CartPole-v1 --n 1024 --hidden 256 256
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings

import jax
import jax.numpy as jnp


warnings.filterwarnings("ignore", message="Explicitly requested dtype")  # gymnax samples int64 actions


def make_env(name: str):
    if name.startswith("Craftax"):
        try:
            from craftax.craftax_env import make_craftax_env_from_name
        except ImportError as e:
            raise SystemExit("craftax not installed: pip install craftax") from e
        env = make_craftax_env_from_name(name, auto_reset=True)
        return env, env.default_params
    import gymnax
    return gymnax.make(name)


def mlp_cost(obs_dim: int, hidden: list[int], n_actions: int, key):
    """A random MLP whose forward pass stands in for the policy."""
    sizes = [obs_dim, *hidden, n_actions]
    keys = jax.random.split(key, len(sizes) - 1)
    params = [jax.random.normal(k, (a, b)) / jnp.sqrt(a) for k, a, b in zip(keys, sizes[:-1], sizes[1:])]

    def fwd(x):
        for w in params[:-1]:
            x = jnp.tanh(x @ w)
        return x @ params[-1]
    return fwd


def bench(env, params, n: int, steps: int, hidden: list[int] | None, reps: int) -> float:
    key = jax.random.key(0)
    obs, state = jax.vmap(env.reset, in_axes=(0, None))(jax.random.split(key, n), params)
    obs_dim = int(obs.reshape(n, -1).shape[1])
    space = env.action_space(params)
    n_act = getattr(space, "n", 1)
    policy = mlp_cost(obs_dim, hidden, n_act, key) if hidden else None

    def one_step(carry, _):
        key, obs, state = carry
        key, k_act, k_step = jax.random.split(key, 3)
        if policy is not None:
            logits = policy(obs.reshape(n, -1).astype(jnp.float32))
            action = jax.random.categorical(k_act, logits) if hasattr(space, "n") else jnp.tanh(logits[:, 0])
        else:
            action = jax.vmap(space.sample)(jax.random.split(k_act, n))
        obs, state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
            jax.random.split(k_step, n), state, action, params)
        return (key, obs, state), reward.sum()

    @jax.jit
    def rollout(key, obs, state):
        (key, obs, state), r = jax.lax.scan(one_step, (key, obs, state), None, length=steps)
        return key, obs, state, r.sum()

    out = rollout(key, obs, state)
    jax.block_until_ready(out)  # compile + warm up
    t0 = time.perf_counter()
    for _ in range(reps):
        out = rollout(*out[:3])
    jax.block_until_ready(out)
    dt = (time.perf_counter() - t0) / reps
    return n * steps / dt


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", default="CartPole-v1")
    p.add_argument("--n", type=int, nargs="+", default=[64, 256, 1024, 4096])
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--hidden", type=int, nargs="*", help="add an MLP policy forward of these widths")
    args = p.parse_args()

    env, params = make_env(args.env)
    print(f"backend={jax.default_backend()} env={args.env} steps/rollout={args.steps}")
    print(f"{'N':>8}  {'steps/s':>12}  {'gain vs prev':>11}")
    prev = None
    for n in sorted(args.n):
        sps = bench(env, params, n, args.steps, args.hidden, args.reps)
        gain = f"{sps / prev:10.2f}x" if prev else " " * 11
        print(f"{n:>8}  {sps:12.3g}  {gain}")
        prev = sps
    print("Saturation: where the gain column falls toward 1.0x as N grows (compare against the ratio of the N values).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
